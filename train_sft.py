"""
Supervised Fine-Tuning (SFT) Engine for Strikler-500M.
Transitions the model from pretraining into a fine-grained, conversational vision-language assistant.
Key Features:
  1. Prompt Loss Masking (labels=-100 for human prompts, trains strictly on assistant answers)
  2. Multi-Task SFT Data (general detection, color QA, spatial queries, garment separation)
  3. Low Learning Rate (1e-5) with Cosine Schedule to preserve visual pretraining features
  4. BF16 Mixed Precision on NVIDIA DGX Spark
"""

import os
import math
import time
import json
import random
import argparse
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np

from config import get_model_preset
from model.vision_llm import StriklerVLM
from data.tokenizer_utils import VLMTokenizer


class SFTDataset(Dataset):
    """
    Dataset loader for multi-task conversational SFT.
    Applies prompt loss masking so loss is computed strictly on response tokens.
    """
    def __init__(self, records: List[Dict[str, Any]], tokenizer: VLMTokenizer, max_len: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, rel_path: str) -> str:
        candidates = [
            rel_path,
            os.path.join("data/user_dataset", rel_path),
            os.path.join("data", rel_path)
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return rel_path

    def _preprocess_image(self, img_path: str) -> torch.Tensor:
        try:
            raw_img = Image.open(img_path).convert("RGB")
            resized_img = raw_img.resize((224, 224))
            arr = np.array(resized_img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensor = (tensor - self.mean) / self.std
            return tensor
        except Exception:
            # Return blank normalized tensor if corrupted
            return torch.zeros((3, 224, 224), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.records[idx]
        img_path = self._resolve_image_path(item.get("image_file", ""))
        img_tensor = self._preprocess_image(img_path)

        conversations = item.get("conversations", [])
        human_text = ""
        gpt_text = ""
        for turn in conversations:
            if turn.get("from") == "human":
                human_text = turn.get("value", "")
            elif turn.get("from") == "gpt":
                gpt_text = turn.get("value", "")

        # Format turns
        # Backward compatibility: if human prompt was <detect_all>, keep it clean
        if human_text.strip() == "<detect_all>":
            prompt_str = "<bos><detect_all>"
        else:
            prompt_str = f"<bos>Human: {human_text}\nAssistant: "

        # Tokenize prompt and response
        prompt_ids = self.tokenizer.encode(prompt_str)
        gpt_ids = self.tokenizer.encode(gpt_text)

        # Full sequence
        input_ids = prompt_ids + gpt_ids
        
        # Loss Masking: labels for prompt tokens are -100 (ignored in loss computation)
        labels = [-100] * len(prompt_ids) + gpt_ids

        # Truncate to max_len
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]

        return {
            "image": img_tensor,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def sft_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences to max length in the batch."""
    images = torch.stack([item["image"] for item in batch])
    
    pad_id = 0  # <pad> token ID
    ignore_id = -100

    max_len = max(item["input_ids"].size(0) for item in batch)
    
    batch_input_ids = []
    batch_labels = []

    for item in batch:
        inp = item["input_ids"]
        lbl = item["labels"]
        pad_len = max_len - inp.size(0)

        padded_inp = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)])
        padded_lbl = torch.cat([lbl, torch.full((pad_len,), ignore_id, dtype=torch.long)])

        batch_input_ids.append(padded_inp)
        batch_labels.append(padded_lbl)

    return {
        "image": images,
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels)
    }


def get_cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def train_sft(
    checkpoint_path: str = "checkpoints/vlm_step_50000.pt",
    data_path: str = "data/user_dataset/sft_dataset_final.json",
    output_dir: str = "checkpoints_sft",
    model_size: str = "500M",
    batch_size: int = 16,
    grad_accum_steps: int = 4,
    total_steps: int = 5000,
    lr: float = 1e-5,
    min_lr: float = 1e-6,
    warmup_steps: int = 200,
    save_every: int = 500,
    eval_every: int = 250,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("🚀 STRIKLER-500M: SUPERVISED FINE-TUNING (SFT) ENGINE")
    print("=" * 70)
    print(f"[*] Base Checkpoint      : {checkpoint_path}")
    print(f"[*] SFT Dataset Pool     : {data_path}")
    print(f"[*] Target Model Scale   : Strikler-{model_size}")
    print(f"[*] Effective Batch Size : {batch_size * grad_accum_steps} ({batch_size} x {grad_accum_steps})")
    print(f"[*] Target Steps         : {total_steps:,}")
    print(f"[*] Peak Learning Rate   : {lr:.2e}")
    print(f"[*] Compute Device       : {device} (BF16 Autocast)")
    print("=" * 70)

    # 1. Load Pretrained Checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Base checkpoint not found at: {checkpoint_path}")

    print(f"[*] Loading pretrained weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    v_cfg = ckpt["config"]["vision"]
    p_cfg = ckpt["config"]["projector"]
    l_cfg = ckpt["config"]["llm"]

    model = StriklerVLM(v_cfg, p_cfg, l_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[✓] Model initialized with {total_params / 1e6:.2f}M parameters.")

    # 2. Load SFT Data
    print(f"[*] Reading SFT records from {data_path}...")
    with open(data_path, "r") as f:
        all_records = json.load(f)
    print(f"[✓] Loaded {len(all_records):,} total SFT conversations.")

    # 95% Train / 5% Val Split
    random.shuffle(all_records)
    val_count = min(5000, int(len(all_records) * 0.05))
    train_records = all_records[val_count:]
    val_records = all_records[:val_count]
    print(f"    • Training pool   : {len(train_records):,} conversations")
    print(f"    • Validation pool : {len(val_records):,} conversations")

    tokenizer = VLMTokenizer()
    train_dataset = SFTDataset(train_records, tokenizer)
    val_dataset = SFTDataset(val_records, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=sft_collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=sft_collate_fn
    )

    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    # 4. Training Loop
    print("\n[🎯 STARTING SFT OPTIMIZATION LOOP]")
    model.train()
    step = 0
    running_loss = 0.0
    t_start = time.time()
    train_iter = iter(train_loader)

    while step < total_steps:
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            images = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images, input_ids, labels=labels)
                loss = outputs["loss"] / grad_accum_steps

            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update Learning Rate
        curr_lr = get_cosine_lr(step, total_steps, lr, min_lr, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        optimizer.step()
        step += 1
        running_loss += loss_accum

        # Periodic Log
        if step % 10 == 0 or step == total_steps:
            log_steps = 10 if step % 10 == 0 else (step % 10)
            avg_loss = running_loss / max(1, log_steps)
            running_loss = 0.0
            elapsed = time.time() - t_start
            speed = log_steps / max(1e-4, elapsed)
            t_start = time.time()
            vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(
                f"[SFT Step {step:5d}/{total_steps}] "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {curr_lr:.2e} | "
                f"Speed: {speed:.2f} step/s | "
                f"VRAM: {vram_gb:.2f} GB",
                flush=True
            )

        # Periodic Validation
        if step % eval_every == 0 or step == total_steps:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for v_batch in val_loader:
                    v_img = v_batch["image"].to(device)
                    v_ids = v_batch["input_ids"].to(device)
                    v_lbl = v_batch["labels"].to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        v_loss = model(v_img, v_ids, labels=v_lbl)["loss"]
                    val_loss += v_loss.item()
                    val_steps += 1
                    if val_steps >= 20:
                        break
            val_loss /= max(1, val_steps)
            print(f"\n[📊 SFT EVALUATION @ Step {step}] Validation Loss: {val_loss:.4f} (Perplexity: {math.exp(min(val_loss, 20)):.2f})\n")
            model.train()

        # Periodic Checkpoint
        if step % save_every == 0 or step == total_steps:
            save_path = os.path.join(output_dir, f"vlm_sft_step_{step}.pt")
            save_dict = {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": {"vision": v_cfg, "projector": p_cfg, "llm": l_cfg},
            }
            torch.save(save_dict, save_path)
            print(f"[💾 CHECKPOINT SAVED] -> {save_path}\n")

    print("\n" + "=" * 70)
    print("🏆 SUPERVISED FINE-TUNING COMPLETE!")
    print(f"[*] Final SFT Checkpoint: {os.path.join(output_dir, f'vlm_sft_step_{total_steps}.pt')}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Strikler-500M SFT")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vlm_step_50000.pt")
    parser.add_argument("--data", type=str, default="data/user_dataset/sft_dataset_final.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints_sft")
    parser.add_argument("--model_size", type=str, default="500M")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size per forward pass (Stable DGX Spark config)")
    parser.add_argument("--grad_accum", type=int, default=3, help="Gradient accumulation steps (Effective batch size = 60)")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=250)
    args = parser.parse_args()

    train_sft(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_dir=args.output_dir,
        model_size=args.model_size,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        total_steps=args.steps,
        lr=args.lr,
        save_every=args.save_every,
        eval_every=args.eval_every
    )
