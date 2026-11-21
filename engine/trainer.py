"""
High-Performance Trainer for Strikler on NVIDIA GPUs / DGX.
Supports BF16 mixed-precision, gradient accumulation, cosine LR scheduling,
periodic validation loss evaluation, and automatic checkpointing.
"""

import os
import time
import math
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import TrainingConfig
from model.vision_llm import StriklerVLM
from data.tokenizer_utils import VLMTokenizer


class VLMTrainer:
    """Trainer class orchestrating the multi-day / multi-step training loop."""

    def __init__(
        self,
        model: StriklerVLM,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        config: TrainingConfig = None,
        tokenizer: Optional[VLMTokenizer] = None,
        resume_checkpoint: Optional[str] = None,
    ):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.tokenizer = tokenizer or VLMTokenizer()

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

        # Setup Optimizer with weight decay separation
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(optim_groups, lr=self.config.learning_rate, betas=(0.9, 0.95))

        # Precision context
        self.use_bf16 = self.config.bf16 and torch.cuda.is_bf16_supported()
        self.scaler = torch.amp.GradScaler("cuda", enabled=(not self.use_bf16 and torch.cuda.is_available()))

        # Checkpoint Resumption
        self.start_step = 0
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"[*] Resuming from checkpoint: {resume_checkpoint}")
            ckpt = torch.load(resume_checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:
                    print(f"[!] Warning: Could not restore optimizer state: {e}")
            self.start_step = ckpt.get("step", 0)
            print(f"[✓] Resumed state successfully from Step {self.start_step}!")

    def get_lr(self, step: int) -> float:
        """Cosine learning rate schedule with linear warmup."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * (step / max(1, self.config.warmup_steps))
        progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_learning_rate + (self.config.learning_rate - self.config.min_learning_rate) * cosine

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, max_eval_batches: int = 50) -> float:
        """Evaluate loss on validation or test split."""
        self.model.eval()
        total_loss = 0.0
        count = 0

        amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        for i, batch in enumerate(dataloader):
            if i >= max_eval_batches:
                break
            images = batch["image"].to(self.device, non_blocking=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                outputs = self.model(images=images, input_ids=input_ids, labels=labels)
                loss = outputs["loss"]

            total_loss += loss.item()
            count += 1

        self.model.train()
        return total_loss / max(1, count)

    def train(self, max_steps: Optional[int] = None):
        """Main training loop."""
        total_steps = max_steps or self.config.max_steps
        self.model.train()
        step = self.start_step
        running_loss = 0.0
        start_time = time.time()
        data_iter = iter(self.train_dataloader)

        print(f"[*] Starting Strikler Training on {self.device}")
        print(f"[*] Total Steps: {total_steps} | BF16: {self.use_bf16} | Batch Size: {self.config.batch_size}")

        try:
            while step < total_steps:
                self.optimizer.zero_grad()
                accum_loss = 0.0

                for _ in range(self.config.gradient_accumulation_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.train_dataloader)
                        batch = next(data_iter)

                    images = batch["image"].to(self.device, non_blocking=True)
                    input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                    labels = batch["labels"].to(self.device, non_blocking=True)

                    # Mixed Precision Context
                    amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=amp_dtype):
                        outputs = self.model(images=images, input_ids=input_ids, labels=labels)
                        loss = outputs["loss"] / self.config.gradient_accumulation_steps

                    if self.use_bf16:
                        loss.backward()
                    else:
                        self.scaler.scale(loss).backward()

                    accum_loss += loss.item()

                # Gradient Clipping & Optimizer Step
                if self.use_bf16:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    lr = self.get_lr(step)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                    self.optimizer.step()
                else:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    lr = self.get_lr(step)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                step += 1
                running_loss += accum_loss

                # Telemetry logging every 20 steps
                if step % 20 == 0:
                    avg_loss = running_loss / 20.0
                    elapsed = time.time() - start_time
                    steps_per_sec = step / max(elapsed, 1e-4)
                    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                    print(
                        f"[Step {step:6d}/{total_steps}] "
                        f"Train Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                        f"Speed: {steps_per_sec:.2f} step/s | VRAM: {vram_gb:.2f} GB"
                    )
                    running_loss = 0.0

                # Periodic Validation on 15% Validation Split
                if self.val_dataloader and (step % self.config.eval_every_steps == 0 or step == total_steps):
                    val_loss = self.evaluate(self.val_dataloader)
                    print(f"\n[📊 EVALUATION @ Step {step}] Validation Split Loss: {val_loss:.4f}\n")

                # Periodic Checkpoint Saving
                if step % self.config.save_every_steps == 0 or step == total_steps:
                    self.save_checkpoint(step)

        except KeyboardInterrupt:
            print("\n[!] Training paused by user. Saving emergency checkpoint...")
            self.save_checkpoint(step, tag="interrupted")

        # Final evaluation on 10% Testing Split
        if self.test_dataloader:
            test_loss = self.evaluate(self.test_dataloader, max_eval_batches=100)
            print(f"\n[🎯 FINAL TEST EVALUATION] 10% Test Split Loss: {test_loss:.4f}")

        print(f"[✓] Training completed! Checkpoints saved to {self.config.checkpoint_dir}")

    def save_checkpoint(self, step: int, tag: Optional[str] = None):
        """Save model weights and optimizer state."""
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(self.config.checkpoint_dir, f"vlm_step_{step}{suffix}.pt")
        torch.save(
            {
                "step": step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": {
                    "vision": self.model.vision_config,
                    "projector": self.model.projector_config,
                    "llm": self.model.llm_config,
                },
            },
            path,
        )
        print(f"[✓] Saved model checkpoint: {path}")
