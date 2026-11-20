"""
Evaluation & Progress Inspector for Strikler Checkpoints.
Usage:
  python eval_report.py --checkpoint checkpoints/vlm_step_3000.pt
"""

import os
import argparse
import json
import torch

from config import get_model_preset
from model.vision_llm import StriklerVLM
from data.dataset import get_dataloaders
from data.tokenizer_utils import VLMTokenizer


def generate_report(checkpoint_path: str = "checkpoints/vlm_step_3000.pt"):
    print("=" * 65)
    print("📊 STRIKLER MODEL PROGRESS & EVALUATION REPORT")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Evaluation Device: {device}")
    print(f"[*] Checkpoint Target: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"[!] Checkpoint not found at {checkpoint_path}")
        return

    # Load Model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    v_cfg, p_cfg, l_cfg = get_model_preset("100M")
    model = StriklerVLM(v_cfg, p_cfg, l_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Total Parameters: {total_params / 1e6:.2f}M")
    print(f"[*] Checkpoint Step : {ckpt.get('step', 'N/A')}")

    # Evaluate on Test Split (10% unseen data)
    print("[*] Evaluating on 10% Unseen Test Split...")
    tokenizer = VLMTokenizer()
    _, _, test_loader, stats = get_dataloaders(total_samples=1000, batch_size=16, tokenizer=tokenizer, num_workers=2)

    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 20:
                break
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images, input_ids, labels)
            total_loss += outputs["loss"].item()
            count += 1

    avg_test_loss = total_loss / max(1, count)
    print(f"  [✓] Test Split Loss: {avg_test_loss:.4f}")
    print(f"  [✓] Partition Ratios: 75% Train | 15% Val | 10% Test ({stats['total']} samples benchmarked)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Strikler checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vlm_step_3000.pt", help="Path to checkpoint")
    args = parser.parse_args()

    generate_report(args.checkpoint)

