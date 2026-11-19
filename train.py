"""
Main Training Entrypoint for Strikler.
Supports Staged Scalable Training:
  • 100M (~102 Million parameters) - Active staged training piece
  • 300M (~303 Million parameters)
  • 1B   (~1.1 Billion parameters)
  • 3B   (~3.05 Billion parameters) - Full flagship scale for multi-day DGX run
Usage:
  python train.py --model_size 100M --steps 2000 --batch_size 32
"""

import argparse
import torch

from config import get_model_preset, TrainingConfig
from model.vision_llm import StriklerVLM
from data.tokenizer_utils import VLMTokenizer
from data.dataset import get_dataloaders
from engine.trainer import VLMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Strikler for Object & Color Detection")
    parser.add_argument(
        "--model_size",
        type=str,
        default="1B",
        choices=["100M", "300M", "1B", "3B"],
        help="Model scale preset (default: 1B)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["quick_test", "standard", "dgx_2day"],
        help="Training duration profile",
    )
    parser.add_argument("--steps", type=int, default=None, help="Custom step count (overrides mode)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--total_samples", type=int, default=100000, help="Total sample pool to partition")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(f"🚀 STRIKLER VLM: STAGED TRAINING RUN [{args.model_size.upper()} PIECE]")
    print("=" * 70)

    # 1. Initialize Tokenizer
    tokenizer = VLMTokenizer()

    # 2. Load Selected Model Preset (100M, 300M, 1B, 3B)
    vision_cfg, projector_cfg, llm_cfg = get_model_preset(args.model_size)

    # 3. Setup Steps Profile
    if args.steps:
        total_steps = args.steps
    elif args.mode == "quick_test":
        total_steps = 100
    elif args.mode == "standard":
        total_steps = 3000
    else:  # dgx_2day
        total_steps = 100000

    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=total_steps,
        save_every_steps=500 if total_steps >= 2000 else 50,
        eval_every_steps=250 if total_steps >= 2000 else 25,
    )

    # 4. Instantiate Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Instantiating Strikler-{args.model_size.upper()} on device: {device}...")
    model = StriklerVLM(vision_cfg, projector_cfg, llm_cfg)

    # Accurate Parameter Counting
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Model Scale: {args.model_size.upper()}")
    print(f"[*] Total Model Parameters: {total_params / 1e6:.2f} Million ({total_params:,} parameters)")
    print(f"[*] Trainable Parameters  : {trainable_params / 1e6:.2f} Million")

    # 5. Partition Data: 75% Train / 15% Val / 10% Test
    print(f"\n[*] Partitioning dataset pool ({args.total_samples:,} samples) into 75% Train / 15% Val / 10% Test...")
    train_loader, val_loader, test_loader, stats = get_dataloaders(
        total_samples=args.total_samples,
        batch_size=train_cfg.batch_size,
        tokenizer=tokenizer,
        num_workers=4 if torch.cuda.is_available() else 0,
    )

    print(f"  • Training Split (75%)   : {stats['train']:,} samples ({stats['train_pct']:.1f}%)")
    print(f"  • Validation Split (15%) : {stats['val']:,} samples ({stats['val_pct']:.1f}%)")
    print(f"  • Testing Split (10%)    : {stats['test']:,} samples ({stats['test_pct']:.1f}%)")

    # 6. Launch Trainer
    trainer = VLMTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        config=train_cfg,
        tokenizer=tokenizer,
    )

    print(f"\n[*] Starting staged training run on Strikler-{args.model_size.upper()} for {total_steps} steps...")
    trainer.train(max_steps=total_steps)


if __name__ == "__main__":
    main()
