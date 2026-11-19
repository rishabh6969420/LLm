"""
Main Training Entrypoint for ChromaDetect-VLM.
Usage:
  python train.py --mode quick_test    # 100 steps to verify pipeline
  python train.py --mode standard      # 5,000 steps (~1-2 hours)
  python train.py --mode dgx_2day      # 100,000 steps (~40-48 hours on NVIDIA DGX Spark)
"""

import argparse
import torch

from config import VisionConfig, ProjectorConfig, LLMConfig, TrainingConfig
from model.vision_llm import ChromaDetectVLM
from data.tokenizer_utils import VLMTokenizer
from data.dataset import get_dataloader
from engine.trainer import VLMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ChromaDetect-VLM for Object & Color Detection")
    parser.add_argument(
        "--mode",
        type=str,
        default="quick_test",
        choices=["quick_test", "standard", "dgx_2day"],
        help="Training duration profile",
    )
    parser.add_argument("--steps", type=int, default=None, help="Custom step count (overrides mode)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("🚀 ChromaDetect-VLM: Object Detection & Color Grounding Model")
    print("=" * 60)

    # Initialize Tokenizer
    tokenizer = VLMTokenizer()

    # Model Configuration
    vision_cfg = VisionConfig(
        image_size=224,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
    )
    projector_cfg = ProjectorConfig(
        vision_dim=768,
        llm_dim=1024,
        hidden_dim=2048,
    )
    llm_cfg = LLMConfig(
        vocab_size=32000,
        hidden_dim=1024,
        num_layers=16,
        num_heads=16,
        num_kv_heads=4,
        intermediate_dim=2816,
        max_seq_len=512,
    )

    # Profile-based Steps
    if args.steps:
        total_steps = args.steps
    elif args.mode == "quick_test":
        total_steps = 100
    elif args.mode == "standard":
        total_steps = 5000
    else:  # dgx_2day
        total_steps = 100000

    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=total_steps,
        save_every_steps=1000 if total_steps >= 5000 else 50,
    )

    print(f"[*] Instantiating VLM Model on device: {'cuda' if torch.cuda.is_available() else 'cpu'}...")
    model = ChromaDetectVLM(vision_cfg, projector_cfg, llm_cfg)

    # Total parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total Model Parameters: {total_params / 1e6:.2f}M | Trainable: {trainable_params / 1e6:.2f}M")

    # Dataloader
    print("[*] Preparing dataset and dataloaders...")
    train_loader = get_dataloader(
        num_samples=max(50000, total_steps * train_cfg.batch_size),
        batch_size=train_cfg.batch_size,
        tokenizer=tokenizer,
        num_workers=4 if torch.cuda.is_available() else 0,
    )

    # Trainer
    trainer = VLMTrainer(
        model=model,
        train_dataloader=train_loader,
        config=train_cfg,
        tokenizer=tokenizer,
    )

    print(f"[*] Launching training profile: '{args.mode}' for {total_steps} steps...")
    trainer.train(max_steps=total_steps)


if __name__ == "__main__":
    main()
