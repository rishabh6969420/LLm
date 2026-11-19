"""
Dataset Sample Inspector for Strikler.
Inspects multimodal training data: images, prompts, targets, bounding boxes, and shades.
Usage:
  python inspect_dataset.py --num_samples 5
"""

import os
import argparse
from data.dataset import VisionColorDataset
from data.tokenizer_utils import VLMTokenizer
from data.synthetic_generator import generate_synthetic_sample
from engine.visualize import draw_detections_and_palette


def inspect(num_samples: int = 3, save_visuals: bool = True):
    print("=" * 70)
    print("🔍 STRIKLER MULTIMODAL TRAINING DATA INSPECTOR")
    print("=" * 70)

    tokenizer = VLMTokenizer()
    dataset = VisionColorDataset(num_samples=num_samples, tokenizer=tokenizer, seed=42)

    os.makedirs("data/samples_preview", exist_ok=True)

    for i in range(num_samples):
        sample = dataset[i]
        annotations = sample["annotations"]

        print(f"\n📦 [SAMPLE #{i+1}]")
        print(f"  • Task Prompt  : {sample['prompt_text']}")
        print(f"  • Model Target : {sample['target_text']}")
        print(f"  • Input Tensor : {tuple(sample['image'].shape)} (ImageNet normalized float32)")
        print(f"  • Token Sequence Length: {len(sample['input_ids'])} tokens")
        print("  • Ground-Truth Objects:")

        dets = []
        for ann in annotations:
            y1, x1, y2, x2 = ann["bbox_pixels"]
            ny1, nx1, ny2, nx2 = ann["bbox_norm"]
            print(f"    - Object: {ann['shape']:10s} | Shade: {ann['shade']:16s} | Hex: {ann['hex']} | BBox: [{y1}, {x1}, {y2}, {x2}] (Norm: <loc_{ny1}><loc_{nx1}><loc_{ny2}><loc_{nx2}>)")

            dets.append({
                "bbox": [y1, x1, y2, x2],
                "description": f"{ann['shade']} {ann['shape']}",
                "shade": ann["shade"],
                "color": ann["shade"],
                "hex": ann["hex"],
                "rgb": ann.get("rgb", (128, 128, 128))
            })

        if save_visuals:
            out_img = f"data/samples_preview/sample_{i+1}.png"
            raw_sample = generate_synthetic_sample(image_size=256, min_objects=len(annotations), max_objects=len(annotations))
            draw_detections_and_palette(raw_sample["image"], dets, output_path=out_img)
            print(f"  [✓] Rendered annotated preview: {out_img}")

    print("\n" + "=" * 70)
    print(f"[✓] Inspected {num_samples} samples. Visual preview images saved in data/samples_preview/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect dataset samples")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to inspect")
    args = parser.parse_args()

    inspect(num_samples=args.num_samples)
