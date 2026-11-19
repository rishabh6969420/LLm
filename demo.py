"""
Quick End-to-End Demo for ChromaDetect-VLM.
Generates a sample test image with colored objects, tests parsing & visualizer,
and saves the annotated output with color palette.
Usage:
  python demo.py
"""

import os
from data.synthetic_generator import generate_synthetic_sample
from engine.visualize import draw_detections_and_palette
from data.tokenizer_utils import VLMTokenizer


def run_demo():
    print("=" * 60)
    print("🎨 ChromaDetect-VLM: Image & Color Detection Demo")
    print("=" * 60)

    # 1. Generate test image with known objects
    print("[*] Generating test image with colored shapes...")
    sample = generate_synthetic_sample(image_size=256, min_objects=3, max_objects=5)
    img = sample["image"]
    annotations = sample["annotations"]

    raw_test_path = "sample_test_image.png"
    img.save(raw_test_path)
    print(f"[✓] Saved test input image to: {raw_test_path}")

    # 2. Format as ground truth detections
    tokenizer = VLMTokenizer()
    target_parts = []
    detections = []

    for ann in annotations:
        y1, x1, y2, x2 = ann["bbox_norm"]
        b_str = f"{tokenizer.coord_to_token(y1)}{tokenizer.coord_to_token(x1)}{tokenizer.coord_to_token(y2)}{tokenizer.coord_to_token(x2)}"
        target_parts.append(f"{b_str} {ann['color']} {ann['shape']}")

        # Pixel bbox for image size 256
        py1 = int((y1 / 999.0) * 256)
        px1 = int((x1 / 999.0) * 256)
        py2 = int((y2 / 999.0) * 256)
        px2 = int((x2 / 999.0) * 256)

        detections.append({
            "bbox": [py1, px1, py2, px2],
            "norm_bbox": [y1, x1, y2, x2],
            "description": f"{ann['color']} {ann['shape']}",
            "color": ann["color"],
            "hex": ann["hex"],
        })

    vlm_output_str = " ; ".join(target_parts)
    print(f"\n[*] Simulated VLM Coordinate & Color Output Stream:")
    print(f"    {vlm_output_str}\n")

    # 3. Draw annotated detections and color palette
    output_path = "demo_annotated_detection.png"
    draw_detections_and_palette(img, detections, output_path=output_path)
    print(f"[✓] Successfully drew detection bounding boxes and color swatches!")
    print(f"[✓] Saved final annotated result to: {output_path}")


if __name__ == "__main__":
    run_demo()

