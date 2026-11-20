"""
Automated User Data Cleaning & Formatting Tool for Strikler.
Takes raw images and annotations, runs all 4 cleaning checks:
  1. Validates image integrity & converts to standard RGB
  2. Filters inverted, out-of-bounds, or microscopic boxes (< 10px)
  3. Standardizes color shade labels using CIELAB matching
  4. Saves clean, formatted data directly into data/user_dataset/
Usage:
  python -m data.clean_my_data --raw_json path/to/raw_annotations.json --raw_images path/to/raw_images/
"""

import os
import json
import argparse
from typing import List, Dict, Any
from PIL import Image
import numpy as np

from data.color_utils import match_color_name, COLOR_SHADE_DATABASE


def clean_raw_data(
    raw_json_path: str,
    raw_images_dir: str,
    output_dir: str = "data/user_dataset",
    min_box_pixels: int = 10,
):
    print("=" * 70)
    print("🧹 STRIKLER AUTOMATED DATA CLEANING UTILITY")
    print("=" * 70)
    print(f"[*] Input Annotations : {raw_json_path}")
    print(f"[*] Input Images Dir  : {raw_images_dir}")
    print(f"[*] Output Target Dir : {output_dir}")
    print("-" * 70)

    out_images_dir = os.path.join(output_dir, "images")
    os.makedirs(out_images_dir, exist_ok=True)

    with open(raw_json_path, "r") as f:
        raw_data = json.load(f)

    clean_records = []
    dropped_corrupt = 0
    dropped_boxes = 0

    print(f"[*] Cleaning {len(raw_data):,} candidate entries...")

    for idx, item in enumerate(raw_data):
        rel_img = item.get("image_file") or item.get("file_name") or item.get("image")
        if not rel_img:
            continue

        raw_img_path = os.path.join(raw_images_dir, rel_img) if not os.path.isabs(rel_img) else rel_img
        if not os.path.exists(raw_img_path):
            dropped_corrupt += 1
            continue

        # 1. Image Check & RGB Conversion
        try:
            with Image.open(raw_img_path) as im:
                im.verify()
            img = Image.open(raw_img_path).convert("RGB")
            w, h = img.size
            if w < 50 or h < 50:
                dropped_corrupt += 1
                continue
        except Exception:
            dropped_corrupt += 1
            continue

        # 2. Bounding Box & Shade Cleaning
        raw_objs = item.get("objects") or item.get("annotations") or []
        clean_objs = []

        for obj in raw_objs:
            bbox = obj.get("bbox")
            if not bbox or len(bbox) != 4:
                dropped_boxes += 1
                continue

            # Convert format: [x, y, w, h] to [ymin, xmin, ymax, xmax] if needed
            if "width" in obj or "height" in obj or (bbox[2] < bbox[0] and bbox[3] < bbox[1]):
                xmin, ymin, box_w, box_h = bbox
                ymin, xmin, ymax, xmax = ymin, xmin, ymin + box_h, xmin + box_w
            else:
                ymin, xmin, ymax, xmax = bbox

            # Normalize to [0, 999] if in pixels
            if ymax > 1.0 or xmax > 1.0:
                ny1 = int((max(0, ymin) / h) * 999)
                nx1 = int((max(0, xmin) / w) * 999)
                ny2 = int((min(h, ymax) / h) * 999)
                nx2 = int((min(w, xmax) / w) * 999)
            else:
                ny1, nx1, ny2, nx2 = int(ymin * 999), int(xmin * 999), int(ymax * 999), int(xmax * 999)

            # Quality bounds check
            if not (0 <= ny1 < ny2 <= 999 and 0 <= nx1 < nx2 <= 999):
                dropped_boxes += 1
                continue

            if (ny2 - ny1) < min_box_pixels or (nx2 - nx1) < min_box_pixels:
                dropped_boxes += 1
                continue

            # Clean Shade Label
            raw_shade = str(obj.get("shade") or obj.get("color") or "unknown").lower().strip()
            # If shade is not in standard database, extract dominant color from image crop
            if raw_shade not in COLOR_SHADE_DATABASE:
                py1, px1 = int((ny1 / 999.0) * h), int((nx1 / 999.0) * w)
                py2, px2 = int((ny2 / 999.0) * h), int((nx2 / 999.0) * w)
                crop = img.crop((px1, py1, px2, py2))
                avg_rgb = tuple(np.array(crop).mean(axis=(0, 1)).astype(int))
                clean_shade = match_color_name(avg_rgb)
            else:
                clean_shade = raw_shade

            clean_objs.append({
                "label": str(obj.get("label") or obj.get("category") or "object"),
                "shade": clean_shade,
                "bbox_norm": [ny1, nx1, ny2, nx2],
            })

        if clean_objs:
            # Copy image to clean destination
            save_name = f"clean_{idx:06d}.jpg"
            dest_img_path = os.path.join(out_images_dir, save_name)
            img.save(dest_img_path, quality=95)

            clean_records.append({
                "image_file": f"images/{save_name}",
                "objects": clean_objs,
            })

    # Save cleaned annotations
    out_ann_path = os.path.join(output_dir, "annotations.json")
    with open(out_ann_path, "w") as f:
        json.dump(clean_records, f, indent=2)

    print("\n" + "=" * 70)
    print("🎉 DATA CLEANING COMPLETE!")
    print("=" * 70)
    print(f"• Successfully Cleaned Samples : {len(clean_records):,}")
    print(f"• Dropped Corrupted Images     : {dropped_corrupt:,}")
    print(f"• Dropped Invalid Bounding Boxes: {dropped_boxes:,}")
    print(f"• Clean Dataset Saved To       : {out_ann_path}")
    print("-" * 70)
    print("Now run eligibility check:")
    print("  python check_and_train.py --model_size 500M")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw visual dataset for Strikler")
    parser.add_argument("--raw_json", type=str, required=True, help="Path to raw annotations JSON")
    parser.add_argument("--raw_images", type=str, required=True, help="Path to raw images folder")
    parser.add_argument("--output_dir", type=str, default="data/user_dataset", help="Output directory")
    args = parser.parse_args()

    clean_raw_data(args.raw_json, args.raw_images, args.output_dir)
