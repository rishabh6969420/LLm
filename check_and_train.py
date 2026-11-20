"""
Dataset Eligibility Inspector & Pretraining Gate for User-Collected Data.
Workflow:
  1. Audits user-collected data in data/user_dataset/ (images, coordinates, color shades)
  2. Evaluates eligibility and sample volume thresholds for the selected model scale (100M, 500M, 1B, 3B)
  3. IF INSUFFICIENT OR CORRUPT: Tells user exactly what is missing and how much more to collect.
  4. IF ELIGIBLE & READY: Strictly partitions into 75% Train / 15% Val / 10% Test and starts pretraining!
Usage:
  python check_and_train.py --model_size 500M
  python check_and_train.py --model_size 1B --check_only
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from config import get_model_preset, TrainingConfig
from model.vision_llm import StriklerVLM
from data.tokenizer_utils import VLMTokenizer
from data.color_utils import COLOR_SHADE_DATABASE, match_color_name
from engine.trainer import VLMTrainer
from data.dataset import vlm_collate_fn

# Recommended minimum sample thresholds per model scale
MIN_THRESHOLD_MAP = {
    "100M": 5000,
    "300M": 25000,
    "500M": 100000,
    "1B": 250000,
    "3B": 1000000,
}


class UserCollectedDataset(Dataset):
    """PyTorch Dataset loading user images and annotations from disk."""

    def __init__(self, data_records: List[Dict[str, Any]], data_dir: str, tokenizer: VLMTokenizer, image_size: int = 224):
        self.records = data_records
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_path = os.path.join(self.data_dir, rec["image_file"]) if not os.path.isabs(rec["image_file"]) else rec["image_file"]

        # Load and preprocess image
        image = Image.open(img_path).convert("RGB").resize((self.image_size, self.image_size))
        img_arr = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_arr - mean) / std

        # Format prompt & targets
        prompt = "<bos><detect_all>"
        target_parts = []
        for obj in rec.get("objects", []):
            coords = obj.get("bbox_norm") or obj.get("bbox")
            if not coords or len(coords) != 4:
                continue
            y1, x1, y2, x2 = coords
            b_str = f"{self.tokenizer.coord_to_token(y1)}{self.tokenizer.coord_to_token(x1)}{self.tokenizer.coord_to_token(y2)}{self.tokenizer.coord_to_token(x2)}"
            target_parts.append(f"{b_str} {obj['shade']} {obj['label']}")

        target = " ; ".join(target_parts) + "<eos>"

        prompt_ids = self.tokenizer.encode(prompt)
        target_ids = self.tokenizer.encode(target)
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        max_seq_len = 512
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]
        else:
            pad_len = max_seq_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "image": img_tensor,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_text": prompt,
            "target_text": target,
            "annotations": rec.get("objects", []),
        }


import numpy as np


class DatasetAuditor:
    """Performs rigorous quality and consistency checks on user-provided data."""

    def __init__(self, data_dir: str = "data/user_dataset", model_size: str = "500M", min_samples: Optional[int] = None):
        self.data_dir = data_dir
        self.model_size = model_size.upper()
        self.min_samples = min_samples or MIN_THRESHOLD_MAP.get(self.model_size, 100000)
        self.valid_records: List[Dict[str, Any]] = []
        self.corrupted_files: List[str] = []
        self.geometry_errors: List[str] = []
        self.shade_coverage: Dict[str, int] = {}

    def audit(self) -> Tuple[bool, Dict[str, Any]]:
        print("=" * 75)
        print("🔍 STRIKLER USER DATASET ELIGIBILITY & QUALITY AUDITOR")
        print("=" * 75)
        print(f"[*] Target Model Scale         : Strikler-{self.model_size}")
        print(f"[*] Minimum Required Samples   : {self.min_samples:,}")
        print(f"[*] Checking Directory         : {self.data_dir}")
        print("-" * 75)

        if not os.path.exists(self.data_dir):
            return False, {"error": f"Directory '{self.data_dir}' does not exist. Please create it and add your data."}

        # 1. Locate annotation file
        json_path = os.path.join(self.data_dir, "annotations.json")
        jsonl_path = os.path.join(self.data_dir, "annotations.jsonl")

        raw_entries = []
        if os.path.exists(json_path):
            print(f"[*] Found annotations file: {json_path}")
            try:
                with open(json_path, "r") as f:
                    raw_entries = json.load(f)
            except Exception as e:
                return False, {"error": f"Failed to parse annotations.json: {e}"}
        elif os.path.exists(jsonl_path):
            print(f"[*] Found annotations file: {jsonl_path}")
            try:
                with open(jsonl_path, "r") as f:
                    for line in f:
                        if line.strip():
                            raw_entries.append(json.loads(line.strip()))
            except Exception as e:
                return False, {"error": f"Failed to parse annotations.jsonl: {e}"}
        else:
            return False, {
                "error": f"No annotations.json or annotations.jsonl found in {self.data_dir}.\n"
                         f"Please refer to data/user_dataset/README.md for the expected format."
            }

        print(f"[*] Loaded {len(raw_entries):,} raw candidate entries. Validating integrity...")

        # 2. Validate images and bounding box geometry
        for idx, entry in enumerate(raw_entries):
            img_rel = entry.get("image_file")
            if not img_rel:
                self.corrupted_files.append(f"Entry #{idx}: missing 'image_file' key")
                continue

            img_full = os.path.join(self.data_dir, img_rel) if not os.path.isabs(img_rel) else img_rel
            if not os.path.exists(img_full):
                self.corrupted_files.append(f"Image not found on disk: {img_rel}")
                continue

            # Verify Pillow can open without corruption
            try:
                with Image.open(img_full) as im:
                    im.verify()
                    w, h = im.size
            except Exception as e:
                self.corrupted_files.append(f"Corrupt image {img_rel}: {e}")
                continue

            # Validate objects
            objects = entry.get("objects", [])
            valid_objs = []
            for obj in objects:
                bbox = obj.get("bbox") or obj.get("bbox_norm")
                if not bbox or len(bbox) != 4:
                    self.geometry_errors.append(f"{img_rel}: invalid bbox format {bbox}")
                    continue

                y1, x1, y2, x2 = bbox
                # If normalized in float [0, 1], scale to [0, 999]
                if all(isinstance(c, float) and 0.0 <= c <= 1.0 for c in bbox):
                    y1, x1, y2, x2 = int(y1 * 999), int(x1 * 999), int(y2 * 999), int(x2 * 999)

                # Geometry check
                if not (0 <= y1 < y2 <= 999 and 0 <= x1 < x2 <= 999):
                    self.geometry_errors.append(f"{img_rel}: inverted or out-of-bounds coordinates {bbox}")
                    continue

                shade = obj.get("shade", "unknown").lower()
                self.shade_coverage[shade] = self.shade_coverage.get(shade, 0) + 1

                valid_objs.append({
                    "label": obj.get("label", "object"),
                    "shade": shade,
                    "bbox_norm": [y1, x1, y2, x2],
                })

            if len(valid_objs) > 0:
                self.valid_records.append({
                    "image_file": img_rel,
                    "objects": valid_objs,
                })

        valid_count = len(self.valid_records)
        is_eligible = valid_count >= self.min_samples and len(self.corrupted_files) == 0

        report = {
            "valid_samples": valid_count,
            "min_required": self.min_samples,
            "corrupted_count": len(self.corrupted_files),
            "geometry_errors_count": len(self.geometry_errors),
            "unique_shades_found": len(self.shade_coverage),
            "deficit": max(0, self.min_samples - valid_count),
        }

        return is_eligible, report


def check_and_train_pipeline(
    data_dir: str = "data/user_dataset",
    model_size: str = "500M",
    min_samples: Optional[int] = None,
    batch_size: int = 32,
    lr: float = 2e-4,
    steps: int = 50000,
    check_only: bool = False,
):
    auditor = DatasetAuditor(data_dir=data_dir, model_size=model_size, min_samples=min_samples)
    is_eligible, report = auditor.audit()

    if "error" in report:
        print(f"\n❌ DATASET AUDIT FAILED:")
        print(f"   {report['error']}")
        print(f"\n👉 ACTION REQUIRED: Please prepare data in '{data_dir}' and run again.")
        return False

    print("\n" + "=" * 75)
    print("📋 AUDIT SUMMARY REPORT")
    print("=" * 75)
    print(f"• Verified Valid Samples     : {report['valid_samples']:,} / {report['min_required']:,}")
    print(f"• Corrupted / Missing Images : {report['corrupted_count']:,}")
    print(f"• Geometry Bounding Box Errs : {report['geometry_errors_count']:,}")
    print(f"• Distinct Shades Identified : {report['unique_shades_found']:,}")

    if not is_eligible:
        print("\n" + "!" * 75)
        print(f"❌ DATASET IS NOT YET READY FOR STRIKLER-{model_size.upper()}")
        print("!" * 75)
        if report["deficit"] > 0:
            print(f"⚠️  SAMPLE VOLUME DEFICIT:")
            print(f"   You currently have {report['valid_samples']:,} valid samples.")
            print(f"   Strikler-{model_size.upper()} requires at least {report['min_required']:,} samples.")
            print(f"   👉 ACTION: Please prepare and add {report['deficit']:,} MORE clean samples to proceed.")

        if report["corrupted_count"] > 0:
            print(f"\n⚠️  CORRUPT / MISSING FILES DETECTED:")
            for err in auditor.corrupted_files[:10]:
                print(f"   - {err}")
            if len(auditor.corrupted_files) > 10:
                print(f"   ... and {len(auditor.corrupted_files) - 10} more files.")

        print("\nPretraining will NOT start until these issues are resolved.")
        print("Please prepare more data, clean the corrupted files, and run this command again!")
        return False

    # DATASET IS 100% READY
    print("\n" + "=" * 75)
    print(f"🎉 DATASET IS 100% ELIGIBLE & READY FOR STRIKLER-{model_size.upper()}!")
    print("=" * 75)

    if check_only:
        print("[*] check_only flag was set. Pretraining not started.")
        return True

    # Strict 75% / 15% / 10% Split
    print("\n[*] Applying strict 75% Train / 15% Val / 10% Test partitioning...")
    total = len(auditor.valid_records)
    indices = list(range(total))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_n = int(total * 0.75)
    val_n = int(total * 0.15)

    train_idx = indices[:train_n]
    val_idx = indices[train_n : train_n + val_n]
    test_idx = indices[train_n + val_n :]

    tokenizer = VLMTokenizer()
    full_dataset = UserCollectedDataset(auditor.valid_records, data_dir, tokenizer)

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True, collate_fn=vlm_collate_fn)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False, collate_fn=vlm_collate_fn)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=batch_size, shuffle=False, collate_fn=vlm_collate_fn)

    print(f"  [✓] Training Split (75%)   : {len(train_idx):,} samples")
    print(f"  [✓] Validation Split (15%) : {len(val_idx):,} samples")
    print(f"  [✓] Testing Split (10%)    : {len(test_idx):,} samples")

    # Start Pretraining
    print(f"\n🚀 LAUNCHING STRIKLER-{model_size.upper()} PRETRAINING ON GPU...")
    v_cfg, p_cfg, l_cfg = get_model_preset(model_size)
    model = StriklerVLM(v_cfg, p_cfg, l_cfg)

    train_cfg = TrainingConfig(
        batch_size=batch_size,
        learning_rate=lr,
        max_steps=steps,
        save_every_steps=1000 if steps >= 5000 else 100,
        eval_every_steps=500 if steps >= 5000 else 50,
    )

    trainer = VLMTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        config=train_cfg,
        tokenizer=tokenizer,
    )

    trainer.train(max_steps=steps)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check user dataset eligibility and start pretraining")
    parser.add_argument("--data_dir", type=str, default="data/user_dataset", help="Directory where user placed dataset")
    parser.add_argument("--model_size", type=str, default="500M", choices=["100M", "300M", "500M", "1B", "3B"], help="Model size to train")
    parser.add_argument("--min_samples", type=int, default=None, help="Custom minimum sample threshold")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=50000, help="Training steps")
    parser.add_argument("--check_only", action="store_true", help="Only check eligibility without training")
    args = parser.parse_args()

    check_and_train_pipeline(
        data_dir=args.data_dir,
        model_size=args.model_size,
        min_samples=args.min_samples,
        batch_size=args.batch_size,
        lr=args.lr,
        steps=args.steps,
        check_only=args.check_only,
    )
