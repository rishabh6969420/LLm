"""
PyTorch Dataset and Split Manager for Object Detection and Color Grounding VLM.
Implements strict 75% Training, 15% Validation, 10% Testing partition.
"""

import math
import random
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

from data.synthetic_generator import generate_synthetic_sample
from data.tokenizer_utils import VLMTokenizer


def vlm_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collator that stacks tensors and leaves raw metadata as lists."""
    images = torch.stack([item["image"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    prompt_texts = [item["prompt_text"] for item in batch]
    target_texts = [item["target_text"] for item in batch]
    annotations = [item.get("annotations", []) for item in batch]

    return {
        "image": images,
        "input_ids": input_ids,
        "labels": labels,
        "prompt_text": prompt_texts,
        "target_text": target_texts,
        "annotations": annotations,
    }


class VisionColorDataset(Dataset):
    """
    Multimodal Dataset for Object Detection and Color Grounding.
    Supports:
      1. Scene-level object & color detection (<detect_all>)
      2. Color-conditioned object detection (<detect_color> <color>)
      3. Region-specific color and hex interrogation (<color_at> <bbox>)
    """

    def __init__(
        self,
        num_samples: int = 100000,
        image_size: int = 224,
        tokenizer: Optional[VLMTokenizer] = None,
        max_seq_len: int = 256,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.tokenizer = tokenizer or VLMTokenizer()
        self.max_seq_len = max_seq_len
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Seed generator per index to ensure deterministic and reproducible samples across splits
        sample_seed = self.seed + idx
        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32 - 1))

        sample = generate_synthetic_sample(image_size=self.image_size)
        image = sample["image"]
        annotations = sample["annotations"]

        # Normalize image to tensor (3, H, W) with ImageNet mean and std
        img_arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Rotate tasks deterministically based on index
        task_id = idx % 3

        if task_id == 0 or len(annotations) == 0:
            prompt = "<bos><detect_all>"
            target_parts = []
            for ann in annotations:
                y1, x1, y2, x2 = ann["bbox_norm"]
                b_str = f"{self.tokenizer.coord_to_token(y1)}{self.tokenizer.coord_to_token(x1)}{self.tokenizer.coord_to_token(y2)}{self.tokenizer.coord_to_token(x2)}"
                target_parts.append(f"{b_str} {ann['color']} {ann['shape']}")
            target = " ; ".join(target_parts) + "<eos>"

        elif task_id == 1:
            target_obj = annotations[0]
            chosen_color = target_obj["color"]
            prompt = f"<bos><detect_color> {chosen_color}"
            matches = [a for a in annotations if a["color"] == chosen_color]
            target_parts = []
            for ann in matches:
                y1, x1, y2, x2 = ann["bbox_norm"]
                b_str = f"{self.tokenizer.coord_to_token(y1)}{self.tokenizer.coord_to_token(x1)}{self.tokenizer.coord_to_token(y2)}{self.tokenizer.coord_to_token(x2)}"
                target_parts.append(f"{b_str} {ann['shape']}")
            target = " ; ".join(target_parts) + "<eos>"

        else:  # task_id == 2: color_at
            target_obj = annotations[0]
            y1, x1, y2, x2 = target_obj["bbox_norm"]
            b_str = f"{self.tokenizer.coord_to_token(y1)}{self.tokenizer.coord_to_token(x1)}{self.tokenizer.coord_to_token(y2)}{self.tokenizer.coord_to_token(x2)}"
            prompt = f"<bos><color_at> {b_str}"
            target = f" {target_obj['color']} {target_obj['shape']} hex {target_obj['hex']}<eos>"

        prompt_ids = self.tokenizer.encode(prompt)
        target_ids = self.tokenizer.encode(target)

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        # Truncate or pad to max_seq_len
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "image": img_tensor,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_text": prompt,
            "target_text": target,
            "annotations": annotations,
        }


def get_dataset_splits(
    total_samples: int = 100000,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    image_size: int = 224,
    tokenizer: Optional[VLMTokenizer] = None,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, Any]]:
    """
    Partitions the dataset strictly into:
      • 75% Training
      • 15% Validation
      • 10% Testing
    """
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-5), \
        "Ratios must sum to 1.0 (75% + 15% + 10%)"

    base_dataset = VisionColorDataset(
        num_samples=total_samples,
        image_size=image_size,
        tokenizer=tokenizer,
        seed=seed,
    )

    indices = list(range(total_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_count = int(total_samples * train_ratio)
    val_count = int(total_samples * val_ratio)
    test_count = total_samples - train_count - val_count

    train_indices = indices[:train_count]
    val_indices = indices[train_count : train_count + val_count]
    test_indices = indices[train_count + val_count :]

    train_split = Subset(base_dataset, train_indices)
    val_split = Subset(base_dataset, val_indices)
    test_split = Subset(base_dataset, test_indices)

    split_stats = {
        "total": total_samples,
        "train": len(train_split),
        "val": len(val_split),
        "test": len(test_split),
        "train_pct": (len(train_split) / total_samples) * 100.0,
        "val_pct": (len(val_split) / total_samples) * 100.0,
        "test_pct": (len(test_split) / total_samples) * 100.0,
    }

    return train_split, val_split, test_split, split_stats


def get_dataloaders(
    total_samples: int = 100000,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    tokenizer: Optional[VLMTokenizer] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Returns three isolated DataLoaders for 75% Train, 15% Val, and 10% Test.
    """
    train_split, val_split, test_split, stats = get_dataset_splits(
        total_samples=total_samples,
        train_ratio=0.75,
        val_ratio=0.15,
        test_ratio=0.10,
        image_size=image_size,
        tokenizer=tokenizer,
        seed=seed,
    )

    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=vlm_collate_fn,
    )
    val_loader = DataLoader(
        val_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=vlm_collate_fn,
    )
    test_loader = DataLoader(
        test_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=vlm_collate_fn,
    )

    return train_loader, val_loader, test_loader, stats
