"""
PyTorch Dataset and DataLoader for Object Detection and Color Grounding VLM.
"""

import random
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from data.synthetic_generator import generate_synthetic_sample
from data.tokenizer_utils import VLMTokenizer


class VisionColorDataset(Dataset):
    """
    Multimodal Dataset generating training pairs for:
    1. Full scene object & color detection (<detect_all>)
    2. Color-conditioned object detection (<detect_color> <color>)
    3. Region color interrogation (<color_at> <bbox>)
    """

    def __init__(
        self,
        num_samples: int = 50000,
        image_size: int = 224,
        tokenizer: VLMTokenizer = None,
        max_seq_len: int = 256,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.tokenizer = tokenizer or VLMTokenizer()
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = generate_synthetic_sample(image_size=self.image_size)
        image = sample["image"]
        annotations = sample["annotations"]

        # Normalize image to [-1, 1] tensor (3, H, W)
        img_arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)
        # Normalize with ImageNet standard mean & std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Pick random task variant
        task_type = random.choice(["detect_all", "detect_color", "color_at"])

        if task_type == "detect_all" or len(annotations) == 0:
            prompt = "<bos><detect_all>"
            target_parts = []
            for ann in annotations:
                y1, x1, y2, x2 = ann["bbox_norm"]
                b_str = f"{self.tokenizer.coord_to_token(y1)}{self.tokenizer.coord_to_token(x1)}{self.tokenizer.coord_to_token(y2)}{self.tokenizer.coord_to_token(x2)}"
                target_parts.append(f"{b_str} {ann['color']} {ann['shape']}")
            target = " ; ".join(target_parts) + "<eos>"

        elif task_type == "detect_color":
            target_obj = random.choice(annotations)
            chosen_color = target_obj["color"]
            prompt = f"<bos><detect_color> {chosen_color}"
            matches = [a for a in annotations if a["color"] == chosen_color]
            target_parts = []
            for ann in matches:
                y1, x1, y2, x2 = ann["bbox_norm"]
                b_str = f"{self.tokenizer.coord_to_token(y1)}{self.tokenizer.coord_to_token(x1)}{self.tokenizer.coord_to_token(y2)}{self.tokenizer.coord_to_token(x2)}"
                target_parts.append(f"{b_str} {ann['shape']}")
            target = " ; ".join(target_parts) + "<eos>"

        else:  # color_at
            target_obj = random.choice(annotations)
            y1, x1, y2, x2 = target_obj["bbox_norm"]
            b_str = f"{self.tokenizer.coord_to_token(y1)}{self.tokenizer.coord_to_token(x1)}{self.tokenizer.coord_to_token(y2)}{self.tokenizer.coord_to_token(x2)}"
            prompt = f"<bos><color_at> {b_str}"
            target = f" {target_obj['color']} {target_obj['shape']} hex {target_obj['hex']}<eos>"

        prompt_ids = self.tokenizer.encode(prompt)
        target_ids = self.tokenizer.encode(target)

        input_ids = prompt_ids + target_ids
        # Labels: -100 for prompt positions so loss is only computed on target generation
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
        }


def get_dataloader(
    num_samples: int = 50000,
    batch_size: int = 32,
    image_size: int = 224,
    tokenizer: VLMTokenizer = None,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create a high-throughput PyTorch DataLoader."""
    dataset = VisionColorDataset(
        num_samples=num_samples,
        image_size=image_size,
        tokenizer=tokenizer,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
