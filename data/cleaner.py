"""
Data Cleaning, Quality Auditing, and Verification Engine for Strikler-3B.
Filters out noisy, corrupted, inverted, or color-inconsistent samples.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from PIL import Image
import torch

from data.color_utils import COLOR_SHADE_DATABASE, extract_box_dominant_color, delta_e_cie76, rgb_to_cielab
from data.tokenizer_utils import VLMTokenizer


class StriklerDataCleaner:
    """
    Validates and cleans multimodal grounding data:
      1. Bounding Box Geometry Validity
      2. Color Shade Consistency (CIELAB Delta-E tolerance)
      3. Image Contrast & Non-Degeneracy
      4. Token Sequence & Boundary Integrity
    """

    def __init__(self, max_delta_e: float = 18.0, min_box_pixels: int = 10, max_seq_len: int = 512):
        self.max_delta_e = max_delta_e
        self.min_box_pixels = min_box_pixels
        self.max_seq_len = max_seq_len
        self.tokenizer = VLMTokenizer()

    def audit_sample(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Runs comprehensive quality audits on a single multimodal sample.
        Returns (is_valid: bool, reason: str).
        """
        # 1. Audit Image
        image = sample.get("image")
        if image is None:
            return False, "Missing image"

        if isinstance(image, torch.Tensor):
            if torch.isnan(image).any() or torch.isinf(image).any():
                return False, "Image tensor contains NaN or Inf"
            std = float(image.std().item())
            if std < 0.05:
                return False, f"Image is blank/washed-out (std={std:.4f})"
        elif isinstance(image, Image.Image):
            arr = np.array(image)
            if arr.std() < 5.0:
                return False, "Low-contrast / blank image"

        # 2. Audit Annotations
        annotations = sample.get("annotations", [])
        if not annotations:
            return False, "Empty annotations list"

        for ann in annotations:
            bbox_norm = ann.get("bbox_norm")
            if not bbox_norm or len(bbox_norm) != 4:
                return False, "Malformed bounding box coordinates"

            ymin, xmin, ymax, xmax = bbox_norm
            # Bounds check
            if not (0 <= ymin < ymax <= 999 and 0 <= xmin < xmax <= 999):
                return False, f"Inverted or out-of-bound box [{ymin}, {xmin}, {ymax}, {xmax}]"

            # Min size check
            if (ymax - ymin) < 8 or (xmax - xmin) < 8:
                return False, f"Microscopic box dimension ({ymax - ymin}x{xmax - xmin})"

            # Shade consistency check
            shade = ann.get("shade")
            if shade and shade in COLOR_SHADE_DATABASE:
                ref_rgb = COLOR_SHADE_DATABASE[shade]["rgb"]
                hex_str = ann.get("hex")
                if hex_str and hex_str.startswith("#"):
                    try:
                        r = int(hex_str[1:3], 16)
                        g = int(hex_str[3:5], 16)
                        b = int(hex_str[5:7], 16)
                        de = delta_e_cie76(rgb_to_cielab((r, g, b)), rgb_to_cielab(ref_rgb))
                        if de > self.max_delta_e:
                            return False, f"Color shade inconsistency: Delta-E={de:.2f} exceeds threshold"
                    except ValueError:
                        pass

        # 3. Audit Token Boundaries
        input_ids = sample.get("input_ids")
        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                ids_list = input_ids.tolist()
            else:
                ids_list = list(input_ids)

            if len(ids_list) > self.max_seq_len:
                return False, f"Sequence length {len(ids_list)} exceeds max {self.max_seq_len}"

            if self.tokenizer.bos_id not in ids_list:
                return False, "Missing <bos> token"

        return True, "Valid"

    def clean_batch(self, batch_samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Cleans a batch of samples and reports rejection reasons."""
        clean_samples = []
        reject_counts = {}

        for sample in batch_samples:
            is_valid, reason = self.audit_sample(sample)
            if is_valid:
                clean_samples.append(sample)
            else:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1

        return clean_samples, reject_counts
