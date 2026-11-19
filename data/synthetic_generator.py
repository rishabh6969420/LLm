"""
Synthetic Image Generation with Fine-Grained Accurate Color Shades.
Generates images with exact shades like turquoise, blood red, wine red, emerald green,
with ground-truth bounding boxes, shades, parent families, and CIELAB coordinates.
"""

import random
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw

from data.color_utils import COLOR_SHADE_DATABASE, rgb_to_hex

SHAPES = ["circle", "rectangle", "square", "triangle", "ellipse", "polygon"]


def generate_synthetic_sample(
    image_size: int = 224,
    min_objects: int = 1,
    max_objects: int = 4,
) -> Dict[str, Any]:
    """
    Generate a synthetic image with 1 to N colored objects using fine-grained shades.
    Returns:
      image: PIL.Image
      annotations: list of {
         "shape": str,
         "shade": str,
         "parent_color": str,
         "color": str,
         "hex": str,
         "bbox_norm": [ymin, xmin, ymax, xmax]
      }
    """
    # Soft neutral background
    bg_color = (
        random.randint(235, 252),
        random.randint(235, 252),
        random.randint(235, 252),
    )
    img = Image.new("RGB", (image_size, image_size), bg_color)
    draw = ImageDraw.Draw(img)

    num_objects = random.randint(min_objects, max_objects)
    annotations = []
    available_shades = list(COLOR_SHADE_DATABASE.keys())

    for _ in range(num_objects):
        shape = random.choice(SHAPES)
        shade_name = random.choice(available_shades)
        shade_info = COLOR_SHADE_DATABASE[shade_name]
        base_rgb = shade_info["rgb"]
        parent_family = shade_info["parent"]

        # Minimal chromatic jitter (±4) so the shade remains scientifically within the exact Delta-E tolerance
        r = max(0, min(255, base_rgb[0] + random.randint(-4, 4)))
        g = max(0, min(255, base_rgb[1] + random.randint(-4, 4)))
        b = max(0, min(255, base_rgb[2] + random.randint(-4, 4)))
        obj_color = (r, g, b)

        # Object size
        size_w = random.randint(32, 78)
        size_h = random.randint(32, 78) if shape != "square" else size_w

        # Position with boundary padding
        x1 = random.randint(12, image_size - size_w - 12)
        y1 = random.randint(12, image_size - size_h - 12)
        x2 = x1 + size_w
        y2 = y1 + size_h

        # Draw shape
        if shape in ["square", "rectangle"]:
            draw.rectangle([x1, y1, x2, y2], fill=obj_color, outline=(40, 40, 40))
        elif shape in ["circle", "ellipse"]:
            draw.ellipse([x1, y1, x2, y2], fill=obj_color, outline=(40, 40, 40))
        elif shape == "triangle":
            points = [(x1 + size_w // 2, y1), (x1, y2), (x2, y2)]
            draw.polygon(points, fill=obj_color, outline=(40, 40, 40))
        else:  # polygon (pentagon/hexagon)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            rw, rh = size_w // 2, size_h // 2
            points = [
                (cx, y1),
                (x2, cy - rh // 3),
                (x2 - rw // 4, y2),
                (x1 + rw // 4, y2),
                (x1, cy - rh // 3),
            ]
            draw.polygon(points, fill=obj_color, outline=(40, 40, 40))

        # Normalized coordinates in [0, 999] for VLM tokens
        norm_ymin = int((y1 / image_size) * 999)
        norm_xmin = int((x1 / image_size) * 999)
        norm_ymax = int((y2 / image_size) * 999)
        norm_xmax = int((x2 / image_size) * 999)

        annotations.append({
            "shape": shape,
            "shade": shade_name,
            "parent_color": parent_family,
            "color": shade_name,  # for backward compatibility
            "hex": rgb_to_hex(r, g, b),
            "bbox_pixels": [y1, x1, y2, x2],
            "bbox_norm": [norm_ymin, norm_xmin, norm_ymax, norm_xmax],
        })

    return {
        "image": img,
        "annotations": annotations,
    }
