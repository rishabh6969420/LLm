"""
Synthetic Image Generation for Object and Color Grounding.
Generates images with diverse colored geometric shapes and objects,
with ground-truth bounding boxes, colors, and spatial descriptions.
"""

import random
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw

from data.color_utils import COLOR_SPECTRUM, rgb_to_hex

SHAPES = ["circle", "rectangle", "square", "triangle", "ellipse", "polygon"]


def generate_synthetic_sample(
    image_size: int = 224,
    min_objects: int = 1,
    max_objects: int = 4,
) -> Dict[str, Any]:
    """
    Generate a synthetic image with 1 to N colored objects.
    Returns:
      image: PIL.Image
      annotations: list of {"label": str, "color": str, "hex": str, "bbox": [ymin, xmin, ymax, xmax]}
    """
    # Random soft background color
    bg_color = (
        random.randint(220, 250),
        random.randint(220, 250),
        random.randint(220, 250),
    )
    img = Image.new("RGB", (image_size, image_size), bg_color)
    draw = ImageDraw.Draw(img)

    num_objects = random.randint(min_objects, max_objects)
    annotations = []
    color_names = list(COLOR_SPECTRUM.keys())

    for _ in range(num_objects):
        shape = random.choice(SHAPES)
        color_name = random.choice(color_names)
        base_rgb = COLOR_SPECTRUM[color_name]

        # Add slight jitter to RGB for realism
        r = max(0, min(255, base_rgb[0] + random.randint(-10, 10)))
        g = max(0, min(255, base_rgb[1] + random.randint(-10, 10)))
        b = max(0, min(255, base_rgb[2] + random.randint(-10, 10)))
        obj_color = (r, g, b)

        # Object size
        size_w = random.randint(30, 80)
        size_h = random.randint(30, 80) if shape != "square" else size_w

        # Position
        x1 = random.randint(10, image_size - size_w - 10)
        y1 = random.randint(10, image_size - size_h - 10)
        x2 = x1 + size_w
        y2 = y1 + size_h

        # Draw shape
        if shape in ["circle", "square", "rectangle", "ellipse"]:
            draw.rectangle([x1, y1, x2, y2], fill=obj_color, outline=(30, 30, 30))
        elif shape == "triangle":
            points = [(x1 + size_w // 2, y1), (x1, y2), (x2, y2)]
            draw.polygon(points, fill=obj_color, outline=(30, 30, 30))
        else:
            draw.ellipse([x1, y1, x2, y2], fill=obj_color, outline=(30, 30, 30))

        # Normalized coordinates in [0, 999] for VLM tokens
        norm_ymin = int((y1 / image_size) * 999)
        norm_xmin = int((x1 / image_size) * 999)
        norm_ymax = int((y2 / image_size) * 999)
        norm_xmax = int((x2 / image_size) * 999)

        annotations.append({
            "shape": shape,
            "color": color_name,
            "hex": rgb_to_hex(r, g, b),
            "bbox_pixels": [y1, x1, y2, x2],
            "bbox_norm": [norm_ymin, norm_xmin, norm_ymax, norm_xmax],
        })

    return {
        "image": img,
        "annotations": annotations,
    }

