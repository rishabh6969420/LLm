"""
Visualization utilities for Object Detection and Accurate Color Shade Grounding.
Draws labeled bounding boxes, fine-grained shade tags (e.g. turquoise, blood red, wine red),
and high-precision color palette swatches directly on images.
"""

from typing import List, Dict, Tuple, Any
import re
import numpy as np
from PIL import Image, ImageDraw

from data.color_utils import COLOR_SHADE_DATABASE, rgb_to_hex


def parse_vlm_detection_output(text: str, image_size: int = 224) -> List[Dict[str, Any]]:
    """
    Parses VLM generation string formatted like:
    <loc_120><loc_340><loc_450><loc_600> blood red circle ; <loc_50><loc_100><loc_200><loc_300> turquoise square
    into structured detections with accurate shades.
    """
    results = []
    pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>\s*([^;]+)"
    matches = re.findall(pattern, text)

    for m in matches:
        norm_y1, norm_x1, norm_y2, norm_x2 = int(m[0]), int(m[1]), int(m[2]), int(m[3])
        desc = m[4].strip()

        # Convert normalized coordinates [0, 999] back to image pixels
        y1 = int((norm_y1 / 999.0) * image_size)
        x1 = int((norm_x1 / 999.0) * image_size)
        y2 = int((norm_y2 / 999.0) * image_size)
        x2 = int((norm_x2 / 999.0) * image_size)

        # Detect longest matching shade in description (e.g. "blood red" before "red")
        found_shade = "unknown"
        desc_lower = desc.lower()
        # Sort shades by length descending to match compound shades first
        sorted_shades = sorted(COLOR_SHADE_DATABASE.keys(), key=lambda s: len(s), reverse=True)
        for shade_name in sorted_shades:
            if shade_name in desc_lower:
                found_shade = shade_name
                break

        shade_info = COLOR_SHADE_DATABASE.get(
            found_shade,
            {"rgb": (128, 128, 128), "parent": "unknown", "hex": "#808080"}
        )

        results.append({
            "bbox": [y1, x1, y2, x2],
            "norm_bbox": [norm_y1, norm_x1, norm_y2, norm_x2],
            "description": desc,
            "shade": found_shade,
            "color": found_shade,
            "parent_color": shade_info["parent"],
            "rgb": shade_info["rgb"],
            "hex": shade_info["hex"],
        })

    return results


def draw_detections_and_palette(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    output_path: str = "output_detection.png",
) -> Image.Image:
    """
    Draws bounding boxes, fine-grained shade labels, and an attached color-palette summary bar.
    """
    img_w, img_h = image.size
    palette_width = 180
    total_w = img_w + palette_width

    canvas = Image.new("RGB", (total_w, img_h), (245, 245, 248))
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)

    # 1. Draw Detections
    for det in detections:
        y1, x1, y2, x2 = det["bbox"]
        box_rgb = det.get("rgb", (220, 20, 60))

        # Thick bounding box
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=box_rgb,
            )

        # Label background tag
        label_text = f"{det['description']}"
        text_bbox = draw.textbbox((x1, max(0, y1 - 18)), label_text)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2],
            fill=box_rgb,
        )
        # Determine text contrast (white vs dark text)
        r, g, b = box_rgb
        text_color = (20, 20, 20) if (0.299 * r + 0.587 * g + 0.114 * b) > 160 else (255, 255, 255)
        draw.text((x1 + 1, max(0, y1 - 18)), label_text, fill=text_color)

    # 2. Draw Color Palette Sidebar
    draw.rectangle([img_w, 0, total_w, img_h], fill=(238, 240, 244), outline=(190, 195, 200))
    draw.text((img_w + 12, 10), "DETECTED SHADES", fill=(30, 30, 30))

    y_offset = 35
    seen_shades = set()

    for det in detections:
        shade_name = det.get("shade", "unknown")
        if shade_name in seen_shades or shade_name == "unknown":
            continue
        seen_shades.add(shade_name)

        c_rgb = det.get("rgb", (128, 128, 128))
        c_hex = det.get("hex", "#000000")
        parent = det.get("parent_color", "")

        # Swatch box
        draw.rectangle(
            [img_w + 12, y_offset, img_w + 38, y_offset + 24],
            fill=c_rgb,
            outline=(70, 70, 70),
        )
        # Shade name & Hex
        draw.text((img_w + 46, y_offset), shade_name.title(), fill=(15, 15, 15))
        sub_desc = f"{c_hex} ({parent})" if parent else c_hex
        draw.text((img_w + 46, y_offset + 14), sub_desc, fill=(90, 95, 105))

        y_offset += 34
        if y_offset > img_h - 40:
            break

    canvas.save(output_path)
    return canvas
