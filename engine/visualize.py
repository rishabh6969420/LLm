"""
Visualization utilities for Object Detection and Color Grounding.
Draws labeled bounding boxes, color tags, and palette swatches directly on images.
"""

from typing import List, Dict, Tuple, Any
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.color_utils import COLOR_SPECTRUM, rgb_to_hex


def parse_vlm_detection_output(text: str, image_size: int = 224) -> List[Dict[str, Any]]:
    """
    Parses VLM generation string formatted like:
    <loc_120><loc_340><loc_450><loc_600> red circle ; <loc_50><loc_100><loc_200><loc_300> blue square
    into structured detections.
    """
    results = []
    # Pattern to match 4 location tokens followed by description
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

        # Detect color words in description
        found_color = "unknown"
        for color_name in COLOR_SPECTRUM.keys():
            if color_name in desc.lower():
                found_color = color_name
                break

        results.append({
            "bbox": [y1, x1, y2, x2],
            "norm_bbox": [norm_y1, norm_x1, norm_y2, norm_x2],
            "description": desc,
            "color": found_color,
            "hex": rgb_to_hex(*COLOR_SPECTRUM.get(found_color, (128, 128, 128))),
        })

    return results


def draw_detections_and_palette(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    output_path: str = "output_detection.png",
) -> Image.Image:
    """
    Draws bounding boxes, labels, and an attached color-palette summary bar.
    """
    img_w, img_h = image.size
    palette_width = 160
    total_w = img_w + palette_width

    # Create canvas with extra width on the right for color palette
    canvas = Image.new("RGB", (total_w, img_h), (245, 245, 248))
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)

    # 1. Draw Detections
    for det in detections:
        y1, x1, y2, x2 = det["bbox"]
        color_name = det.get("color", "unknown")
        box_rgb = COLOR_SPECTRUM.get(color_name, (255, 0, 0))

        # Draw thick box
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=box_rgb,
            )

        # Draw label background tag
        label_text = f"{det['description']}"
        text_bbox = draw.textbbox((x1, max(0, y1 - 18)), label_text)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2],
            fill=box_rgb,
        )
        draw.text((x1 + 1, max(0, y1 - 18)), label_text, fill=(255, 255, 255))

    # 2. Draw Color Palette Sidebar
    draw.rectangle([img_w, 0, total_w, img_h], fill=(240, 242, 245), outline=(200, 200, 200))
    draw.text((img_w + 12, 10), "DETECTED COLORS", fill=(40, 40, 40))

    y_offset = 35
    seen_colors = set()

    for det in detections:
        c_name = det.get("color", "unknown")
        if c_name in seen_colors or c_name == "unknown":
            continue
        seen_colors.add(c_name)

        c_rgb = COLOR_SPECTRUM.get(c_name, (128, 128, 128))
        c_hex = det.get("hex", "#000000")

        # Color swatch box
        draw.rectangle(
            [img_w + 12, y_offset, img_w + 36, y_offset + 24],
            fill=c_rgb,
            outline=(80, 80, 80),
        )
        # Color title & hex
        draw.text((img_w + 44, y_offset + 1), c_name.capitalize(), fill=(20, 20, 20))
        draw.text((img_w + 44, y_offset + 14), c_hex, fill=(100, 100, 100))

        y_offset += 32
        if y_offset > img_h - 40:
            break

    canvas.save(output_path)
    return canvas

