"""
Color detection and color analysis utilities.
Maps RGB values to standard color names and extracts color palettes from bounding boxes.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image

# Standard RGB color palette for grounding
COLOR_SPECTRUM: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 20, 60),
    "crimson": (178, 34, 34),
    "maroon": (128, 0, 0),
    "pink": (255, 105, 180),
    "orange": (255, 140, 0),
    "gold": (255, 215, 0),
    "yellow": (255, 255, 0),
    "lime": (50, 205, 50),
    "green": (34, 139, 34),
    "emerald": (0, 201, 87),
    "olive": (107, 142, 35),
    "cyan": (0, 255, 255),
    "teal": (0, 128, 128),
    "sky blue": (135, 206, 235),
    "blue": (30, 144, 255),
    "navy": (0, 0, 128),
    "purple": (128, 0, 128),
    "violet": (138, 43, 226),
    "magenta": (255, 0, 255),
    "brown": (139, 69, 19),
    "beige": (245, 245, 220),
    "tan": (210, 180, 140),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "silver": (192, 192, 192),
    "black": (20, 20, 20),
}


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB tuple to Hex string."""
    return f"#{r:02x}{g:02x}{b:02x}"


def match_color_name(rgb: Tuple[int, int, int]) -> str:
    """Find the closest human-readable color name using Euclidean distance in RGB/Lab space."""
    r, g, b = rgb
    min_dist = float("inf")
    best_color = "unknown"

    for color_name, (cr, cg, cb) in COLOR_SPECTRUM.items():
        # Weighted Euclidean distance (human eye is more sensitive to green)
        dist = 0.3 * (r - cr) ** 2 + 0.59 * (g - cg) ** 2 + 0.11 * (b - cb) ** 2
        if dist < min_dist:
            min_dist = dist
            best_color = color_name

    return best_color


def extract_box_dominant_color(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
) -> Dict[str, Any]:
    """
    Extract the dominant color from a cropped bounding box [ymin, xmin, ymax, xmax].
    Returns color name, average RGB, and Hex code.
    """
    w, h = image.size
    ymin, xmin, ymax, xmax = bbox

    # Clamp coordinates
    ymin = max(0, min(h - 1, int(ymin)))
    ymax = max(ymin + 1, min(h, int(ymax)))
    xmin = max(0, min(w - 1, int(xmin)))
    xmax = max(xmin + 1, min(w, int(xmax)))

    crop = image.crop((xmin, ymin, xmax, ymax))
    arr = np.array(crop.convert("RGB"))

    # Compute median / mean color of inner region (avoiding edge background)
    h_c, w_c, _ = arr.shape
    y1, y2 = int(0.2 * h_c), max(int(0.8 * h_c), 1)
    x1, x2 = int(0.2 * w_c), max(int(0.8 * w_c), 1)
    inner = arr[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else arr

    avg_rgb = tuple(np.mean(inner, axis=(0, 1)).astype(int))
    color_name = match_color_name(avg_rgb)
    hex_code = rgb_to_hex(*avg_rgb)

    return {
        "color_name": color_name,
        "rgb": avg_rgb,
        "hex": hex_code,
    }
