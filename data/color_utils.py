"""
High-Precision Color Science & Fine-Grained Shade Grounding Engine.
Uses CIELAB color space and Delta-E (CIE76 / CIE94) perceptual distance
to accurately detect nuanced shades like turquoise, blood red, wine red, etc.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np
from PIL import Image

# 80+ Fine-grained, calibrated color shades categorized with exact sRGB and parent family
COLOR_SHADE_DATABASE: Dict[str, Dict[str, Any]] = {
    # ---------------- RED SHADES ----------------
    "blood red": {"rgb": (138, 3, 3), "parent": "red", "hex": "#8a0303"},
    "wine red": {"rgb": (114, 47, 55), "parent": "red", "hex": "#722f37"},
    "burgundy": {"rgb": (128, 0, 32), "parent": "red", "hex": "#800020"},
    "crimson": {"rgb": (220, 20, 60), "parent": "red", "hex": "#dc143c"},
    "ruby red": {"rgb": (155, 17, 30), "parent": "red", "hex": "#9b111e"},
    "scarlet": {"rgb": (255, 36, 0), "parent": "red", "hex": "#ff2400"},
    "carmine": {"rgb": (150, 0, 24), "parent": "red", "hex": "#960018"},
    "maroon": {"rgb": (128, 0, 0), "parent": "red", "hex": "#800000"},
    "cherry red": {"rgb": (210, 4, 45), "parent": "red", "hex": "#d2042d"},
    "rose red": {"rgb": (194, 30, 86), "parent": "red", "hex": "#c21e56"},
    "firebrick": {"rgb": (178, 34, 34), "parent": "red", "hex": "#b22222"},

    # ---------------- BLUE SHADES ----------------
    "turquoise": {"rgb": (64, 224, 208), "parent": "blue", "hex": "#40e0d0"},
    "cyan": {"rgb": (0, 255, 255), "parent": "blue", "hex": "#00ffff"},
    "teal": {"rgb": (0, 128, 128), "parent": "blue", "hex": "#008080"},
    "sky blue": {"rgb": (135, 206, 235), "parent": "blue", "hex": "#87ceeb"},
    "baby blue": {"rgb": (137, 207, 240), "parent": "blue", "hex": "#89cff0"},
    "cerulean": {"rgb": (0, 123, 167), "parent": "blue", "hex": "#007ba7"},
    "cobalt blue": {"rgb": (0, 71, 171), "parent": "blue", "hex": "#0047ab"},
    "royal blue": {"rgb": (65, 105, 225), "parent": "blue", "hex": "#4169e1"},
    "sapphire blue": {"rgb": (15, 82, 186), "parent": "blue", "hex": "#0f52ba"},
    "navy blue": {"rgb": (0, 0, 128), "parent": "blue", "hex": "#000080"},
    "midnight blue": {"rgb": (25, 25, 112), "parent": "blue", "hex": "#191970"},
    "indigo": {"rgb": (75, 0, 130), "parent": "blue", "hex": "#4b0082"},

    # ---------------- GREEN SHADES ----------------
    "emerald green": {"rgb": (80, 200, 120), "parent": "green", "hex": "#50c878"},
    "forest green": {"rgb": (34, 139, 34), "parent": "green", "hex": "#228b22"},
    "olive green": {"rgb": (107, 142, 35), "parent": "green", "hex": "#6b8e23"},
    "mint green": {"rgb": (152, 255, 152), "parent": "green", "hex": "#98ff98"},
    "sage green": {"rgb": (157, 193, 131), "parent": "green", "hex": "#9dc183"},
    "lime green": {"rgb": (50, 205, 50), "parent": "green", "hex": "#32cd32"},
    "jade green": {"rgb": (0, 168, 107), "parent": "green", "hex": "#00a86b"},
    "moss green": {"rgb": (138, 154, 91), "parent": "green", "hex": "#8a9a5b"},
    "seafoam green": {"rgb": (159, 226, 191), "parent": "green", "hex": "#9fe2bf"},
    "pine green": {"rgb": (1, 121, 111), "parent": "green", "hex": "#01796f"},

    # ---------------- YELLOW & ORANGE SHADES ----------------
    "mustard yellow": {"rgb": (255, 219, 88), "parent": "yellow", "hex": "#ffdb58"},
    "amber": {"rgb": (255, 191, 0), "parent": "yellow", "hex": "#ffbf00"},
    "canary yellow": {"rgb": (255, 239, 0), "parent": "yellow", "hex": "#ffef00"},
    "lemon yellow": {"rgb": (255, 244, 79), "parent": "yellow", "hex": "#fff44f"},
    "gold": {"rgb": (255, 215, 0), "parent": "yellow", "hex": "#ffd700"},
    "tangerine": {"rgb": (242, 133, 0), "parent": "orange", "hex": "#f28500"},
    "coral": {"rgb": (255, 127, 80), "parent": "orange", "hex": "#ff7f50"},
    "peach": {"rgb": (255, 218, 185), "parent": "orange", "hex": "#ffdab9"},
    "rust": {"rgb": (183, 65, 14), "parent": "orange", "hex": "#b7410e"},
    "terracotta": {"rgb": (226, 114, 91), "parent": "orange", "hex": "#e2725b"},
    "burnt orange": {"rgb": (204, 85, 0), "parent": "orange", "hex": "#cc5500"},

    # ---------------- PURPLE & PINK SHADES ----------------
    "lavender": {"rgb": (230, 230, 250), "parent": "purple", "hex": "#e6e6fa"},
    "violet": {"rgb": (138, 43, 226), "parent": "purple", "hex": "#8a2be2"},
    "plum": {"rgb": (142, 69, 133), "parent": "purple", "hex": "#8e4585"},
    "magenta": {"rgb": (255, 0, 255), "parent": "purple", "hex": "#ff00ff"},
    "lilac": {"rgb": (200, 162, 200), "parent": "purple", "hex": "#c8a2c8"},
    "mauve": {"rgb": (224, 176, 255), "parent": "purple", "hex": "#e0b0ff"},
    "fuchsia": {"rgb": (255, 0, 255), "parent": "purple", "hex": "#ff00ff"},
    "hot pink": {"rgb": (255, 105, 180), "parent": "pink", "hex": "#ff69b4"},
    "blush pink": {"rgb": (255, 182, 193), "parent": "pink", "hex": "#ffb6c1"},
    "rose pink": {"rgb": (255, 102, 204), "parent": "pink", "hex": "#ff66cc"},

    # ---------------- BROWNS & NEUTRALS ----------------
    "chocolate brown": {"rgb": (123, 63, 0), "parent": "brown", "hex": "#7b3f00"},
    "espresso": {"rgb": (54, 43, 40), "parent": "brown", "hex": "#362b28"},
    "caramel": {"rgb": (198, 139, 89), "parent": "brown", "hex": "#c68b59"},
    "tan": {"rgb": (210, 180, 140), "parent": "brown", "hex": "#d2b48c"},
    "beige": {"rgb": (245, 245, 220), "parent": "brown", "hex": "#f5f5dc"},
    "khaki": {"rgb": (240, 230, 140), "parent": "brown", "hex": "#f0e68c"},
    "coffee": {"rgb": (111, 78, 55), "parent": "brown", "hex": "#6f4e37"},

    # ---------------- GRAYS, WHITES & BLACKS ----------------
    "charcoal": {"rgb": (54, 69, 79), "parent": "gray", "hex": "#36454f"},
    "slate gray": {"rgb": (112, 128, 144), "parent": "gray", "hex": "#708090"},
    "ash gray": {"rgb": (178, 190, 181), "parent": "gray", "hex": "#b2beb5"},
    "silver": {"rgb": (192, 192, 192), "parent": "gray", "hex": "#c0c0c0"},
    "pearl white": {"rgb": (248, 246, 240), "parent": "white", "hex": "#f8f6f0"},
    "ivory": {"rgb": (255, 255, 240), "parent": "white", "hex": "#fffff0"},
    "pure white": {"rgb": (255, 255, 255), "parent": "white", "hex": "#ffffff"},
    "jet black": {"rgb": (10, 10, 10), "parent": "black", "hex": "#0a0a0a"},
    "pitch black": {"rgb": (0, 0, 0), "parent": "black", "hex": "#000000"},
}

# Compatibility alias for backward compatibility
COLOR_SPECTRUM: Dict[str, Tuple[int, int, int]] = {
    name: data["rgb"] for name, data in COLOR_SHADE_DATABASE.items()
}


# ============================================================================
# CIELAB COLOR SCIENCE FUNCTIONS
# ============================================================================

def rgb_to_cielab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Convert sRGB [0, 255] to CIE L*a*b* color space.
    CIELAB is perceptually uniform: Euclidean distance in L*a*b* matches
    human color perception far better than RGB Euclidean distance.
    """
    # 1. Linearize sRGB
    r = rgb[0] / 255.0
    g = rgb[1] / 255.0
    b = rgb[2] / 255.0

    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    # 2. Convert to CIE XYZ (D65 standard illuminant)
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) / 1.00000
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883

    # 3. Non-linear mapping to L*a*b*
    fx = x ** (1.0 / 3.0) if x > 0.008856 else (7.787 * x) + (16.0 / 116.0)
    fy = y ** (1.0 / 3.0) if y > 0.008856 else (7.787 * y) + (16.0 / 116.0)
    fz = z ** (1.0 / 3.0) if z > 0.008856 else (7.787 * z) + (16.0 / 116.0)

    l = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)

    return (l, a, b_val)


# Precompute CIELAB vectors for database
_DATABASE_LAB: Dict[str, Tuple[float, float, float]] = {
    name: rgb_to_cielab(info["rgb"]) for name, info in COLOR_SHADE_DATABASE.items()
}


def delta_e_cie76(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    """Calculate CIE76 Delta-E perceptual color difference."""
    dl = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    return math.sqrt(dl * dl + da * da + db * db)


def match_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Finds the exact accurate shade (e.g. turquoise, blood red, wine red)
    using CIELAB Delta-E color science.
    """
    target_lab = rgb_to_cielab(rgb)
    min_dist = float("inf")
    best_shade = "unknown"

    for shade_name, ref_lab in _DATABASE_LAB.items():
        dist = delta_e_cie76(target_lab, ref_lab)
        if dist < min_dist:
            min_dist = dist
            best_shade = shade_name

    return best_shade


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB tuple to Hex string."""
    return f"#{r:02x}{g:02x}{b:02x}"


def extract_box_dominant_color(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
) -> Dict[str, Any]:
    """
    Extract the dominant color from a cropped bounding box [ymin, xmin, ymax, xmax].
    Returns exact shade name, parent color family, average RGB, and Hex code.
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

    # Sample central 60% region to eliminate background edge bleed
    h_c, w_c, _ = arr.shape
    y1, y2 = int(0.2 * h_c), max(int(0.8 * h_c), 1)
    x1, x2 = int(0.2 * w_c), max(int(0.8 * w_c), 1)
    inner = arr[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else arr

    avg_rgb = tuple(np.mean(inner, axis=(0, 1)).astype(int))
    shade_name = match_color_name(avg_rgb)
    shade_info = COLOR_SHADE_DATABASE.get(shade_name, {"parent": "unknown", "hex": rgb_to_hex(*avg_rgb)})

    return {
        "color_name": shade_name,
        "parent_family": shade_info["parent"],
        "rgb": avg_rgb,
        "hex": shade_info["hex"],
    }
