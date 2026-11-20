"""
Configuration for Strikler: Multimodal Vision-Language Model
for Object Detection and Color Grounding.
Supports staged training from Strikler-100M up to Strikler-3B.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    embed_dim: int = 384
    depth: int = 10
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass
class ProjectorConfig:
    vision_dim: int = 384
    llm_dim: int = 640
    hidden_dim: int = 1024


@dataclass
class LLMConfig:
    vocab_size: int = 32000
    hidden_dim: int = 640
    num_layers: int = 14
    num_heads: int = 10
    num_kv_heads: int = 2  # Grouped Query Attention (GQA)
    intermediate_dim: int = 1792  # SwiGLU FFN
    max_seq_len: int = 2048
    num_loc_tokens: int = 1000  # For normalized bounding boxes [0, 999]
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 500
    max_steps: int = 5000
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    bf16: bool = True
    save_every_steps: int = 500
    eval_every_steps: int = 250
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str = "cuda"


def get_model_preset(size: str = "100M") -> Tuple[VisionConfig, ProjectorConfig, LLMConfig]:
    """
    Returns architecture configurations scaled for:
      • 100M (~102M parameters) - Fast, lightweight, staged piece 1
      • 300M (~303M parameters) - Mid-scale baseline
      • 500M (~506M parameters) - Ideal high-efficiency sweet spot
      • 1B   (~1.1B parameters) - Medium foundation scale
      • 3B   (~3.05B parameters) - Full flagship scale for multi-day DGX runs
    """
    size = size.upper().strip()
    if size == "100M":
        v_cfg = VisionConfig(
            embed_dim=384,
            depth=10,
            num_heads=6,
        )
        p_cfg = ProjectorConfig(
            vision_dim=384,
            llm_dim=640,
            hidden_dim=1024,
        )
        l_cfg = LLMConfig(
            vocab_size=32000,
            hidden_dim=640,
            num_layers=14,
            num_heads=10,
            num_kv_heads=2,
            intermediate_dim=1792,
            max_seq_len=2048,
        )
    elif size == "300M":
        v_cfg = VisionConfig(
            embed_dim=768,
            depth=12,
            num_heads=12,
        )
        p_cfg = ProjectorConfig(
            vision_dim=768,
            llm_dim=1024,
            hidden_dim=2048,
        )
        l_cfg = LLMConfig(
            vocab_size=32000,
            hidden_dim=1024,
            num_layers=16,
            num_heads=16,
            num_kv_heads=4,
            intermediate_dim=2816,
            max_seq_len=2048,
        )
    elif size == "500M":
        v_cfg = VisionConfig(
            embed_dim=768,
            depth=12,
            num_heads=12,
        )
        p_cfg = ProjectorConfig(
            vision_dim=768,
            llm_dim=1280,
            hidden_dim=2560,
        )
        l_cfg = LLMConfig(
            vocab_size=32000,
            hidden_dim=1280,
            num_layers=20,
            num_heads=10,
            num_kv_heads=2,
            intermediate_dim=3840,
            max_seq_len=2048,
        )
    elif size == "1B":
        v_cfg = VisionConfig(
            embed_dim=768,
            depth=16,
            num_heads=12,
        )
        p_cfg = ProjectorConfig(
            vision_dim=768,
            llm_dim=1792,
            hidden_dim=3072,
        )
        l_cfg = LLMConfig(
            vocab_size=32000,
            hidden_dim=1792,
            num_layers=24,
            num_heads=14,
            num_kv_heads=2,
            intermediate_dim=4864,
            max_seq_len=2048,
        )
    elif size == "3B":
        v_cfg = VisionConfig(
            embed_dim=1024,
            depth=24,
            num_heads=16,
        )
        p_cfg = ProjectorConfig(
            vision_dim=1024,
            llm_dim=2816,
            hidden_dim=4096,
        )
        l_cfg = LLMConfig(
            vocab_size=32000,
            hidden_dim=2816,
            num_layers=32,
            num_heads=22,
            num_kv_heads=4,
            intermediate_dim=8192,
            max_seq_len=2048,
        )
    else:
        raise ValueError(f"Unknown model preset size '{size}'. Choose from: 100M, 300M, 1B, 3B")

    return v_cfg, p_cfg, l_cfg


@dataclass
class ColorPaletteConfig:
    color_shades: List[str] = field(default_factory=lambda: [
        # Red shades
        "blood red", "wine red", "burgundy", "crimson", "ruby red", "scarlet", "carmine", "maroon", "cherry red",
        # Blue shades
        "turquoise", "cyan", "teal", "sky blue", "baby blue", "cerulean", "cobalt blue", "royal blue", "sapphire blue", "navy blue", "midnight blue",
        # Green shades
        "emerald green", "forest green", "olive green", "mint green", "sage green", "lime green", "jade green", "seafoam green",
        # Yellow & Orange shades
        "mustard yellow", "amber", "canary yellow", "lemon yellow", "gold", "tangerine", "coral", "peach", "rust", "terracotta",
        # Purple & Pink shades
        "lavender", "violet", "plum", "magenta", "lilac", "mauve", "fuchsia", "hot pink", "blush pink",
        # Browns & Neutrals
        "chocolate brown", "espresso", "caramel", "tan", "beige", "khaki",
        # Monochromes
        "charcoal", "slate gray", "ash gray", "silver", "pearl white", "jet black"
    ])
