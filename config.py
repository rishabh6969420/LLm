"""
Configuration for ChromaDetect-VLM: Multimodal Vision-Language Model
for Object Detection and Color Grounding.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass
class ProjectorConfig:
    vision_dim: int = 768
    llm_dim: int = 1024
    hidden_dim: int = 2048


@dataclass
class LLMConfig:
    vocab_size: int = 32000
    hidden_dim: int = 1024
    num_layers: int = 16
    num_heads: int = 16
    num_kv_heads: int = 4  # Grouped Query Attention (GQA)
    intermediate_dim: int = 2816  # SwiGLU FFN
    max_seq_len: int = 1024
    num_loc_tokens: int = 1000  # For normalized bounding boxes [0, 999]
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 2000
    max_steps: int = 100000  # Can run for 40-48 hours on DGX GB10
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    bf16: bool = True
    save_every_steps: int = 5000
    eval_every_steps: int = 1000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str = "cuda"


@dataclass
class ColorPaletteConfig:
    common_colors: List[str] = field(default_factory=lambda: [
        "red", "crimson", "maroon", "pink", "rose",
        "orange", "coral", "amber", "gold", "yellow",
        "lime", "green", "emerald", "olive", "forest green",
        "cyan", "teal", "turquoise", "sky blue", "blue", "navy",
        "purple", "violet", "indigo", "magenta",
        "brown", "beige", "tan", "khaki",
        "white", "gray", "silver", "charcoal", "black"
    ])
