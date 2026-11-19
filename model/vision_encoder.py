"""
Vision Transformer (ViT) Encoder for visual feature extraction.
"""

import math
import torch
import torch.nn as nn
from config import VisionConfig


class PatchEmbedding(nn.Module):
    """Splits image into non-overlapping patches and projects to embedding space."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, Embed_dim, H/P, W/P)
        x = self.projection(x)
        # Flatten spatial dimensions: (B, Embed_dim, N_patches)
        x = x.flatten(2)
        # Transpose to (B, N_patches, Embed_dim)
        x = x.transpose(1, 2)
        return x


class ViTAttention(nn.Module):
    """Multi-Head Self-Attention for Vision Transformer."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3, bias=True)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.drop(out)
        return out


class ViTMLP(nn.Module):
    """Feedforward layer with GELU activation."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTBlock(nn.Module):
    """Standard Vision Transformer encoder block with pre-LayerNorm."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.attn = ViTAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.mlp = ViTMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """Complete Vision Transformer encoder module."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_embed = PatchEmbedding(config)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([ViTBlock(config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Return all patch tokens (excluding [CLS] token, or including)
        # Patch tokens represent localized spatial regions essential for detection & color
        return x[:, 1:, :]

