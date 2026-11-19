"""
Causal Transformer Decoder LLM Backbone with RoPE, RMSNorm, SwiGLU, and GQA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LLMConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA / Qwen)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex exponential frequencies for Rotary Position Embeddings (RoPE)."""
    dim_range = torch.arange(0, head_dim, 2).float()
    freqs = 1.0 / (theta ** (dim_range / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex tensor (max_seq_len, head_dim // 2)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to Query or Key tensors."""
    B, H, S, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, S, D // 2, 2))
    freqs_cis = freqs_cis[:S].unsqueeze(0).unsqueeze(1)  # (1, 1, S, D // 2)
    x_rotated = torch.view_as_real(x_complex * freqs_cis).reshape(B, H, S, D)
    return x_rotated.type_as(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feedforward Network (used in modern high-performance LLMs)."""

    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_dim, bias=False)  # gate
        self.w2 = nn.Linear(intermediate_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, intermediate_dim, bias=False)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) with RoPE and Causal Masking."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply Rotary Position Embeddings
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Repeat KV heads for GQA if num_kv_heads < num_heads
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Scaled Dot-Product Attention (PyTorch optimized FlashAttention/SDPA)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None and S > 1))
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)


class TransformerDecoderBlock(nn.Module):
    """Decoder block with RMSNorm, GQA, and SwiGLU."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.ffn = SwiGLUFFN(config.hidden_dim, config.intermediate_dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LLMBackbone(nn.Module):
    """The complete Decoder-Only Causal Language Model."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size + config.num_loc_tokens, config.hidden_dim)
        self.layers = nn.ModuleList([TransformerDecoderBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size + config.num_loc_tokens, bias=False)

        # Weight tying
        self.token_embed.weight = self.lm_head.weight

        head_dim = config.hidden_dim // config.num_heads
        freqs = precompute_rope_freqs(head_dim, config.max_seq_len)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Takes token or multimodal embeddings (Batch, Seq_Len, Hidden_Dim)
        and returns logits (Batch, Seq_Len, Vocab_Size + Loc_Tokens).
        """
        B, S, D = embeddings.shape
        freqs_cis = self.freqs_cis[:S].to(embeddings.device)

        x = embeddings
        for layer in self.layers:
            x = layer(x, freqs_cis)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
