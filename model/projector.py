"""
Multimodal Vision-to-Language Projector.
Projects visual tokens into the embedding space of the LLM.
"""

import torch
import torch.nn as nn
from config import ProjectorConfig


class MultimodalProjector(nn.Module):
    """
    Two-layer MLP projector with GELU activation that aligns
    visual patch embeddings to the LLM's text token hidden space.
    """

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.vision_dim, config.hidden_dim, bias=True)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(config.hidden_dim, config.llm_dim, bias=True)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        # visual_tokens: (Batch, N_patches, Vision_Dim)
        x = self.linear1(visual_tokens)
        x = self.act(x)
        x = self.linear2(x)
        # Returns: (Batch, N_patches, LLM_Dim)
        return x
