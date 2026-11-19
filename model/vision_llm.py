"""
Strikler: Unified Multimodal Vision-Language Model
for Object Detection and Color Grounding.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VisionConfig, ProjectorConfig, LLMConfig
from model.vision_encoder import VisionEncoder
from model.projector import MultimodalProjector
from model.llm_backbone import LLMBackbone


class StriklerVLM(nn.Module):
    """
    Strikler End-to-End Multimodal Vision-Language Model.
    Architecture:
      Image -> ViT Encoder -> Projector -> Visual Tokens
      Prompt -> Token Embeddings --------> Text Tokens
      [Visual Tokens | Text Tokens] -> LLM Backbone -> Coordinate & Language Tokens
    """

    def __init__(
        self,
        vision_config: VisionConfig = None,
        projector_config: ProjectorConfig = None,
        llm_config: LLMConfig = None,
    ):
        super().__init__()
        self.vision_config = vision_config or VisionConfig()
        self.projector_config = projector_config or ProjectorConfig()
        self.llm_config = llm_config or LLMConfig()

        self.vision_encoder = VisionEncoder(self.vision_config)
        self.projector = MultimodalProjector(self.projector_config)
        self.llm = LLMBackbone(self.llm_config)

        # Coordinate tokens offset: tokens from vocab_size to vocab_size + num_loc_tokens
        # represent normalized coordinates 0 to 999.
        self.loc_offset = self.llm_config.vocab_size

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Extract and project visual patch tokens."""
        visual_features = self.vision_encoder(images)  # (B, N_patches, Vision_Dim)
        visual_tokens = self.projector(visual_features)  # (B, N_patches, LLM_Dim)
        return visual_tokens

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        images: (B, 3, H, W)
        input_ids: (B, Seq_Len)
        labels: Optional (B, Seq_Len) with -100 for ignored positions (e.g. prompt/image)
        """
        visual_tokens = self.encode_images(images)  # (B, N_patches, LLM_Dim)
        text_embeddings = self.llm.token_embed(input_ids)  # (B, Seq_Len, LLM_Dim)

        # Prepend visual tokens before text tokens
        multimodal_embeddings = torch.cat([visual_tokens, text_embeddings], dim=1)

        logits = self.llm(multimodal_embeddings)  # (B, N_patches + Seq_Len, Total_Vocab)

        loss = None
        if labels is not None:
            # Shift targets for next-token prediction
            num_patches = visual_tokens.shape[1]
            # Logits for text generation start at visual_tokens end - 1
            pred_logits = logits[:, num_patches - 1 : -1, :].contiguous()
            loss = F.cross_entropy(
                pred_logits.view(-1, pred_logits.size(-1)),
                labels.contiguous().view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressively generate detection and color tokens given an image and prompt.
        """
        self.eval()
        visual_tokens = self.encode_images(images)
        text_embeddings = self.llm.token_embed(prompt_ids)
        current_embeddings = torch.cat([visual_tokens, text_embeddings], dim=1)

        generated_ids = []
        curr_len = current_embeddings.shape[1]

        for _ in range(max_new_tokens):
            if curr_len >= self.llm_config.max_seq_len:
                break

            logits = self.llm(current_embeddings)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids.append(next_token)

            # Embed next token and append
            next_embed = self.llm.token_embed(next_token)
            current_embeddings = torch.cat([current_embeddings, next_embed], dim=1)
            curr_len += 1

        if len(generated_ids) > 0:
            return torch.cat(generated_ids, dim=1)
        return torch.empty((images.size(0), 0), dtype=torch.long, device=images.device)


