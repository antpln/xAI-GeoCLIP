"""
Image encoder for GeoCLIP: our VisionTransformer + a trainable projection head.

The ViT backbone is our own implementation (vit.py). We optionally initialise
it with CLIP pre-trained weights so that we start from a strong visual
representation without importing any higher-level CLIP abstraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from geoclip.models.vit import VisionTransformer, VIT_CONFIGS, load_clip_weights


class CLIPImageEncoder(nn.Module):
    """
    Our ViT backbone + a trainable projection head.

    The projection head (hidden_dim → embedding_dim) is always randomly
    initialised and fine-tuned as part of GeoCLIP training.  It decouples
    the embedding dimension from the backbone's hidden size, which lets us
    align image and GPS embeddings in a common space of our choosing.

    Args:
        clip_model_name:    Architecture variant ("ViT-B/16", "ViT-B/32", "ViT-L/14").
        freeze_layers:      Number of early ViT blocks to freeze (0 = all trainable).
        embedding_dim:      Output dimension after projection.
        pretrained:         If True, initialise ViT weights from CLIP via HuggingFace.
                            If False, use random initialisation (useful for ablations).
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/16",
        freeze_layers:   int = 9,
        embedding_dim:   int = 512,
        pretrained:      bool = True,
    ):
        super().__init__()
        cfg = VIT_CONFIGS[clip_model_name]
        self.vit = VisionTransformer(image_size=224, in_channels=3, **cfg)

        if pretrained:
            load_clip_weights(self.vit, clip_model_name)

        hidden_dim = cfg["hidden_dim"]
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self._embedding_dim = embedding_dim
        self._freeze_early_layers(freeze_layers)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _freeze_early_layers(self, n_freeze: int) -> None:
        """Freeze patch embedding, positional embedding, and the first n_freeze blocks."""
        for param in self.vit.patch_embed.parameters():
            param.requires_grad_(False)
        self.vit.cls_token.requires_grad_(False)
        self.vit.pos_embed.requires_grad_(False)

        for i, block in enumerate(self.vit.blocks):
            if i < n_freeze:
                for param in block.parameters():
                    param.requires_grad_(False)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[ImageEncoder] {trainable:,} / {total:,} parameters trainable "
            f"({100 * trainable / total:.1f}%)"
        )

    def forward(
        self,
        images: torch.Tensor,
        output_attentions:    bool = False,
        output_hidden_states: bool = False,
    ):
        """
        Args:
            images:               [B, 3, 224, 224] CLIP-normalized float32.
            output_attentions:    Also return per-layer [B, heads, S, S] maps.
            output_hidden_states: Also return per-layer [B, S, D] hidden states.

        Returns:
            emb: [B, embedding_dim] L2-normalized.
            extras (dict, only when requested): keys "attentions", "hidden_states".
        """
        if output_attentions or output_hidden_states:
            cls_hidden, extras = self.vit(
                images,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            emb = F.normalize(self.projection(cls_hidden), dim=-1)
            return emb, extras
        else:
            cls_hidden = self.vit(images)
            return F.normalize(self.projection(cls_hidden), dim=-1)
