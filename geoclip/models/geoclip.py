import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from geoclip.models.image_encoder import CLIPImageEncoder
from geoclip.models.gps_encoder import LocationEncoder


class GeoCLIP(nn.Module):
    """
    GeoCLIP: contrastive image-to-GPS alignment.

    Aligns CLIP image embeddings with GPS location embeddings via InfoNCE
    contrastive loss. A learnable log-temperature is trained alongside both
    encoders.

    Image encoder: our VisionTransformer (vit.py) + explicit projection head.
    GPS encoder:   Multi-scale Random Fourier Features + MLP.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/16",
        freeze_layers: int = 9,
        rff_num_scales: int = 10,
        rff_dim: int = 256,
        mlp_hidden: int = 1024,
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.image_encoder = CLIPImageEncoder(
            clip_model_name=clip_model_name,
            freeze_layers=freeze_layers,
            embedding_dim=embedding_dim,
        )
        self.gps_encoder = LocationEncoder(
            num_scales=rff_num_scales,
            rff_dim=rff_dim,
            mlp_hidden=mlp_hidden,
            embedding_dim=embedding_dim,
        )
        # Learnable log temperature; init at log(1/0.07) ≈ 2.66 (CLIP default)
        self.log_logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / 0.07))
        )

    def encode_image(
        self,
        images: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """
        Returns [B, D] L2-normalized image embeddings.
        Optionally returns per-layer attention maps / hidden states for interpretability.
        """
        return self.image_encoder(
            images,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

    def encode_gps(self, coords: torch.Tensor) -> torch.Tensor:
        """Returns [B, D] L2-normalized GPS embeddings."""
        return self.gps_encoder(coords)

    def forward(self, images: torch.Tensor, coords: torch.Tensor):
        """
        Args:
            images: [B, 3, 224, 224] preprocessed images.
            coords: [B, 2] (lat, lon) in degrees.

        Returns:
            img_emb:     [B, D] normalized image embeddings.
            gps_emb:     [B, D] normalized GPS embeddings.
            logit_scale: scalar temperature (exponentiated, clamped).
        """
        img_emb = self.image_encoder(images)
        gps_emb = self.gps_encoder(coords)
        logit_scale = self.log_logit_scale.exp().clamp(max=100.0)
        return img_emb, gps_emb, logit_scale

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        gallery_coords: torch.Tensor,
        gallery_embs: torch.Tensor = None,
        top_k: int = 1,
    ):
        """
        Predict GPS for a batch of images via gallery retrieval.

        Args:
            images:        [B, 3, H, W] preprocessed images.
            gallery_coords:[G, 2] gallery GPS coordinates.
            gallery_embs:  [G, D] pre-computed GPS embeddings (reuse to skip re-encoding).
            top_k:         Number of candidates to return per image.

        Returns:
            coords: [B, top_k, 2] predicted GPS coordinates.
            scores: [B, top_k]    cosine similarity scores.
        """
        self.eval()
        dev = next(self.parameters()).device
        if gallery_embs is None:
            gallery_embs = self.encode_gps(gallery_coords.to(dev))
        img_embs = self.encode_image(images.to(dev))
        sims = img_embs @ gallery_embs.T                    # [B, G]
        top_scores, top_idx = sims.topk(top_k, dim=-1)     # [B, top_k]
        return gallery_coords[top_idx.cpu()], top_scores.cpu()

    def clamp_temperature(self) -> None:
        """Must be called after each optimizer step to prevent degenerate temperatures."""
        with torch.no_grad():
            self.log_logit_scale.clamp_(-4.6, 4.6)
