import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationEncoder(nn.Module):
    """
    GPS location encoder using Random Fourier Features (RFF) at multiple scales.

    Encodes (lat, lon) coordinates into a fixed-size embedding that can be
    aligned with CLIP image embeddings via contrastive learning.

    Architecture (following GeoCLIP paper):
      - Multi-scale RFF: 10 frequency scales, 256 random features each
      - RFF dimension: 10 * 256 * 2 = 5120
      - MLP: 5120 -> 1024 -> 512 -> embedding_dim
    """

    def __init__(
        self,
        num_scales: int = 10,
        rff_dim: int = 256,
        mlp_hidden: int = 1024,
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.rff_dim = rff_dim
        self.embedding_dim = embedding_dim

        # Fixed random frequency matrices, one per scale — NOT updated by optimizer.
        # sigma_k = 2^k controls the spatial frequency: low k = coarse (continental),
        # high k = fine-grained (city-level).
        for k in range(num_scales):
            sigma = 2.0 ** k
            freq_matrix = torch.randn(rff_dim, 2) * sigma  # [rff_dim, 2]
            self.register_buffer(f"B_{k}", freq_matrix)

        rff_total_dim = num_scales * rff_dim * 2  # cos + sin per scale
        self.mlp = nn.Sequential(
            nn.Linear(rff_total_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden // 2),
            nn.Linear(mlp_hidden // 2, embedding_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [B, 2] tensor with (latitude, longitude) in degrees.

        Returns:
            [B, embedding_dim] L2-normalized embeddings.
        """
        # Normalize to [-1, 1]
        scale = torch.tensor([90.0, 180.0], device=coords.device, dtype=coords.dtype)
        x = coords / scale  # [B, 2]

        features = []
        for k in range(self.num_scales):
            freq_matrix = getattr(self, f"B_{k}")    # [rff_dim, 2]
            proj = x @ freq_matrix.T                  # [B, rff_dim]
            features.append(torch.cos(proj))
            features.append(torch.sin(proj))

        phi = torch.cat(features, dim=-1)  # [B, num_scales * rff_dim * 2]
        out = self.mlp(phi)                # [B, embedding_dim]
        return F.normalize(out, dim=-1)
