"""
Occlusion sensitivity maps for GeoCLIP.

Systematically occlude each image patch and measure how much the predicted
GPS location shifts.  Unlike Grad-CAM, this method:
  - Is gradient-free (no backpropagation required)
  - Makes no linear approximation of the model's behaviour
  - Directly measures the causal effect of each patch on the prediction

When occlusion sensitivity and Grad-CAM highlight the same regions, this
provides strong evidence that the model is genuinely using those patches
for geo-localization rather than responding to gradient artifacts.

Patch layout:
  For ViT-B/16: image 224×224, patch 16×16 → 14×14 = 196 patches.
  For ViT-B/32: image 224×224, patch 32×32 → 7×7  =  49 patches.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from geoclip.utils.geo_math import haversine_distance


class OcclusionSensitivity:
    """
    Patch-level occlusion sensitivity for the GeoCLIP image encoder.

    For each patch position, the patch pixels are replaced by a constant
    fill value (default: mean of the CLIP normalisation, i.e. grey in
    normalised space = 0), and the change in image embedding is recorded.
    The sensitivity score for each patch is the cosine distance between the
    original embedding and the occluded embedding.

    This is "perturbation-based" saliency: high sensitivity = this patch
    strongly influences the representation used for geo-localization.

    Args:
        model:      GeoCLIP model.
        patch_size: Pixel side-length of each patch (must match the ViT config).
        fill_value: Value to fill occluded patches with (in normalised space).
                    0.0 corresponds to the CLIP mean colour (grey).
    """

    def __init__(self, model, patch_size: int = 16, fill_value: float = 0.0):
        self.model = model
        self.patch_size = patch_size
        self.fill_value = fill_value

    @torch.no_grad()
    def compute(
        self,
        images: torch.Tensor,
        gallery_coords: Optional[torch.Tensor] = None,
        gallery_embs:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute occlusion sensitivity maps.

        Two scoring modes:
          - Embedding mode (gallery_coords/gallery_embs not provided):
            sensitivity = 1 - cosine_similarity(original_emb, occluded_emb).
            Measures how much each patch changes the image representation.

          - GPS mode (gallery provided):
            sensitivity = haversine_distance(original_pred, occluded_pred).
            Measures how much each patch changes the predicted GPS location (km).
            More directly tied to geo-localization than embedding distance.

        Args:
            images:         [B, 3, H, W] CLIP-normalized images.
            gallery_coords: [G, 2] GPS gallery coordinates (optional).
            gallery_embs:   [G, D] pre-computed gallery embeddings (optional).

        Returns:
            [B, H, W] sensitivity maps (same resolution as input), in [0, 1].
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        images = images.to(device)
        B, C, H, W = images.shape
        p = self.patch_size

        grid_h = H // p
        grid_w = W // p
        n_patches = grid_h * grid_w

        use_gps_mode = (gallery_coords is not None and gallery_embs is not None)

        # Baseline: original image embeddings / predicted locations
        baseline_emb = self.model.encode_image(images)   # [B, D]

        if use_gps_mode:
            gallery_embs = gallery_embs.to(device)
            baseline_pred = self._predict_coords(baseline_emb, gallery_coords, gallery_embs)

        # Score for each patch position
        scores = torch.zeros(B, n_patches, device=device)

        for patch_idx in range(n_patches):
            row = patch_idx // grid_w
            col = patch_idx  % grid_w
            r0, r1 = row * p, (row + 1) * p
            c0, c1 = col * p, (col + 1) * p

            # Occlude patch
            occluded = images.clone()
            occluded[:, :, r0:r1, c0:c1] = self.fill_value

            occ_emb = self.model.encode_image(occluded)   # [B, D]

            if use_gps_mode:
                occ_pred = self._predict_coords(occ_emb, gallery_coords, gallery_embs)
                # Sensitivity = GPS shift in km
                dist = haversine_distance(baseline_pred, occ_pred)  # [B]
                scores[:, patch_idx] = dist
            else:
                # Sensitivity = cosine distance (1 - similarity)
                sim = (baseline_emb * occ_emb).sum(dim=-1)          # [B]
                scores[:, patch_idx] = 1.0 - sim

        # Reshape to spatial grid
        scores = scores.reshape(B, 1, grid_h, grid_w)
        scores = F.interpolate(scores, size=(H, W), mode="nearest")
        scores = scores.squeeze(1)   # [B, H, W]

        # Normalize to [0, 1] per sample
        s_min = scores.flatten(1).min(1).values[:, None, None]
        s_max = scores.flatten(1).max(1).values[:, None, None]
        scores = (scores - s_min) / (s_max - s_min + 1e-8)

        return scores.cpu()

    def _predict_coords(
        self,
        img_emb: torch.Tensor,
        gallery_coords: torch.Tensor,
        gallery_embs: torch.Tensor,
    ) -> torch.Tensor:
        sims = img_emb @ gallery_embs.T          # [B, G]
        best = sims.argmax(dim=-1)               # [B]
        return gallery_coords[best.cpu()].to(img_emb.device)
