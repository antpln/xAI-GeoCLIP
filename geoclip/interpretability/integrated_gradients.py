"""
Integrated Gradients for GeoCLIP.

Integrated Gradients (IG) computes pixel-level attribution by averaging
gradients along the straight-line path from a baseline input (grey image)
to the actual input:

    IG(x) = (x - x') × (1/N) Σ_{k=1}^{N} ∂f/∂x (x' + k/N · (x - x'))

Unlike Grad-CAM, IG satisfies the *completeness axiom*:
    Σ_i IG_i(x) = f(x) - f(x')

meaning the attributions sum exactly to the difference in model output between
the input and the baseline.  This makes them verifiable — you can check that
the highlighted pixels actually account for the prediction.

Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class IntegratedGradients:
    """
    Pixel-level attribution via Integrated Gradients.

    Gradients are taken w.r.t. the input image pixels (not internal
    activations), giving a [B, 3, H, W] attribution that is then collapsed
    to a [B, H, W] spatial map by summing absolute values across channels.

    Args:
        model:          GeoCLIP model.
        n_steps:        Number of interpolation steps (50 is standard).
        baseline_value: Fill value for the baseline image.  0.0 ≈ grey in
                        CLIP-normalized space (mean colour of natural images).
    """

    def __init__(self, model, n_steps: int = 50, baseline_value: float = 0.0):
        self.model = model
        self.n_steps = n_steps
        self.baseline_value = baseline_value

    def compute(
        self,
        images: torch.Tensor,
        target_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients attribution maps.

        Args:
            images:       [B, 3, H, W] CLIP-normalized images.
            target_coords:[B, 2] target (lat, lon) in degrees.

        Returns:
            [B, H, W] attribution maps normalized to [0, 1].
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        images = images.to(device)
        target_coords = target_coords.to(device)

        baseline = torch.full_like(images, self.baseline_value)

        # Pre-compute GPS embedding once — it doesn't change across steps
        with torch.no_grad():
            gps_emb = self.model.encode_gps(target_coords)  # [B, D]

        accumulated_grads = torch.zeros_like(images)

        for step in range(1, self.n_steps + 1):
            alpha = step / self.n_steps
            interp = (baseline + alpha * (images - baseline)).detach().requires_grad_(True)

            img_emb = self.model.image_encoder(interp)              # [B, D]
            score = (img_emb * gps_emb.detach()).sum()
            score.backward()

            accumulated_grads += interp.grad.detach()

        avg_grads = accumulated_grads / self.n_steps

        # IG = (input - baseline) × average gradient  [B, 3, H, W]
        attributions = (images.detach() - baseline) * avg_grads

        # Collapse channels: sum of absolute attributions per pixel [B, H, W]
        attr_map = attributions.abs().sum(dim=1)

        # Per-sample normalization to [0, 1]
        a_min = attr_map.flatten(1).min(1).values[:, None, None]
        a_max = attr_map.flatten(1).max(1).values[:, None, None]
        attr_map = (attr_map - a_min) / (a_max - a_min + 1e-8)

        return attr_map.cpu()
