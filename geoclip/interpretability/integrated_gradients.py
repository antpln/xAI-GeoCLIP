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
from tqdm import tqdm


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

        B = images.shape[0]
        all_attrs = []

        for b in tqdm(range(B), desc="IG samples", leave=False):
            img_b      = images[b:b+1]
            baseline_b = baseline[b:b+1]
            gps_b      = gps_emb[b:b+1]
            accum      = torch.zeros_like(img_b)

            for step in tqdm(range(1, self.n_steps + 1), desc="steps", leave=False):
                alpha  = step / self.n_steps
                interp = (baseline_b + alpha * (img_b - baseline_b)).detach().requires_grad_(True)
                score  = (self.model.image_encoder(interp) * gps_b.detach()).sum()
                score.backward()
                accum += interp.grad.detach()

            avg_grads = accum / self.n_steps
            all_attrs.append((img_b.detach() - baseline_b) * avg_grads)

        # IG = (input - baseline) × average gradient  [B, 3, H, W]
        attributions = torch.cat(all_attrs, dim=0)

        # Collapse channels: sum of absolute attributions per pixel [B, H, W]
        attr_map = attributions.abs().sum(dim=1)

        # Average within each patch to remove the Conv2d stride artifact:
        # pixel-level gradients through a strided patch embedding are identical
        # within each patch, producing a visible grid. Pooling then upsampling
        # replaces that grid with a smooth per-patch saliency map.
        # Pool to patch resolution then upsample — removes the Conv2d stride artifact
        patch_size = self.model.image_encoder.vit.patch_embed.proj.kernel_size[0]
        H, W = images.shape[-2:]
        pooled   = F.avg_pool2d(attr_map.unsqueeze(1), kernel_size=patch_size, stride=patch_size)
        attr_map = F.interpolate(pooled, size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        # Gaussian blur to smooth hard patch boundaries
        sigma  = patch_size / 2.0
        ks     = patch_size | 1  # next odd number ≥ patch_size
        coords = torch.arange(ks, dtype=torch.float32) - ks // 2
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]      # [ks, ks]
        kernel_2d = kernel_2d[None, None].to(attr_map.device)    # [1, 1, ks, ks]
        attr_map  = F.conv2d(attr_map.unsqueeze(1), kernel_2d,
                             padding=ks // 2).squeeze(1)

        # Per-sample normalization to [0, 1]
        a_min = attr_map.flatten(1).min(1).values[:, None, None]
        a_max = attr_map.flatten(1).max(1).values[:, None, None]
        attr_map = (attr_map - a_min) / (a_max - a_min + 1e-8)

        return attr_map.cpu()
