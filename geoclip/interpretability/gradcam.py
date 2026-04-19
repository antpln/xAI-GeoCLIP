"""
Grad-CAM for our VisionTransformer.

Hooks into a chosen transformer block, back-propagates the image–GPS cosine
similarity, and weights patch activations by their mean gradient.
Token layout: index 0 = CLS, indices [1:] = patch tokens.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM heatmaps for GeoCLIP. Use via gradcam_context() to auto-remove hooks."""

    def __init__(self, model, target_layer_idx: int = -1):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Hook the chosen transformer block (default: last)
        target = model.image_encoder.vit.blocks[target_layer_idx]
        self._fwd_hook = target.register_forward_hook(self._save_activation)
        self._bwd_hook = target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # TransformerBlock returns (hidden, attn_weights); skip CLS at index 0
        self._activations = output[0][:, 1:, :].detach()  # [B, num_patches, D]

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0][:, 1:, :].detach()  # [B, num_patches, D]

    def compute(
        self,
        images: torch.Tensor,
        target_coords: torch.Tensor,
        relu: bool = True,
    ) -> torch.Tensor:
        """Returns [B, H, W] heatmaps in [0, 1]."""
        self.model.eval()
        device = next(self.model.parameters()).device
        images        = images.to(device).requires_grad_(False)
        target_coords = target_coords.to(device)

        for p in self.model.image_encoder.parameters():
            p.requires_grad_(True)

        img_emb = self.model.image_encoder(images)       # [B, D]
        gps_emb = self.model.encode_gps(target_coords)  # [B, D]
        score = (img_emb * gps_emb).sum()               # scalar cosine similarity
        score.backward()

        # Weight each patch activation by its mean gradient across the feature dimension
        weights = self._gradients.mean(dim=-1, keepdim=True)  # [B, num_patches, 1]
        cam = (weights * self._activations).sum(dim=-1)        # [B, num_patches]

        if relu:
            cam = F.relu(cam)

        # Reshape flat patch scores to a 2D grid, then upsample to image resolution
        batch_size = cam.shape[0]
        n_patches  = cam.shape[1]
        grid_size  = int(n_patches ** 0.5)
        cam = cam.reshape(batch_size, 1, grid_size, grid_size)
        H, W = images.shape[-2:]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)  # [B, H, W]

        # Normalize each sample independently to [0, 1]
        cam_min = cam.flatten(1).min(1).values[:, None, None]
        cam_max = cam.flatten(1).max(1).values[:, None, None]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        self.model.zero_grad()
        return cam.detach().cpu()

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()


@contextmanager
def gradcam_context(model, target_layer_idx: int = -1):
    """Context manager that registers hooks on entry and removes them on exit."""
    gc = GradCAM(model, target_layer_idx)
    try:
        yield gc
    finally:
        gc.remove_hooks()


def gradcam_layerwise(
    model,
    images: torch.Tensor,
    target_coords: torch.Tensor,
    layer_indices: tuple = (2, 5, 8, 11),
) -> dict:
    """
    Run Grad-CAM at multiple depths to see how geographic abstraction builds up.
    Early layers → texture; later layers → high-level geographic structure.
    Returns a dict mapping layer index → [B, H, W] heatmap.
    """
    results = {}
    for idx in layer_indices:
        with gradcam_context(model, target_layer_idx=idx) as gc:
            results[idx] = gc.compute(images, target_coords)
    return results
