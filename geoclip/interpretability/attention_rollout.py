"""
Attention Rollout for our VisionTransformer.

Accumulates attention across all layers: R = (Â_L+I) ⊗ … ⊗ (Â_1+I).
The CLS row of R gives each patch's effective contribution to the final representation.

Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F


class AttentionRollout:
    """Accumulates multi-layer attention into a single spatial saliency map."""

    def __init__(self, model, head_fusion: str = "mean", discard_ratio: float = 0.0):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    @torch.no_grad()
    def compute(self, images: torch.Tensor) -> torch.Tensor:
        """Returns [B, H, W] rollout maps in [0, 1]."""
        self.model.eval()
        device = next(self.model.parameters()).device
        images = images.to(device)

        _, extras = self.model.image_encoder(images, output_attentions=True)
        attn_layers = extras["attentions"]

        B, _, S, _ = attn_layers[0].shape
        device = attn_layers[0].device

        # Start with identity — each token attends only to itself
        rollout = torch.eye(S, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        for attn in attn_layers:
            # Fuse attention heads
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=1)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=1).values
            else:
                raise ValueError(f"Unknown head_fusion: {self.head_fusion!r}. Use 'mean' or 'max'.")

            # Optionally zero out the lowest-attention tokens to reduce noise
            if self.discard_ratio > 0.0:
                flat = attn_fused.flatten(1)
                k = max(1, int(flat.shape[-1] * self.discard_ratio))
                threshold = flat.kthvalue(k, dim=-1).values
                attn_fused = attn_fused * (attn_fused > threshold[:, None, None])

            # Add identity for the residual connection, then re-normalize rows
            attn_fused = attn_fused + torch.eye(S, device=device)
            attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            rollout = attn_fused @ rollout  # [B, S, S]

        # Extract CLS row: how much each patch contributed end-to-end
        cls_attn  = rollout[:, 0, 1:]           # [B, num_patches]
        n_patches = cls_attn.shape[-1]
        grid_size = int(n_patches ** 0.5)
        cls_attn  = cls_attn.reshape(B, 1, grid_size, grid_size)

        H, W = images.shape[-2:]
        cls_attn = F.interpolate(cls_attn, size=(H, W), mode="bilinear", align_corners=False)
        cls_attn = cls_attn.squeeze(1)  # [B, H, W]

        a_min = cls_attn.flatten(1).min(1).values[:, None, None]
        a_max = cls_attn.flatten(1).max(1).values[:, None, None]
        return ((cls_attn - a_min) / (a_max - a_min + 1e-8)).cpu()


@contextmanager
def attention_rollout_context(model, head_fusion: str = "mean", discard_ratio: float = 0.0):
    """Context manager for API symmetry with gradcam_context (no hooks to clean up)."""
    yield AttentionRollout(model, head_fusion=head_fusion, discard_ratio=discard_ratio)


class PerHeadAttention:
    """
    Shows what each individual attention head attends to in one layer.

    Unlike rollout (which averages heads across all layers), this gives one
    map per head so you can spot specialization — e.g. heads that focus on
    sky vs. ground vs. text.
    """

    def __init__(self, model, layer_idx: int = -1):
        self.model = model
        self.layer_idx = layer_idx

    @torch.no_grad()
    def compute(self, images: torch.Tensor) -> torch.Tensor:
        """Returns [B, num_heads, H, W] maps, each normalized to [0, 1]."""
        self.model.eval()
        device = next(self.model.parameters()).device
        images = images.to(device)

        _, extras = self.model.image_encoder(images, output_attentions=True)
        attn = extras["attentions"][self.layer_idx]  # [B, num_heads, S, S]

        # CLS row → how much each patch was attended to, per head
        cls_attn = attn[:, :, 0, 1:]  # [B, num_heads, num_patches]

        B, num_heads, n_patches = cls_attn.shape
        grid_size = int(n_patches ** 0.5)
        H_img, W_img = images.shape[-2:]

        cls_attn = cls_attn.reshape(B * num_heads, 1, grid_size, grid_size)
        cls_attn = F.interpolate(cls_attn, size=(H_img, W_img), mode="bilinear", align_corners=False)
        cls_attn = cls_attn.reshape(B, num_heads, H_img, W_img)

        # Normalize each (sample, head) pair independently
        flat  = cls_attn.reshape(B, num_heads, -1)
        h_min = flat.min(dim=-1).values[:, :, None, None]
        h_max = flat.max(dim=-1).values[:, :, None, None]
        return ((cls_attn - h_min) / (h_max - h_min + 1e-8)).cpu()
