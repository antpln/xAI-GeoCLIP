"""
Vision Transformer (ViT) implemented from scratch in PyTorch.

Matches the architecture of the CLIP ViT variants used in GeoCLIP:
  ViT-B/32: patch=32, hidden=768, layers=12, heads=12, mlp=3072
  ViT-B/16: patch=16, hidden=768, layers=12, heads=12, mlp=3072
  ViT-L/14: patch=14, hidden=1024, layers=24, heads=16, mlp=4096

The forward pass natively returns per-layer attention weights and hidden
states, which are consumed by GradCAM and AttentionRollout without any
hooks or monkey-patching.
"""
import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architecture configs
# ---------------------------------------------------------------------------

VIT_CONFIGS: Dict[str, dict] = {
    "ViT-B/32": dict(patch_size=32, hidden_dim=768,  num_layers=12, num_heads=12, mlp_dim=3072),
    "ViT-B/16": dict(patch_size=16, hidden_dim=768,  num_layers=12, num_heads=12, mlp_dim=3072),
    "ViT-L/14": dict(patch_size=14, hidden_dim=1024, num_layers=24, num_heads=16, mlp_dim=4096),
}

# HuggingFace model IDs used only for weight loading — not for running inference
CLIP_HF_IDS: Dict[str, str] = {
    "ViT-B/32": "openai/clip-vit-base-patch32",
    "ViT-B/16": "openai/clip-vit-base-patch16",
    "ViT-L/14": "openai/clip-vit-large-patch14",
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Splits an image into non-overlapping patches and linearly projects each
    patch to the hidden dimension.

    Implemented as a strided convolution: kernel_size = stride = patch_size,
    which is equivalent to extracting patches and applying a shared linear layer.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_dim: int):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, num_patches, hidden_dim]
        """
        x = self.proj(x)           # [B, hidden_dim, H/p, W/p]
        x = x.flatten(2)           # [B, hidden_dim, num_patches]
        x = x.transpose(1, 2)      # [B, num_patches, hidden_dim]
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention.

    Explicit Q, K, V projections + output projection, matching the CLIP ViT
    implementation. Attention weights are returned when requested so that
    AttentionRollout can consume them without any side-channel hooks.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:                  [B, S, D]
            output_attentions:  If True, also return [B, heads, S, S].

        Returns:
            out:          [B, S, D]
            attn_weights: [B, heads, S, S] or None
        """
        B, S, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, S, H, Dh).transpose(1, 2)  # [B, H, S, Dh]

        q = split_heads(self.q_proj(x))  # [B, H, S, Dh]
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, S, S]
        attn = attn.softmax(dim=-1)

        out = (attn @ v)                                 # [B, H, S, Dh]
        out = out.transpose(1, 2).reshape(B, S, D)       # [B, S, D]
        out = self.out_proj(out)

        return out, (attn if output_attentions else None)


class MLP(nn.Module):
    """
    Position-wise two-layer feedforward network with GELU activation,
    matching the CLIP ViT MLP block.
    """

    def __init__(self, hidden_dim: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block:
        x → LayerNorm → MHSA → residual → LayerNorm → MLP → residual

    CLIP uses pre-norm (LayerNorm before sub-layers), not the post-norm
    of the original "Attention is All You Need" paper.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn  = MultiHeadSelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp   = MLP(hidden_dim, mlp_dim)

    def forward(
        self,
        x: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.norm1(x), output_attentions=output_attentions)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


# ---------------------------------------------------------------------------
# Full Vision Transformer
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image encoding.

    Token layout after patch embedding:
        [CLS, patch_1, patch_2, ..., patch_N]   (N = num_patches)
    The CLS token aggregates global image information and is used as the
    image representation after the final LayerNorm.

    Args:
        image_size:  Input image resolution (assumed square).
        patch_size:  Side length of each patch in pixels.
        in_channels: Number of image channels (3 for RGB).
        hidden_dim:  Token embedding dimension.
        num_layers:  Number of transformer blocks.
        num_heads:   Number of attention heads per block.
        mlp_dim:     Inner dimension of the MLP blocks.
    """

    def __init__(
        self,
        image_size:  int = 224,
        patch_size:  int = 16,
        in_channels: int = 3,
        hidden_dim:  int = 768,
        num_layers:  int = 12,
        num_heads:   int = 12,
        mlp_dim:     int = 3072,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, hidden_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))

        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.blocks   = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])
        self.post_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        output_attentions:    bool = False,
        output_hidden_states: bool = False,
    ):
        """
        Args:
            x:                    [B, C, H, W] float32 images.
            output_attentions:    Return per-layer attention weights.
            output_hidden_states: Return per-layer hidden states.

        Returns:
            cls_out: [B, hidden_dim] — post-norm CLS token embedding.
            extras (dict, only if requested):
                "attentions":    tuple of [B, heads, S, S] per layer.
                "hidden_states": tuple of [B, S, hidden_dim] per layer + initial.
        """
        B = x.shape[0]

        # 1. Patch embedding
        tokens = self.patch_embed(x)                             # [B, N, D]

        # 2. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)                  # [B, 1, D]
        tokens = torch.cat([cls, tokens], dim=1)                 # [B, N+1, D]

        # 3. Add positional embeddings + initial LayerNorm (pre-norm style)
        tokens = self.pre_norm(tokens + self.pos_embed)

        # 4. Transformer blocks
        all_attentions:    List[torch.Tensor] = []
        all_hidden_states: List[torch.Tensor] = [tokens] if output_hidden_states else []

        for block in self.blocks:
            tokens, attn_w = block(tokens, output_attentions=output_attentions)
            if output_attentions and attn_w is not None:
                all_attentions.append(attn_w)
            if output_hidden_states:
                all_hidden_states.append(tokens)

        # 5. Extract CLS token and apply final LayerNorm
        cls_out = self.post_norm(tokens[:, 0])                   # [B, D]

        if output_attentions or output_hidden_states:
            extras: Dict = {}
            if output_attentions:
                extras["attentions"]    = tuple(all_attentions)
            if output_hidden_states:
                extras["hidden_states"] = tuple(all_hidden_states)
            return cls_out, extras

        return cls_out


# ---------------------------------------------------------------------------
# CLIP weight loading
# ---------------------------------------------------------------------------

def load_clip_weights(vit: VisionTransformer, model_name: str) -> None:
    """
    Load pre-trained CLIP weights from HuggingFace into our custom ViT.

    Only the vision encoder weights are loaded; our projection head is
    randomly initialized and trained from scratch as part of GeoCLIP.

    The weight names in HuggingFace's CLIPVisionModel differ from ours,
    so we map them explicitly below.

    Args:
        vit:        Our VisionTransformer instance.
        model_name: One of "ViT-B/32", "ViT-B/16", "ViT-L/14".
    """
    from transformers import CLIPVisionModel as _HFVision

    hf_id = CLIP_HF_IDS[model_name]
    print(f"[ViT] Downloading CLIP weights from '{hf_id}' ...")
    hf_model = _HFVision.from_pretrained(hf_id)
    hf_sd = hf_model.state_dict()

    # Some transformers versions prefix keys with "vision_model.", others don't
    p = "vision_model." if any(k.startswith("vision_model.") for k in hf_sd) else ""

    our_sd = vit.state_dict()
    mapping: Dict[str, str] = {}

    mapping[f"{p}embeddings.patch_embedding.weight"] = "patch_embed.proj.weight"
    mapping[f"{p}embeddings.patch_embedding.bias"]   = "patch_embed.proj.bias"
    mapping[f"{p}pre_layrnorm.weight"]               = "pre_norm.weight"
    mapping[f"{p}pre_layrnorm.bias"]                 = "pre_norm.bias"
    mapping[f"{p}post_layernorm.weight"]             = "post_norm.weight"
    mapping[f"{p}post_layernorm.bias"]               = "post_norm.bias"

    for i in range(vit.num_layers):
        hf_pfx = f"{p}encoder.layers.{i}"
        our_pfx = f"blocks.{i}"
        for hf_k, our_k in [
            ("layer_norm1.weight",        "norm1.weight"),
            ("layer_norm1.bias",          "norm1.bias"),
            ("self_attn.q_proj.weight",   "attn.q_proj.weight"),
            ("self_attn.q_proj.bias",     "attn.q_proj.bias"),
            ("self_attn.k_proj.weight",   "attn.k_proj.weight"),
            ("self_attn.k_proj.bias",     "attn.k_proj.bias"),
            ("self_attn.v_proj.weight",   "attn.v_proj.weight"),
            ("self_attn.v_proj.bias",     "attn.v_proj.bias"),
            ("self_attn.out_proj.weight", "attn.out_proj.weight"),
            ("self_attn.out_proj.bias",   "attn.out_proj.bias"),
            ("layer_norm2.weight",        "norm2.weight"),
            ("layer_norm2.bias",          "norm2.bias"),
            ("mlp.fc1.weight",            "mlp.fc1.weight"),
            ("mlp.fc1.bias",              "mlp.fc1.bias"),
            ("mlp.fc2.weight",            "mlp.fc2.weight"),
            ("mlp.fc2.bias",              "mlp.fc2.bias"),
        ]:
            mapping[f"{hf_pfx}.{hf_k}"] = f"{our_pfx}.{our_k}"

    loaded, skipped = 0, 0
    for hf_key, our_key in mapping.items():
        if hf_key not in hf_sd:
            print(f"  [!] HF key not found: {hf_key}")
            skipped += 1
            continue
        if our_key not in our_sd:
            print(f"  [!] Our key not found: {our_key}")
            skipped += 1
            continue
        our_sd[our_key].copy_(hf_sd[hf_key])
        loaded += 1

    cls_hf = hf_sd[f"{p}embeddings.class_embedding"]
    our_sd["cls_token"].copy_(cls_hf.reshape(1, 1, -1))

    pos_hf = hf_sd[f"{p}embeddings.position_embedding.weight"]
    our_sd["pos_embed"].copy_(pos_hf.unsqueeze(0))

    vit.load_state_dict(our_sd)
    print(f"[ViT] Loaded {loaded + 2} weight tensors, skipped {skipped}.")

    del hf_model, hf_sd
