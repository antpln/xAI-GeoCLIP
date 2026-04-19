"""
Unit tests for Attention Rollout and PerHeadAttention
(interpretability/attention_rollout.py).
Images are 224×224 as required by the ViT-B/32 patch embedding.
"""
import torch
import pytest

from geoclip.interpretability.attention_rollout import (
    AttentionRollout,
    attention_rollout_context,
    PerHeadAttention,
)

B = 2
H_IMG, W_IMG = 224, 224
NUM_HEADS_VIT_B32 = 12


@pytest.fixture(scope="module")
def images():
    return torch.randn(B, 3, H_IMG, W_IMG)


# ---------------------------------------------------------------------------
# AttentionRollout
# ---------------------------------------------------------------------------

class TestAttentionRollout:
    def test_output_shape(self, tiny_model, images):
        with attention_rollout_context(tiny_model) as ar:
            maps = ar.compute(images)
        assert maps.shape == (B, H_IMG, W_IMG), f"Got {maps.shape}"

    def test_values_in_range(self, tiny_model, images):
        with attention_rollout_context(tiny_model) as ar:
            maps = ar.compute(images)
        assert maps.min().item() >= 0.0
        assert maps.max().item() <= 1.0 + 1e-5

    def test_no_nan(self, tiny_model, images):
        with attention_rollout_context(tiny_model) as ar:
            maps = ar.compute(images)
        assert not torch.isnan(maps).any()

    def test_head_fusion_max(self, tiny_model, images):
        """head_fusion='max' should also produce valid maps."""
        with attention_rollout_context(tiny_model, head_fusion="max") as ar:
            maps = ar.compute(images)
        assert maps.shape == (B, H_IMG, W_IMG)
        assert maps.min().item() >= 0.0

    def test_invalid_head_fusion_raises(self, tiny_model, images):
        with pytest.raises(ValueError, match="head_fusion"):
            with attention_rollout_context(tiny_model, head_fusion="bad") as ar:
                ar.compute(images)

    def test_discard_ratio(self, tiny_model, images):
        """Non-zero discard_ratio should still produce valid maps."""
        with attention_rollout_context(tiny_model, discard_ratio=0.2) as ar:
            maps = ar.compute(images)
        assert maps.shape == (B, H_IMG, W_IMG)
        assert maps.min().item() >= 0.0

    def test_context_manager_no_side_effects(self, tiny_model, images):
        """After the context exits the model forward pass is unaffected."""
        with attention_rollout_context(tiny_model) as ar:
            _ = ar.compute(images)
        with torch.no_grad():
            out = tiny_model.encode_image(images)
        assert out.shape == (B, 512)


# ---------------------------------------------------------------------------
# PerHeadAttention
# ---------------------------------------------------------------------------

class TestPerHeadAttention:
    def test_output_shape(self, tiny_model, images):
        pha = PerHeadAttention(tiny_model, layer_idx=-1)
        maps = pha.compute(images)
        assert maps.shape == (B, NUM_HEADS_VIT_B32, H_IMG, W_IMG), f"Got {maps.shape}"

    def test_values_in_range(self, tiny_model, images):
        pha = PerHeadAttention(tiny_model, layer_idx=-1)
        maps = pha.compute(images)
        assert maps.min().item() >= 0.0
        assert maps.max().item() <= 1.0 + 1e-5

    def test_no_nan(self, tiny_model, images):
        pha = PerHeadAttention(tiny_model, layer_idx=-1)
        maps = pha.compute(images)
        assert not torch.isnan(maps).any()

    def test_different_layers_differ(self, tiny_model, images):
        """Attention patterns at the first and last layer should not be identical."""
        maps_first = PerHeadAttention(tiny_model, layer_idx=0).compute(images)
        maps_last  = PerHeadAttention(tiny_model, layer_idx=-1).compute(images)
        assert not torch.allclose(maps_first, maps_last)

    def test_per_sample_normalized(self, tiny_model, images):
        """Each (sample, head) map must independently span the full [0,1] range."""
        pha  = PerHeadAttention(tiny_model, layer_idx=-1)
        maps = pha.compute(images)   # [B, heads, H, W]
        flat = maps.reshape(B, NUM_HEADS_VIT_B32, -1)
        maxs = flat.max(dim=-1).values
        mins = flat.min(dim=-1).values
        assert (maxs > 0.5).all(), "Each head map should have a max > 0.5 after normalization"
        assert (mins < 0.1).all(), "Each head map should have a min < 0.1 after normalization"
