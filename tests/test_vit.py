"""
Unit tests for the custom Vision Transformer (vit.py).

All tests use a tiny configuration (64×64 image, 32-pixel patches,
2 layers, 64 hidden dims) so they run fast on CPU without any downloads.
"""
import torch
import pytest

from geoclip.models.vit import (
    PatchEmbedding,
    MultiHeadSelfAttention,
    MLP,
    TransformerBlock,
    VisionTransformer,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------

IMAGE_SIZE  = 64
PATCH_SIZE  = 32       # → 2×2 = 4 patches
HIDDEN_DIM  = 64
NUM_HEADS   = 4
NUM_LAYERS  = 2
MLP_DIM     = 128
BATCH       = 3
SEQ_LEN     = 1 + (IMAGE_SIZE // PATCH_SIZE) ** 2   # CLS + 4 patches = 5


@pytest.fixture(scope="module")
def tiny_vit():
    return VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
    ).eval()


# ---------------------------------------------------------------------------
# PatchEmbedding
# ---------------------------------------------------------------------------

class TestPatchEmbedding:
    def setup_method(self):
        self.pe = PatchEmbedding(IMAGE_SIZE, PATCH_SIZE, in_channels=3, hidden_dim=HIDDEN_DIM)

    def test_num_patches(self):
        expected = (IMAGE_SIZE // PATCH_SIZE) ** 2
        assert self.pe.num_patches == expected

    def test_output_shape(self):
        x = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = self.pe(x)
        assert out.shape == (BATCH, self.pe.num_patches, HIDDEN_DIM)

    def test_no_nan(self):
        x = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = self.pe(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# MultiHeadSelfAttention
# ---------------------------------------------------------------------------

class TestMHSA:
    def setup_method(self):
        self.mhsa = MultiHeadSelfAttention(HIDDEN_DIM, NUM_HEADS)

    def test_output_shape(self):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out, attn = self.mhsa(x, output_attentions=False)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)
        assert attn is None

    def test_attention_shape(self):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out, attn = self.mhsa(x, output_attentions=True)
        assert attn.shape == (BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)

    def test_attention_sums_to_one(self):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        _, attn = self.mhsa(x, output_attentions=True)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_no_nan(self):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out, _ = self.mhsa(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def setup_method(self):
        self.mlp = MLP(HIDDEN_DIM, MLP_DIM)

    def test_output_shape(self):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out = self.mlp(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_no_nan(self):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        assert not torch.isnan(self.mlp(x)).any()


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    def setup_method(self):
        self.block = TransformerBlock(HIDDEN_DIM, NUM_HEADS, MLP_DIM)

    def test_output_shape_no_attn(self):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out, attn = self.block(x, output_attentions=False)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)
        assert attn is None

    def test_output_shape_with_attn(self):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out, attn = self.block(x, output_attentions=True)
        assert attn.shape == (BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)

    def test_residual_not_zero(self):
        """Residual connection: output must not equal input (block has non-zero params)."""
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        out, _ = self.block(x)
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# VisionTransformer (end-to-end)
# ---------------------------------------------------------------------------

class TestVisionTransformer:
    def test_output_shape(self, tiny_vit):
        x = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = tiny_vit(x)
        assert out.shape == (BATCH, HIDDEN_DIM)

    def test_no_nan(self, tiny_vit):
        x = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert not torch.isnan(tiny_vit(x)).any()

    def test_output_attentions(self, tiny_vit):
        x = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
        cls_out, extras = tiny_vit(x, output_attentions=True)
        assert cls_out.shape == (BATCH, HIDDEN_DIM)
        assert "attentions" in extras
        assert len(extras["attentions"]) == NUM_LAYERS
        for attn in extras["attentions"]:
            assert attn.shape == (BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)

    def test_output_hidden_states(self, tiny_vit):
        x = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
        _, extras = tiny_vit(x, output_hidden_states=True)
        assert "hidden_states" in extras
        # NUM_LAYERS + 1 (initial embedding before first block)
        assert len(extras["hidden_states"]) == NUM_LAYERS + 1
        for hs in extras["hidden_states"]:
            assert hs.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)

    def test_deterministic_in_eval(self, tiny_vit):
        x = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
        out1 = tiny_vit(x)
        out2 = tiny_vit(x)
        assert torch.allclose(out1, out2)

    def test_cls_token_shape(self, tiny_vit):
        assert tiny_vit.cls_token.shape == (1, 1, HIDDEN_DIM)

    def test_pos_embed_shape(self, tiny_vit):
        n_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
        assert tiny_vit.pos_embed.shape == (1, n_patches + 1, HIDDEN_DIM)

    def test_batch_size_one(self, tiny_vit):
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = tiny_vit(x)
        assert out.shape == (1, HIDDEN_DIM)
