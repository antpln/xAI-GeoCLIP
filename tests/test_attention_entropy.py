"""
Unit tests for attention entropy regularization (losses/attention_entropy.py).
"""
import math

import torch
import pytest

from geoclip.losses.attention_entropy import attention_entropy, attention_entropy_loss


B      = 4
H      = 8    # num heads
S      = 10   # seq len (CLS + patches)


def _uniform_attn(b=B, h=H, s=S):
    """Uniform attention: every token attends equally to all tokens."""
    return torch.full((b, h, s, s), 1.0 / s)


def _sharp_attn(b=B, h=H, s=S):
    """All attention on token 0: minimum entropy."""
    t = torch.zeros(b, h, s, s)
    t[:, :, :, 0] = 1.0
    return t


# ---------------------------------------------------------------------------
# attention_entropy
# ---------------------------------------------------------------------------

class TestAttentionEntropy:
    def test_output_shape(self):
        attn = _uniform_attn()
        ent = attention_entropy(attn)
        assert ent.shape == (B,), f"Expected ({B},), got {ent.shape}"

    def test_uniform_equals_log_s(self):
        """H(uniform) = log(S) — maximum possible entropy."""
        attn = _uniform_attn()
        ent = attention_entropy(attn)
        expected = math.log(S)
        assert torch.allclose(ent, torch.full((B,), expected), atol=1e-4)

    def test_sharp_near_zero(self):
        """Sharp attention on one token → near-zero entropy."""
        attn = _sharp_attn()
        ent = attention_entropy(attn)
        assert (ent < 0.1).all(), f"Expected near-zero entropy, got {ent}"

    def test_uniform_greater_than_sharp(self):
        """Uniform attention must have strictly higher entropy than sharp."""
        ent_uniform = attention_entropy(_uniform_attn())
        ent_sharp   = attention_entropy(_sharp_attn())
        assert (ent_uniform > ent_sharp).all()

    def test_no_nan(self):
        """No NaN even for near-zero probabilities."""
        attn = _sharp_attn()
        ent = attention_entropy(attn)
        assert not torch.isnan(ent).any()


# ---------------------------------------------------------------------------
# attention_entropy_loss
# ---------------------------------------------------------------------------

class TestAttentionEntropyLoss:
    def test_scalar_output(self):
        attn_layers = tuple(_uniform_attn() for _ in range(4))
        loss = attention_entropy_loss(attn_layers, layer_idx=-1)
        assert loss.ndim == 0

    def test_uses_last_layer_by_default(self):
        """With layer_idx=-1, only the last layer's attention matters."""
        sharp  = tuple([_uniform_attn(), _uniform_attn(), _sharp_attn()])
        result = attention_entropy_loss(sharp, layer_idx=-1)
        direct = attention_entropy(_sharp_attn()).mean()
        assert torch.allclose(result, direct, atol=1e-5)

    def test_positive_for_uniform(self):
        """Uniform attention has positive entropy loss."""
        attn_layers = (_uniform_attn(),)
        loss = attention_entropy_loss(attn_layers, layer_idx=0)
        assert loss.item() > 0

    def test_lower_for_sharp(self):
        """Sharp attention loss is lower than uniform attention loss."""
        loss_uniform = attention_entropy_loss((_uniform_attn(),), layer_idx=0)
        loss_sharp   = attention_entropy_loss((_sharp_attn(),),   layer_idx=0)
        assert loss_sharp.item() < loss_uniform.item()
