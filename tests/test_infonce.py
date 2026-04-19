import torch
import torch.nn.functional as F
import pytest
from geoclip.losses.infonce import info_nce_loss


def _random_embeddings(B: int, D: int = 128):
    x = torch.randn(B, D)
    return F.normalize(x, dim=-1)


def test_loss_scalar():
    img = _random_embeddings(16)
    gps = _random_embeddings(16)
    scale = torch.tensor(14.3)
    loss = info_nce_loss(img, gps, scale)
    assert loss.ndim == 0, "Loss should be a scalar"


def test_loss_positive():
    img = _random_embeddings(16)
    gps = _random_embeddings(16)
    scale = torch.tensor(14.3)
    loss = info_nce_loss(img, gps, scale)
    assert loss.item() > 0, "Loss should be positive"


def test_perfect_alignment_lower_loss():
    """When image and GPS embeddings are identical, loss should be lower."""
    emb = _random_embeddings(16)
    scale = torch.tensor(14.3)
    loss_perfect = info_nce_loss(emb, emb, scale)
    loss_random = info_nce_loss(emb, _random_embeddings(16), scale)
    assert loss_perfect.item() < loss_random.item()


def test_higher_temperature_lower_loss():
    """Higher temperature (lower scale) -> smoother distribution -> higher entropy -> higher loss."""
    emb = _random_embeddings(16)
    loss_high_temp = info_nce_loss(emb, emb.clone(), torch.tensor(1.0))
    loss_low_temp = info_nce_loss(emb, emb.clone(), torch.tensor(100.0))
    assert loss_high_temp.item() > loss_low_temp.item()
