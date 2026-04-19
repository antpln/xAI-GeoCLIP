import torch
import pytest
from geoclip.models.gps_encoder import LocationEncoder


def test_output_shape():
    enc = LocationEncoder(num_scales=4, rff_dim=32, mlp_hidden=64, embedding_dim=128)
    coords = torch.randn(8, 2)
    out = enc(coords)
    assert out.shape == (8, 128), f"Expected (8, 128), got {out.shape}"


def test_output_normalized():
    enc = LocationEncoder(num_scales=4, rff_dim=32, mlp_hidden=64, embedding_dim=128)
    coords = torch.tensor([[48.8566, 2.3522], [-33.8688, 151.2093]])  # Paris, Sydney
    out = enc(coords)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), f"Not unit norm: {norms}"


def test_rff_buffers_not_trainable():
    enc = LocationEncoder(num_scales=4, rff_dim=32, mlp_hidden=64, embedding_dim=128)
    for k in range(4):
        buf = getattr(enc, f"B_{k}")
        # Buffers should not have requires_grad
        assert not buf.requires_grad, f"B_{k} should not require grad"


def test_forward_deterministic():
    enc = LocationEncoder(num_scales=4, rff_dim=32, mlp_hidden=64, embedding_dim=128)
    coords = torch.tensor([[48.8566, 2.3522]])
    out1 = enc(coords)
    out2 = enc(coords)
    assert torch.allclose(out1, out2), "Forward is not deterministic"


def test_extreme_coordinates():
    enc = LocationEncoder(num_scales=4, rff_dim=32, mlp_hidden=64, embedding_dim=128)
    coords = torch.tensor([
        [90.0, 180.0],    # North Pole, date line
        [-90.0, -180.0],  # South Pole
        [0.0, 0.0],       # Equator/Greenwich
    ])
    out = enc(coords)
    assert not torch.isnan(out).any(), "NaN in output for extreme coordinates"
