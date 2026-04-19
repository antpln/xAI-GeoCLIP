"""
Shared pytest for the GeoCLIP test suite.
"""
from unittest.mock import patch

import pytest

from geoclip.models.geoclip import GeoCLIP


@pytest.fixture(scope="session")
def tiny_model():
    """
    GeoCLIP with ViT-B/32 + tiny GPS encoder, randomly initialised.

    ``load_clip_weights`` is patched to a no-op so the fixture works
    offline and in CI without network access.
    """
    with patch("geoclip.models.image_encoder.load_clip_weights", return_value=None):
        model = GeoCLIP(
            clip_model_name="ViT-B/32",
            freeze_layers=0,
            rff_num_scales=2,
            rff_dim=16,
            mlp_hidden=64,
            embedding_dim=512,
        )
    model.eval()
    return model
