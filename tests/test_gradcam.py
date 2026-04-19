"""
Smoke tests for Grad-CAM - verifies output shapes and hook lifecycle.
Uses a tiny ViT-B/32 config and random tensors (no dataset required).
"""
import torch
import pytest

pytest.importorskip("transformers", reason="transformers not installed")

from geoclip.models.geoclip import GeoCLIP
from geoclip.interpretability.gradcam import GradCAM, gradcam_context


@pytest.fixture(scope="module")
def model():
    m = GeoCLIP(
        clip_model_name="ViT-B/32",
        freeze_layers=11,
        rff_num_scales=2,
        rff_dim=16,
        mlp_hidden=64,
        embedding_dim=512,
    )
    m.eval()
    return m


def test_gradcam_output_shape(model):
    images = torch.randn(2, 3, 224, 224)
    coords = torch.tensor([[48.8566, 2.3522], [-33.8688, 151.2093]])
    with gradcam_context(model) as gc:
        heatmaps = gc.compute(images, coords)
    assert heatmaps.shape == (2, 224, 224), f"Got {heatmaps.shape}"


def test_gradcam_values_in_range(model):
    images = torch.randn(2, 3, 224, 224)
    coords = torch.tensor([[48.8566, 2.3522], [-33.8688, 151.2093]])
    with gradcam_context(model) as gc:
        heatmaps = gc.compute(images, coords)
    assert heatmaps.min().item() >= 0.0
    assert heatmaps.max().item() <= 1.0 + 1e-5


def test_hooks_cleaned_up(model):
    images = torch.randn(1, 3, 224, 224)
    coords = torch.tensor([[0.0, 0.0]])
    gc = GradCAM(model)
    _ = gc.compute(images, coords)
    gc.remove_hooks()
    # Subsequent forward passes must not be affected
    with torch.no_grad():
        _ = model.encode_image(images)
