"""
Unit tests for OcclusionSensitivity (interpretability/occlusion.py).
"""
import torch
import pytest

from geoclip.interpretability.occlusion import OcclusionSensitivity

B = 2
H_IMG, W_IMG = 224, 224
PATCH_SIZE = 32    # ViT-B/32


@pytest.fixture(scope="module")
def images():
    return torch.randn(B, 3, H_IMG, W_IMG)


# ---------------------------------------------------------------------------
# Embedding mode (no gallery)
# ---------------------------------------------------------------------------

class TestOcclusionEmbeddingMode:
    def test_output_shape(self, tiny_model, images):
        occ = OcclusionSensitivity(tiny_model, patch_size=PATCH_SIZE)
        maps = occ.compute(images)
        assert maps.shape == (B, H_IMG, W_IMG), f"Got {maps.shape}"

    def test_values_in_range(self, tiny_model, images):
        occ = OcclusionSensitivity(tiny_model, patch_size=PATCH_SIZE)
        maps = occ.compute(images)
        assert maps.min().item() >= 0.0
        assert maps.max().item() <= 1.0 + 1e-5

    def test_no_nan(self, tiny_model, images):
        occ = OcclusionSensitivity(tiny_model, patch_size=PATCH_SIZE)
        maps = occ.compute(images)
        assert not torch.isnan(maps).any()

    def test_returns_cpu(self, tiny_model, images):
        occ = OcclusionSensitivity(tiny_model, patch_size=PATCH_SIZE)
        maps = occ.compute(images)
        assert maps.device.type == "cpu"

    def test_uniform_image_gives_flat_map(self, tiny_model):
        """A constant image loses no information from any patch occlusion → uniform scores."""
        flat_images = torch.zeros(1, 3, H_IMG, W_IMG)
        occ = OcclusionSensitivity(tiny_model, patch_size=PATCH_SIZE, fill_value=0.0)
        maps = occ.compute(flat_images)
        # All occlusions change the embedding by the same amount → normalized to all-zero
        assert maps.max().item() < 1e-3, f"Expected flat map, got max={maps.max():.4f}"

    def test_batch_size_one(self, tiny_model):
        x = torch.randn(1, 3, H_IMG, W_IMG)
        occ = OcclusionSensitivity(tiny_model, patch_size=PATCH_SIZE)
        maps = occ.compute(x)
        assert maps.shape == (1, H_IMG, W_IMG)


# ---------------------------------------------------------------------------
# GPS mode (with gallery)
# ---------------------------------------------------------------------------

class TestOcclusionGpsMode:
    def _make_gallery(self, G: int = 20, D: int = 512):
        import torch.nn.functional as F
        coords = torch.rand(G, 2) * 180 - 90
        embs   = F.normalize(torch.randn(G, D), dim=-1)
        return coords, embs

    def test_output_shape_gps_mode(self, tiny_model, images):
        gallery_coords, gallery_embs = self._make_gallery()
        occ = OcclusionSensitivity(tiny_model, patch_size=PATCH_SIZE)
        maps = occ.compute(images, gallery_coords=gallery_coords, gallery_embs=gallery_embs)
        assert maps.shape == (B, H_IMG, W_IMG)

    def test_values_in_range_gps_mode(self, tiny_model, images):
        gallery_coords, gallery_embs = self._make_gallery()
        occ = OcclusionSensitivity(tiny_model, patch_size=PATCH_SIZE)
        maps = occ.compute(images, gallery_coords=gallery_coords, gallery_embs=gallery_embs)
        assert maps.min().item() >= 0.0
        assert maps.max().item() <= 1.0 + 1e-5

    def test_no_nan_gps_mode(self, tiny_model, images):
        gallery_coords, gallery_embs = self._make_gallery()
        occ = OcclusionSensitivity(tiny_model, patch_size=PATCH_SIZE)
        maps = occ.compute(images, gallery_coords=gallery_coords, gallery_embs=gallery_embs)
        assert not torch.isnan(maps).any()
