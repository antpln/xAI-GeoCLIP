"""
Unit tests for Integrated Gradients (interpretability/integrated_gradients.py).

Uses the shared `tiny_model` fixture (ViT-B/32, no pretrained weights).
"""
import torch
import pytest

from geoclip.interpretability.integrated_gradients import IntegratedGradients

B = 2
H_IMG, W_IMG = 224, 224


@pytest.fixture(scope="module")
def images():
    return torch.randn(B, 3, H_IMG, W_IMG)


@pytest.fixture(scope="module")
def coords():
    return torch.tensor([[48.85, 2.35], [-33.87, 151.21]])


# ---------------------------------------------------------------------------
# Shape and range
# ---------------------------------------------------------------------------

class TestIntegratedGradients:
    def test_output_shape(self, tiny_model, images, coords):
        ig = IntegratedGradients(tiny_model, n_steps=5)
        maps = ig.compute(images, coords)
        assert maps.shape == (B, H_IMG, W_IMG), f"Got {maps.shape}"

    def test_values_in_range(self, tiny_model, images, coords):
        ig = IntegratedGradients(tiny_model, n_steps=5)
        maps = ig.compute(images, coords)
        assert maps.min().item() >= 0.0
        assert maps.max().item() <= 1.0 + 1e-5

    def test_no_nan(self, tiny_model, images, coords):
        ig = IntegratedGradients(tiny_model, n_steps=5)
        maps = ig.compute(images, coords)
        assert not torch.isnan(maps).any()

    def test_returns_cpu_tensor(self, tiny_model, images, coords):
        ig = IntegratedGradients(tiny_model, n_steps=5)
        maps = ig.compute(images, coords)
        assert maps.device.type == "cpu"

    def test_baseline_image_gives_zero_map(self, tiny_model, coords):
        """When input == baseline, (input - baseline) = 0 → zero attributions → flat map."""
        baseline = torch.zeros(1, 3, H_IMG, W_IMG)
        ig = IntegratedGradients(tiny_model, n_steps=5, baseline_value=0.0)
        maps = ig.compute(baseline, coords[:1])
        # Map is normalized: a flat zero attribution gives a uniform map (all 0 after norm)
        assert maps.max().item() < 1e-3

    def test_different_coords_different_maps(self, tiny_model, images):
        """Different target GPS coordinates should yield different attribution maps."""
        coords_a = torch.tensor([[48.85, 2.35]])    # Paris
        coords_b = torch.tensor([[-33.87, 151.21]]) # Sydney
        ig = IntegratedGradients(tiny_model, n_steps=5)
        map_a = ig.compute(images[:1], coords_a)
        map_b = ig.compute(images[:1], coords_b)
        # Different GPS embeddings → different gradients → different maps (with high probability)
        assert not torch.allclose(map_a, map_b, atol=1e-3)

    # -----------------------------------------------------------------------
    # Coarse completeness check (approximate, not exact, due to n_steps=5)
    # -----------------------------------------------------------------------

    def test_completeness_direction(self, tiny_model):
        """
        The signed sum of IG attributions should approximate f(x) - f(baseline).

        With n_steps=5 the Riemann approximation is coarse, so we only check
        that the sign of the sum matches the sign of (f(x) - f(baseline)).
        """
        device = next(tiny_model.parameters()).device
        x       = torch.randn(1, 3, H_IMG, W_IMG)
        baseline = torch.zeros_like(x)
        coord   = torch.tensor([[48.85, 2.35]])

        with torch.no_grad():
            gps_emb  = tiny_model.encode_gps(coord.to(device))
            f_x      = (tiny_model.encode_image(x.to(device)) * gps_emb).sum().item()
            f_base   = (tiny_model.encode_image(baseline.to(device)) * gps_emb).sum().item()

        delta = f_x - f_base

        # Re-run IG with signed attributions (before abs + norm)
        accumulated = torch.zeros_like(x)
        for step in range(1, 6):
            alpha = step / 5
            interp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
            emb = tiny_model.encode_image(interp.to(device))
            score = (emb * gps_emb.detach()).sum()
            score.backward()
            accumulated += interp.grad.detach().cpu()

        signed_sum = ((x - baseline) * (accumulated / 5)).sum().item()

        if abs(delta) > 1e-4:
            assert (delta * signed_sum) > 0, (
                f"Sign mismatch: f(x)-f(base)={delta:.4f}, IG sum={signed_sum:.4f}"
            )
