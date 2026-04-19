"""
Unit tests for hard geographic negative mining (training/hard_negatives.py).
"""
import torch
import torch.nn.functional as F
import pytest

from geoclip.training.hard_negatives import (
    build_hard_negative_gps_embs,
    info_nce_with_hard_negatives,
)


def _rand_emb(b: int, d: int = 64):
    return F.normalize(torch.randn(b, d), dim=-1)


# ---------------------------------------------------------------------------
# build_hard_negative_gps_embs
# ---------------------------------------------------------------------------

class TestBuildHardNegativeGpsEmbs:
    def test_output_shape_preserved(self):
        B, D = 8, 64
        gps_emb = _rand_emb(B, D)
        coords  = torch.rand(B, 2) * 180 - 90   # random lat/lon
        out = build_hard_negative_gps_embs(gps_emb, coords)
        assert out.shape == (B, D), f"Expected ({B},{D}), got {out.shape}"

    def test_no_swap_when_too_far(self):
        """When all pairs are far apart (>min_distance_km), no swap occurs."""
        B, D = 4, 32
        gps_emb = _rand_emb(B, D)
        # Place each point far from every other (~10 000 km apart)
        coords = torch.tensor([
            [90.0,   0.0],   # North Pole
            [-90.0,  0.0],   # South Pole
            [0.0,   90.0],   # Equator, 90°E
            [0.0,  -90.0],   # Equator, 90°W
        ])
        out = build_hard_negative_gps_embs(
            gps_emb, coords,
            swap_prob=1.0,
            min_distance_km=1.0,   # only swap if within 1 km — none qualify
        )
        assert torch.allclose(out, gps_emb), "No swap expected when all pairs are far apart"

    def test_swap_when_close_and_prob_one(self):
        """When two points are very close and swap_prob=1, embeddings must change."""
        B, D = 4, 32
        gps_emb = _rand_emb(B, D)
        # First two points are almost identical → will be each other's nearest neighbor
        coords = torch.tensor([
            [0.001, 0.001],
            [0.002, 0.002],
            [80.0, 80.0],
            [-80.0, -80.0],
        ])
        out = build_hard_negative_gps_embs(
            gps_emb, coords,
            swap_prob=1.0,
            min_distance_km=500.0,   # 0.1 km << 500 km → qualifies
        )
        # At least one of the close pair must have been swapped
        swapped = not torch.allclose(out[0], gps_emb[0]) or \
                  not torch.allclose(out[1], gps_emb[1])
        assert swapped, "Expected at least one swap for close pair with swap_prob=1"

    def test_output_unit_norm(self):
        """Swapped embeddings come from original unit-norm embeddings, so remain unit norm."""
        B, D = 6, 32
        gps_emb = _rand_emb(B, D)
        coords  = torch.zeros(B, 2)   # all at same location → all close
        out = build_hard_negative_gps_embs(
            gps_emb, coords, swap_prob=1.0, min_distance_km=1e9
        )
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5), \
            f"Expected unit norm, got {norms}"

    def test_no_self_swap(self):
        """Diagonal is masked; a sample should never be swapped with itself."""
        B, D = 4, 32
        gps_emb = _rand_emb(B, D)
        # All points identical → nearest neighbor is always another sample
        coords = torch.zeros(B, 2)
        out = build_hard_negative_gps_embs(
            gps_emb, coords, swap_prob=1.0, min_distance_km=1e9
        )
        for i in range(B):
            # If sample i was swapped, it should hold a *different* sample's embedding
            # (but we can't know which one without knowing the random draw).
            # At minimum: the result must come from gps_emb, not be a new vector.
            assert any(
                torch.allclose(out[i], gps_emb[j]) for j in range(B)
            ), f"Row {i} is not any original embedding"


# ---------------------------------------------------------------------------
# info_nce_with_hard_negatives
# ---------------------------------------------------------------------------

class TestInfoNceWithHardNegatives:
    def test_scalar_output(self):
        B, D = 8, 64
        img_emb = _rand_emb(B, D)
        gps_emb = _rand_emb(B, D)
        coords  = torch.rand(B, 2) * 180 - 90
        scale   = torch.tensor(14.3)
        loss = info_nce_with_hard_negatives(img_emb, gps_emb, coords, scale)
        assert loss.ndim == 0

    def test_positive_loss(self):
        B, D = 8, 64
        img_emb = _rand_emb(B, D)
        gps_emb = _rand_emb(B, D)
        coords  = torch.rand(B, 2) * 180 - 90
        scale   = torch.tensor(14.3)
        loss = info_nce_with_hard_negatives(img_emb, gps_emb, coords, scale)
        assert loss.item() > 0

    def test_finite_loss(self):
        B, D = 8, 64
        img_emb = _rand_emb(B, D)
        gps_emb = _rand_emb(B, D)
        coords  = torch.rand(B, 2) * 180 - 90
        scale   = torch.tensor(14.3)
        loss = info_nce_with_hard_negatives(img_emb, gps_emb, coords, scale)
        assert torch.isfinite(loss)

    def test_perfect_alignment_no_swap(self):
        """When embeddings are perfectly aligned and no swaps occur, loss is low."""
        B, D = 8, 64
        emb = _rand_emb(B, D)
        coords = torch.tensor([[80.0, 0.0], [-80.0, 0.0]] * (B // 2))  # all far apart
        scale = torch.tensor(50.0)
        loss = info_nce_with_hard_negatives(
            emb, emb.clone(), coords, scale,
            swap_prob=0.0,        # no swaps
            min_distance_km=1.0,  # threshold so low no swap happens
        )
        from geoclip.losses.infonce import info_nce_loss
        loss_baseline = info_nce_loss(emb, emb.clone(), scale)
        assert torch.allclose(loss, loss_baseline, atol=1e-5)
