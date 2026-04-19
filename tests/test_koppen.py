"""
Unit tests for the Köppen-Geiger climate classifier (utils/koppen.py).

The KoppenClassifier tests use a synthetic 0.5° raster (360×720 uint8)
injected via monkeypatching - no network access required.
"""
import numpy as np
import pytest

from geoclip.utils.koppen import (
    KoppenClassifier,
    KOPPEN_CLASSES,
    group_distance,
    classify_error,
)


# ---------------------------------------------------------------------------
# Synthetic raster fixture
# ---------------------------------------------------------------------------

def _make_mock_grid() -> np.ndarray:
    """
    Synthetic 360×720 uint8 Köppen grid (0.5° resolution).

    Assigns Köppen codes based on rough latitude bands:
      - Rows 0–59   (90°N – 60°N)  → ET (29, Polar)
      - Rows 60–119 (60°N – 30°N)  → Cfb (15, Temperate)
      - Rows 120–179 (30°N – 0°N)  → Csa (8, Temperate)
      - Rows 180–239 (0° – 30°S)   → Af (1, Tropical)
      - Rows 240–299 (30°S – 60°S) → BSk (7, Dry)
      - Rows 300–359 (60°S – 90°S) → EF (30, Polar)
    """
    grid = np.zeros((360, 720), dtype=np.uint8)
    grid[0:60, :]   = 29   # ET
    grid[60:120, :] = 15   # Cfb
    grid[120:180, :] = 8   # Csa
    grid[180:240, :] = 1   # Af
    grid[240:300, :] = 7   # BSk
    grid[300:360, :] = 30  # EF
    return grid


@pytest.fixture(scope="module")
def kc():
    """KoppenClassifier with a synthetic raster — no network access required."""
    from unittest.mock import patch
    mock_grid = _make_mock_grid()
    with patch("geoclip.utils.koppen.load_koppen_raster", return_value=mock_grid):
        classifier = KoppenClassifier()
    return classifier


# ---------------------------------------------------------------------------
# KoppenClassifier — lookup correctness
# ---------------------------------------------------------------------------

class TestKoppenClassifier:
    def test_arctic_lat_returns_polar(self, kc):
        """Latitude 80°N should be in polar zone (ET = code 29)."""
        cls = kc.get_class(80.0, 0.0)
        assert cls == "ET", f"Expected ET, got {cls}"

    def test_temperate_lat_returns_temperate(self, kc):
        """Latitude 45°N should be in temperate zone (Cfb = code 15)."""
        cls = kc.get_class(45.0, 0.0)
        assert cls == "Cfb", f"Expected Cfb, got {cls}"

    def test_tropical_lat_returns_tropical(self, kc):
        """Latitude 10°S should be in tropical zone (Af = code 1)."""
        cls = kc.get_class(-10.0, 0.0)
        assert cls == "Af", f"Expected Af, got {cls}"

    def test_group_extraction(self, kc):
        assert kc.get_group(80.0, 0.0)  == "E"
        assert kc.get_group(45.0, 0.0)  == "C"
        assert kc.get_group(-10.0, 0.0) == "A"
        assert kc.get_group(-45.0, 0.0) == "B"

    def test_pole_no_crash(self, kc):
        """Extreme latitudes (±90°) must not raise or produce NaN."""
        assert kc.get_class(90.0, 0.0)  != ""
        assert kc.get_class(-90.0, 0.0) != ""

    def test_dateline_no_crash(self, kc):
        assert kc.get_class(0.0, 180.0)  != ""
        assert kc.get_class(0.0, -180.0) != ""

    def test_batch_classify_shapes(self, kc):
        lats = np.array([80.0, 45.0, -10.0])
        lons = np.array([0.0, 0.0, 0.0])
        result = kc.classify_batch(lats, lons)
        assert len(result["codes"])   == 3
        assert len(result["classes"]) == 3
        assert len(result["groups"])  == 3

    def test_batch_classify_values(self, kc):
        lats = np.array([80.0, 45.0, -10.0])
        lons = np.array([0.0, 0.0, 0.0])
        result = kc.classify_batch(lats, lons)
        assert result["groups"] == ["E", "C", "A"]

    def test_scalar_vs_batch_consistency(self, kc):
        lats = np.array([45.0, -10.0])
        lons = np.array([0.0, 0.0])
        batch  = kc.classify_batch(lats, lons)
        scalar = [kc.get_class(float(la), float(lo)) for la, lo in zip(lats, lons)]
        assert batch["classes"] == scalar


# ---------------------------------------------------------------------------
# group_distance
# ---------------------------------------------------------------------------

class TestGroupDistance:
    def test_same_group_zero(self):
        for g in ["A", "B", "C", "D", "E"]:
            assert group_distance(g, g) == 0

    def test_adjacent_pairs(self):
        assert group_distance("A", "B") == 1
        assert group_distance("B", "C") == 1
        assert group_distance("C", "D") == 1
        assert group_distance("D", "E") == 1

    def test_symmetric(self):
        assert group_distance("A", "B") == group_distance("B", "A")
        assert group_distance("C", "E") == group_distance("E", "C")

    def test_far_pairs(self):
        assert group_distance("A", "E") == 4
        assert group_distance("A", "D") == 3

    def test_unknown_returns_minus_one(self):
        assert group_distance("?", "C") == -1
        assert group_distance("A", "?") == -1


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------

class TestClassifyError:
    def test_same_group_exact(self):
        assert classify_error("C", "C") == "exact"
        assert classify_error("A", "A") == "exact"

    def test_adjacent_groups(self):
        assert classify_error("C", "D") == "adjacent"
        assert classify_error("B", "C") == "adjacent"

    def test_distant_groups(self):
        assert classify_error("A", "E") == "distant"
        assert classify_error("A", "D") == "distant"

    def test_ocean(self):
        assert classify_error("?", "C") == "ocean"
        assert classify_error("A", "?") == "ocean"
        assert classify_error("?", "?") == "ocean"
