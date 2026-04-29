"""
Köppen-Geiger climate classification lookup.

Downloads the Beck et al. (2018) 0.5° present-climate raster on first use
and caches it locally.  Provides fast (lat, lon) → class lookups with no
extra runtime dependencies beyond numpy and urllib.
"""
from __future__ import annotations

import csv
import os
import urllib.request
from pathlib import Path
from typing import Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

# All 30 standard Köppen-Geiger classes in numeric order (1-indexed in raster)
KOPPEN_CLASSES: list[str] = [
    "",       # 0 = no data / ocean
    "Af",  "Am",  "Aw",                          # 1-3   Tropical
    "BWh", "BWk", "BSh", "BSk",                  # 4-7   Dry
    "Csa", "Csb", "Csc",                          # 8-10  Temperate / dry summer
    "Cwa", "Cwb", "Cwc",                          # 11-13 Temperate / dry winter
    "Cfa", "Cfb", "Cfc",                          # 14-16 Temperate / no dry season
    "Dsa", "Dsb", "Dsc", "Dsd",                  # 17-20 Continental / dry summer
    "Dwa", "Dwb", "Dwc", "Dwd",                  # 21-24 Continental / dry winter
    "Dfa", "Dfb", "Dfc", "Dfd",                  # 25-28 Continental / no dry season
    "ET",  "EF",                                  # 29-30 Polar
]

KOPPEN_GROUPS: dict[str, str] = {
    "A": "Tropical",
    "B": "Dry",
    "C": "Temperate",
    "D": "Continental",
    "E": "Polar",
}

# Colour palette for the five major groups (matplotlib-compatible hex)
GROUP_COLORS: dict[str, str] = {
    "A": "#0000FF",   # blue
    "B": "#FF0000",   # red
    "C": "#00AA00",   # green
    "D": "#00FFFF",   # cyan
    "E": "#AAAAAA",   # grey
    "?": "#FFFFFF",   # unknown / ocean
}

# Fine-grained colours matching the standard Beck et al. map (30 classes)
# Index matches KOPPEN_CLASSES (1-based; index 0 = no-data)
CLASS_COLORS: list[str] = [
    "#FFFFFF",  # 0  no data
    "#0000FF", "#0078FF", "#4699FF",                # 1-3  Af Am Aw
    "#FF0000", "#FF9696", "#F5A500", "#FFDC64",     # 4-7  BWh BWk BSh BSk
    "#FFFF00", "#C8C800", "#969600",                # 8-10 Csa Csb Csc
    "#96FF96", "#64C864", "#329632",                # 11-13 Cwa Cwb Cwc
    "#C8FF50", "#64FF50", "#32C800",                # 14-16 Cfa Cfb Cfc
    "#FF00FF", "#C800C8", "#960096", "#640064",     # 17-20 Dsa Dsb Dsc Dsd
    "#AB82FF", "#9B30FF", "#7B00D4", "#5500A0",     # 21-24 Dwa Dwb Dwc Dwd
    "#00FFFF", "#37C8FF", "#007D7D", "#00465F",     # 25-28 Dfa Dfb Dfc Dfd
    "#B2B2B2", "#666666",                           # 29-30 ET EF
]

# ---------------------------------------------------------------------------
# Raster download + caching
# ---------------------------------------------------------------------------

# We use the 0.5-degree resolution grid from Beck et al., encoded as a uint8
# numpy array (rows = latitudes 90→-90, cols = longitudes -180→180).
# The file is ~0.5 MB and is served directly from the authors' supplement.
_DEFAULT_CACHE = Path.home() / ".cache" / "koppen_0p5.npy"

# Candidate sources tried in order
_NPY_URLS = [
    "https://github.com/hylken/koppen-geiger/raw/main/koppen_0p5.npy",
]

# Beck et al. (2018) 0.5° GeoTIFF via Figshare — converted on the fly with tifffile
_TIFF_URL = "https://figshare.com/ndownloader/files/12407516"


def _download_npy(cache_path: Path) -> np.ndarray:
    """Try candidate URLs in order, falling back to the Figshare TIFF if needed."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(cache_path) + ".tmp"

    # 1. Try pre-built .npy files
    for url in _NPY_URLS:
        try:
            print(f"[Koppen] Trying {url} …")
            urllib.request.urlretrieve(url, tmp)
            arr = np.load(tmp)
            np.save(cache_path, arr)
            os.remove(tmp)
            print(f"[Koppen] Cached to {cache_path}  shape={arr.shape}")
            return arr
        except Exception as e:
            print(f"[Koppen] Failed ({e}), trying next source …")
            if os.path.exists(tmp):
                os.remove(tmp)

    # 2. Fallback: download Beck et al. GeoTIFF and convert with tifffile
    try:
        import tifffile
    except ImportError:
        raise RuntimeError(
            "All .npy sources failed and tifffile is not installed.\n"
            "Install it with:  pip install tifffile\n"
            "Or manually save a 0.5° uint8 numpy grid (360×720) to:\n"
            f"  {cache_path}"
        )

    try:
        print(f"[Koppen] Downloading Beck et al. GeoTIFF from Figshare …")
        urllib.request.urlretrieve(_TIFF_URL, tmp)
        arr = tifffile.imread(tmp).astype(np.uint8)
        if arr.ndim == 3:
            arr = arr[0]  # some exports wrap in a band dimension
        np.save(cache_path, arr)
        os.remove(tmp)
        print(f"[Koppen] Cached to {cache_path}  shape={arr.shape}")
        return arr
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"All download attempts failed: {e}\n"
            "Manually save a 0.5° uint8 numpy grid (360 rows × 720 cols) to:\n"
            f"  {cache_path}"
        ) from e


def load_koppen_raster(cache_path: Union[str, Path, None] = None) -> np.ndarray:
    """
    Load (or download) the Köppen-Geiger 0.5° raster.

    Returns:
        uint8 array of shape [360, 720] — rows: 90°N→90°S, cols: 180°W→180°E.
        Values are 1-indexed class codes matching KOPPEN_CLASSES.
        0 = ocean / no data.
    """
    path = Path(cache_path) if cache_path else _DEFAULT_CACHE
    if path.exists():
        return np.load(path)
    return _download_npy(path)


# ---------------------------------------------------------------------------
# KoppenClassifier
# ---------------------------------------------------------------------------

class KoppenClassifier:
    """
    Fast Köppen-Geiger climate class lookup for arbitrary (lat, lon) pairs.

    Loads the 0.5° raster once and answers queries with array indexing.

    Args:
        cache_path: Where to cache the downloaded raster.
                    Defaults to ~/.cache/koppen_0p5.npy.

    Example::

        kc = KoppenClassifier()
        kc.get_class(48.85, 2.35)   # → "Cfb"
        kc.get_group(48.85, 2.35)   # → "C"
    """

    def __init__(self, cache_path: Union[str, Path, None] = None, require_raster: bool = True):
        if require_raster:
            self._grid = load_koppen_raster(cache_path)   # [360, 720] uint8
            self._n_rows, self._n_cols = self._grid.shape
        else:
            self._grid = None
            self._n_rows = self._n_cols = 0
        self._kdtree = None
        self._kdtree_codes: Optional[np.ndarray] = None

    @classmethod
    def from_csv(cls, csv_path: Union[str, Path], max_points: int = 200_000) -> "KoppenClassifier":
        """
        Build a KoppenClassifier backed by a KD-tree of reference points from a CSV.

        Useful when the raster cannot be downloaded: any (lat, lon) is classified
        by finding the nearest reference point in the CSV and returning its climate code.
        Climate zones are spatially smooth so this is accurate to well within 0.5°.

        Args:
            csv_path:   Path to a CSV with 'latitude', 'longitude', 'climate' columns.
            max_points: Subsample to this many points to keep the tree small.
        """
        from scipy.spatial import cKDTree

        print(f"[Koppen] Building KD-tree from {csv_path} (max {max_points:,} pts) …")
        lats, lons, codes = [], [], []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    c = int(float(row["climate"]))
                except (KeyError, ValueError):
                    continue
                if c == 0:
                    continue
                lats.append(float(row["latitude"]))
                lons.append(float(row["longitude"]))
                codes.append(c)

        lats  = np.array(lats,  dtype=np.float32)
        lons  = np.array(lons,  dtype=np.float32)
        codes = np.array(codes, dtype=np.uint8)

        if len(lats) > max_points:
            idx   = np.random.choice(len(lats), max_points, replace=False)
            lats, lons, codes = lats[idx], lons[idx], codes[idx]

        obj = cls(require_raster=False)
        obj._kdtree       = cKDTree(np.stack([lats, lons], axis=1))
        obj._kdtree_codes = codes
        print(f"[Koppen] KD-tree ready: {len(codes):,} reference points.")
        return obj

    # ------------------------------------------------------------------
    # Internal coordinate → grid-index conversion
    # ------------------------------------------------------------------

    def _latlon_to_idx(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
    ) -> tuple:
        """Convert (lat, lon) degrees to (row, col) in the 0.5° grid."""
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        # Row 0 = 90°N; last row = 90°S (step = −0.5°)
        row = np.clip(((90.0 - lat) / 0.5).astype(int), 0, self._n_rows - 1)
        # Col 0 = 180°W; last col = 180°E (step = +0.5°)
        col = np.clip(((lon + 180.0) / 0.5).astype(int), 0, self._n_cols - 1)
        return row, col

    # ------------------------------------------------------------------
    # Public API — scalar or batch
    # ------------------------------------------------------------------

    def get_code(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
    ) -> Union[int, np.ndarray]:
        """Return raw integer class code(s) (1–30; 0 = no data)."""
        row, col = self._latlon_to_idx(lat, lon)
        return self._grid[row, col]

    def get_class(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
    ) -> Union[str, list[str]]:
        """
        Return Köppen class string(s), e.g. "Cfb".

        Scalar input → scalar string.
        Array input  → list of strings.
        """
        codes = self.get_code(lat, lon)
        scalar = codes.ndim == 0
        codes = np.atleast_1d(codes)
        result = [
            KOPPEN_CLASSES[c] if 0 < c < len(KOPPEN_CLASSES) else "?"
            for c in codes
        ]
        return result[0] if scalar else result

    def get_group(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
    ) -> Union[str, list[str]]:
        """
        Return major climate group letter(s): A / B / C / D / E.

        Ocean / no-data pixels return "?".
        """
        classes = self.get_class(lat, lon)
        if isinstance(classes, str):
            return classes[0] if classes and classes != "?" else "?"
        return [c[0] if c and c != "?" else "?" for c in classes]

    def classify_from_codes(self, codes) -> dict:
        """
        Convert precomputed integer climate codes (1-30) to classes/groups.
        Skips the raster lookup — use this when the CSV already has a climate column.
        """
        codes = np.asarray(codes, dtype=int)
        classes = [
            KOPPEN_CLASSES[c] if 0 < c < len(KOPPEN_CLASSES) else "?"
            for c in codes
        ]
        groups = [c[0] if c and c != "?" else "?" for c in classes]
        return {"codes": codes, "classes": classes, "groups": groups}

    def classify_batch(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> dict:
        """
        Classify a batch of coordinates via raster lookup or KD-tree fallback.
        """
        if self._kdtree is not None:
            pts   = np.stack([np.asarray(lats), np.asarray(lons)], axis=1)
            _, ii = self._kdtree.query(pts, workers=-1)
            return self.classify_from_codes(self._kdtree_codes[ii])
        codes   = self.get_code(lats, lons)
        classes = self.get_class(lats, lons)
        groups  = [c[0] if c and c != "?" else "?" for c in classes]
        return {"codes": codes, "classes": classes, "groups": groups}


# ---------------------------------------------------------------------------
# Error coherence helpers
# ---------------------------------------------------------------------------

# Hierarchical distance between major groups (symmetric)
# 0 = same group, 1 = adjacent climate, 2 = distant climate
_GROUP_ADJACENCY: dict[frozenset, int] = {
    # Tropical ↔ Dry: share subtropical boundary
    frozenset({"A", "B"}): 1,
    # Dry ↔ Temperate: Mediterranean transition
    frozenset({"B", "C"}): 1,
    # Temperate ↔ Continental: cold-winter gradient
    frozenset({"C", "D"}): 1,
    # Continental ↔ Polar: high-latitude boundary
    frozenset({"D", "E"}): 1,
    # Tropical ↔ Temperate: two steps
    frozenset({"A", "C"}): 2,
    # Dry ↔ Continental: two steps
    frozenset({"B", "D"}): 2,
    # Temperate ↔ Polar: two steps
    frozenset({"C", "E"}): 2,
    # Three+ steps
    frozenset({"A", "D"}): 3,
    frozenset({"B", "E"}): 3,
    frozenset({"A", "E"}): 4,
}


def group_distance(g1: str, g2: str) -> int:
    """
    Hierarchical distance between two major Köppen groups.

    Returns:
        0 — same group
        1 — adjacent groups (climate gradient boundary)
        2 — two steps apart
        3/4 — distant
        -1 — unknown (ocean / no-data)
    """
    if "?" in (g1, g2):
        return -1
    if g1 == g2:
        return 0
    return _GROUP_ADJACENCY.get(frozenset({g1, g2}), 4)


def classify_error(true_group: str, pred_group: str) -> str:
    """
    Classify a geolocation error by climate coherence.

    Returns:
        "exact"     — same major climate group (model saw real geographic signal)
        "adjacent"  — neighbouring climate groups (plausible geographic confusion)
        "distant"   — different hemisphere or climate regime
        "ocean"     — one or both points on ocean / no-data
    """
    d = group_distance(true_group, pred_group)
    if d == -1:
        return "ocean"
    if d == 0:
        return "exact"
    if d == 1:
        return "adjacent"
    return "distant"
