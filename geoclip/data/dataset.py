from __future__ import annotations

import csv
import glob
import io
import os
import traceback
import zipfile
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class OSV5MDataset(Dataset):
    """
    PyTorch Dataset wrapping the OpenStreetView-5M HuggingFace dataset.

    The HuggingFace dataset returns PIL Image objects directly via the
    'image' field (it is an Image feature type, not a file path).

    Args:
        split:       "train", "test", or "val" (if available).
        subset_size: Limit dataset to first N samples. None = full dataset.
        transform:   torchvision transform applied to PIL images.
    """

    HF_DATASET_ID = "osv5m/osv5m"

    def __init__(
        self,
        split: str = "train",
        subset_size: Optional[int] = None,
        transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ):
        from datasets import load_dataset

        print(f"[Dataset] Loading OSV-5M split='{split}' ...")

        kwargs = dict(trust_remote_code=True, cache_dir=cache_dir)

        try:
            # First, try to load from local storage to avoid any network check/download
            hf = load_dataset(self.HF_DATASET_ID, split=split, local_files_only=True, **kwargs)
            if subset_size is not None:
                self.samples = hf.select(range(min(subset_size, len(hf))))
                print(f"[Dataset] Loaded {len(self.samples)} samples from local storage.")
            else:
                self.samples = hf
                print(f"[Dataset] Loaded full split ({len(hf)} samples) from local storage.")
        except Exception as e:
            if local_files_only:
                raise RuntimeError(f"local_files_only=True but dataset not found locally: {e}")

            # Fallback to normal loading (streaming for subsets, download for full) if not found locally
            if subset_size is not None:
                hf = load_dataset(self.HF_DATASET_ID, split=split, streaming=True, **kwargs)
                self.samples = list(hf.take(subset_size))
                print(f"[Dataset] Streamed {len(self.samples)} samples (local storage not found).")
            else:
                hf = load_dataset(self.HF_DATASET_ID, split=split, **kwargs)
                self.samples = hf
                print(f"[Dataset] Full split: {len(hf)} samples.")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        try:
            item = self.samples[idx]
            image: Image.Image = item["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            lat = float(item["latitude"])
            lon = float(item["longitude"])
            if self.transform is not None:
                image = self.transform(image)
            coords = torch.tensor([lat, lon], dtype=torch.float32)
            return image, coords

        except Exception:
            # Corrupted sample — return a neighbour
            traceback.print_exc()
            fallback_idx = (idx + 1) % len(self)
            return self.__getitem__(fallback_idx)


class OSV5MStreamingDataset(torch.utils.data.IterableDataset):
    """
    Streaming variant for very large datasets that don't fit in memory.
    Does not support random access or len().
    """

    HF_DATASET_ID = "osv5m/osv5m"

    def __init__(
        self,
        split: str = "train",
        subset_size: Optional[int] = None,
        transform: Optional[Callable] = None,
        shuffle_buffer: int = 10_000,
        seed: int = 42,
    ):
        from datasets import load_dataset

        hf = load_dataset(
            self.HF_DATASET_ID,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        hf = hf.shuffle(buffer_size=shuffle_buffer, seed=seed)
        if subset_size is not None:
            hf = hf.take(subset_size)

        self.hf_dataset = hf
        self.transform = transform

    def __iter__(self):
        for item in self.hf_dataset:
            try:
                image: Image.Image = item["image"]
                if image.mode != "RGB":
                    image = image.convert("RGB")
                lat = float(item["latitude"])
                lon = float(item["longitude"])
                if self.transform is not None:
                    image = self.transform(image)
                coords = torch.tensor([lat, lon], dtype=torch.float32)
                yield image, coords
            except Exception:
                continue


class LocalZipOSV5MDataset(Dataset):
    """
    Dataset that reads images from locally downloaded OSV-5M zip files.

    Expects the OSV-5M zip archives in ``zip_dir`` (e.g. 00.zip … 09.zip)
    and the corresponding CSV metadata file (train.csv / test.csv) at
    ``csv_path``.  Only shards whose zip file is present are used.

    Images inside each zip are stored as ``{shard:02d}/{image_id}.jpg``.
    The CSV ``id`` column contains the numeric ``image_id`` that is used to
    join with zip contents.

    Multiprocessing-safe: zip file handles are opened lazily inside each
    DataLoader worker and excluded from pickling.

    Args:
        zip_dir:  Directory containing the zip archives (e.g. ``images/train``).
        csv_path: Path to the metadata CSV (e.g. ``/data/.../train.csv``).
        transform: torchvision transform applied to PIL images.
        shards:   Explicit list of shard indices to include (0-based).
                  ``None`` uses all zip files found in ``zip_dir``.
    """

    def __init__(
        self,
        zip_dir: str,
        csv_path: str,
        transform: Optional[Callable] = None,
        shards: Optional[List[int]] = None,
    ):
        self.zip_dir = zip_dir
        self.transform = transform

        # Discover available zip files
        if shards is not None:
            zip_paths = {
                f"{s:02d}": os.path.join(zip_dir, f"{s:02d}.zip")
                for s in shards
            }
            missing = [p for p in zip_paths.values() if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(f"Shard zip(s) not found: {missing}")
        else:
            found = sorted(glob.glob(os.path.join(zip_dir, "*.zip")))
            zip_paths = {
                os.path.splitext(os.path.basename(p))[0]: p for p in found
            }

        if not zip_paths:
            raise FileNotFoundError(f"No zip files found in {zip_dir}")

        print(f"[LocalZipDataset] Indexing {len(zip_paths)} shard(s): {sorted(zip_paths)}")

        # Build image-id → (shard_key, path_in_zip) index from zip TOC
        id_to_loc: Dict[str, Tuple[str, str]] = {}
        for shard_key, zip_path in zip_paths.items():
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if name.endswith(".jpg") or name.endswith(".jpeg"):
                        stem = os.path.splitext(os.path.basename(name))[0]
                        id_to_loc[stem] = (shard_key, name)

        print(f"[LocalZipDataset] Indexed {len(id_to_loc):,} images from zips")

        # Read CSV and join with zip index
        samples: List[Tuple[str, str, float, float]] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = str(int(float(row["id"])))  # normalise numeric id
                if img_id not in id_to_loc:
                    continue
                shard_key, path_in_zip = id_to_loc[img_id]
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                try:
                    climate = int(float(row.get("climate") or 0))
                except (ValueError, TypeError):
                    climate = 0
                samples.append((shard_key, path_in_zip, lat, lon, climate))

        print(f"[LocalZipDataset] {len(samples):,} matched samples after CSV join")
        self.samples = samples
        self._zip_paths = {k: v for k, v in zip_paths.items()}
        self._zips: Dict[str, zipfile.ZipFile] = {}
        # Climate codes from the CSV (integer 1-30, 0 = ocean/unknown)
        self.climate_codes: List[int] = [s[4] for s in samples]

    # ------------------------------------------------------------------
    # Pickling support for DataLoader multiprocessing
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zips"] = {}  # drop open file handles; will reopen in workers
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._zips = {}

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        shard_key, path_in_zip, lat, lon, *_ = self.samples[idx]
        try:
            if shard_key not in self._zips:
                self._zips[shard_key] = zipfile.ZipFile(self._zip_paths[shard_key])
            jpg_bytes = self._zips[shard_key].read(path_in_zip)
            image = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, torch.tensor([lat, lon], dtype=torch.float32)
        except Exception:
            traceback.print_exc()
            return self.__getitem__((idx + 1) % len(self))
