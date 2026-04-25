from __future__ import annotations

import traceback
from typing import Optional, Callable

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
