from __future__ import annotations

import gc
import io
import os
import queue
import shutil
import threading
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset


# ---------------------------------------------------------------------------
# Shard catalogue for osv5m/osv5m-wds
# ---------------------------------------------------------------------------
# Each tar shard contains {jpg, json} pairs.  Train shards are ~670 MB with
# 10 000 samples; val and test shards are ~60–70 MB with 1 000 samples each.
# These counts come from the sizes.json files in the repository.

_SHARD_CATALOGUE = {
    "train": {"n": 490, "per_shard": 10_000, "size_mb": 670},
    "val":   {"n":  49, "per_shard":  1_000, "size_mb":  67},
    "test":  {"n": 211, "per_shard":  1_000, "size_mb":  61},
}

_HF_BASE = "https://huggingface.co/datasets/osv5m/osv5m-wds/resolve/main"


class ShardedOSV5MDataset(Dataset):
    """
    Rotates osv5m-wds shards with background prefetching.

    Uses the WebDataset (tar) version of OSV-5M instead of the zip-based
    original.  Key improvements over the zip variant:

    - No custom loading script (no load_dataset, no RecursionError).
    - Train shards are ~670 MB instead of ~5 GB — 7× smaller.
    - Validation split exists and each shard is only ~67 MB.
    - TAR format is sequential: hf_hub_download fetches the exact file
      requested with a single HTTP request.

    While one shard is training, the next is downloaded in a daemon thread.
    Calling next_shard() blocks only if the prefetch is not ready yet, then
    swaps and deletes the old directory.

    Disk usage: (shards_per_step + prefetch) × shard_size_mb MB.

    Args:
        split:           "train", "val", or "test".
        transform:       torchvision transform applied to each image.
        shards_per_step: number of shard tars loaded per epoch (default 1).
        prefetch:        number of future shards to download concurrently.
        hf_home:         root dir for isolated shard caches.
        start_shard:     index of the first shard (0-based).
    """

    REPO_ID = "osv5m/osv5m-wds"

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        shards_per_step: int = 1,
        num_shards: Optional[int] = None,
        prefetch: int = 1,
        hf_home: Optional[str] = None,
        start_shard: int = 0,
    ):
        if split not in _SHARD_CATALOGUE:
            raise ValueError(f"split must be one of {list(_SHARD_CATALOGUE)}, got '{split}'")

        self.split = split
        self.transform = transform
        self.shards_per_step = shards_per_step
        self.hf_home = hf_home or os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )

        cat = _SHARD_CATALOGUE[split]
        self.n_shards = num_shards if num_shards is not None else cat["n"]
        size_mb = (shards_per_step + prefetch) * cat["size_mb"]
        print(
            f"[ShardedDataset] {split}: {self.n_shards} shards used (max {cat['n']}) × "
            f"~{cat['per_shard']} samples (~{cat['size_mb']} MB each). "
            f"Active disk: ~{size_mb} MB."
        )

        self._current_idx = start_shard % self.n_shards
        self._current_cache: Optional[str] = None
        self._samples: List[Tuple] = []

        self._queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()

        cache, samples = self._download(self._current_idx)
        self._current_cache = cache
        self._samples = samples

        if prefetch > 0:
            next_idx = (self._current_idx + shards_per_step) % self.n_shards
            threading.Thread(
                target=self._worker, args=(next_idx,), daemon=True
            ).start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_shard(self) -> None:
        """Block until the prefetched shard is ready, then rotate."""
        print("[ShardedDataset] Waiting for prefetched shard ...")
        new_cache, new_samples = self._queue.get(block=True)

        old_cache = self._current_cache
        self._current_cache = new_cache
        self._samples = new_samples
        self._current_idx = (self._current_idx + self.shards_per_step) % self.n_shards

        if old_cache and os.path.exists(old_cache):
            shutil.rmtree(old_cache)
            print(f"[ShardedDataset] Deleted: {os.path.basename(old_cache)}")

        gc.collect()

    def stop(self) -> None:
        self._stop.set()

    @property
    def shard_progress(self) -> str:
        end = (self._current_idx + self.shards_per_step - 1) % self.n_shards
        return f"shards {self._current_idx}–{end} / {self.n_shards}"

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        jpg_bytes, lat, lon = self._samples[idx]
        try:
            image = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, torch.tensor([lat, lon], dtype=torch.float32)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shard_cache_dir(self, idx: int) -> str:
        return os.path.join(self.hf_home, f"osv5m_wds_{self.split}_{idx:05d}")

    def _download(self, start_idx: int) -> Tuple[str, List[Tuple]]:
        """Download shards_per_step tar files and extract (jpg_bytes, lat, lon) tuples.

        hf_hub_download fetches a single named file — no loading script,
        no recursive download, no ZIP central-directory HTTP range tricks.
        Raw JPEG bytes are kept in memory; PIL decoding happens lazily in
        __getitem__ so DataLoader workers can parallelise it.
        """
        import webdataset as wds
        from huggingface_hub import hf_hub_download

        indices = [
            (start_idx + i) % self.n_shards
            for i in range(self.shards_per_step)
        ]
        cache_dir = self._shard_cache_dir(start_idx)
        os.makedirs(cache_dir, exist_ok=True)

        print(f"[ShardedDataset] Downloading shards {indices}")

        tar_paths = []
        for idx in indices:
            fname = f"{self.split}/{idx:04d}.tar"
            tar_path = hf_hub_download(
                repo_id=self.REPO_ID,
                repo_type="dataset",
                filename=fname,
                local_dir=cache_dir,
            )
            tar_paths.append(tar_path)

        # Iterate the tar(s) once to collect raw JPEG bytes + coordinates.
        # Storing bytes (not PIL images) keeps RAM usage minimal.
        # Note: wds returns the JSON field as raw bytes — decode before use.
        import json as _json
        samples: List[Tuple] = []
        for tar_path in tar_paths:
            for jpg_bytes, meta_bytes in wds.WebDataset(tar_path).to_tuple("jpg", "json"):
                try:
                    meta = _json.loads(meta_bytes)
                    lat = float(meta["latitude"])
                    lon = float(meta["longitude"])
                    samples.append((jpg_bytes, lat, lon))
                except (KeyError, ValueError, _json.JSONDecodeError):
                    continue

        print(f"[ShardedDataset] Ready: {len(samples)} samples")
        return cache_dir, samples

    def _worker(self, start_idx: int) -> None:
        """Daemon thread: pre-download the next shard(s) and park in the queue."""
        idx = start_idx
        while not self._stop.is_set():
            # Wait until there is space in the queue before downloading,
            # so we never hold more than (prefetch) extra shards in memory.
            while not self._stop.is_set():
                if not self._queue.full():
                    break
                self._stop.wait(timeout=0.5)
            if self._stop.is_set():
                break

            try:
                result = self._download(idx)
            except Exception as e:
                print(f"[ShardedDataset] Prefetch error shard {idx}: {e}")
                idx = (idx + self.shards_per_step) % self.n_shards
                continue

            while not self._stop.is_set():
                try:
                    self._queue.put(result, timeout=1)
                    break
                except queue.Full:
                    continue

            idx = (idx + self.shards_per_step) % self.n_shards


class StreamingOSV5MDataset(IterableDataset):
    """
    OSV-5M (wds) via WebDataset streaming — no pre-download, zero disk usage.

    Streams tar shards directly from the HuggingFace Hub using the webdataset
    library.  Unlike the zip-based HF streaming loader, WebDataset tars are
    sequential so there are no HTTP range-request penalties.

    Works correctly with num_workers > 0: webdataset distributes shards
    across DataLoader workers automatically (each worker gets a disjoint
    subset of shards).

        DataLoader(ds, num_workers=4, prefetch_factor=2)

    Args:
        split:          "train", "val", or "test".
        transform:      torchvision transform applied to each PIL image.
        shuffle_buffer: in-memory reservoir size for shuffling (0 = no shuffle).
        shardshuffle:   whether to shuffle shard order (default True for train).
        hf_home:        unused (kept for API compatibility with ShardedOSV5MDataset).
    """

    REPO_ID = "osv5m/osv5m-wds"

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        num_shards: Optional[int] = None,
        shuffle_buffer: int = 1000,
        shardshuffle: bool = True,
        hf_home: Optional[str] = None,  # kept for API compat
    ):
        if split not in _SHARD_CATALOGUE:
            raise ValueError(f"split must be one of {list(_SHARD_CATALOGUE)}, got '{split}'")
        self.split = split
        self.transform = transform
        self.num_shards = num_shards
        self.shuffle_buffer = shuffle_buffer
        self.shardshuffle = shardshuffle

    def __iter__(self):
        import webdataset as wds

        cat = _SHARD_CATALOGUE[self.split]
        n = self.num_shards if self.num_shards is not None else cat["n"]
        # WebDataset brace expansion: {0000..0489}
        url = f"{_HF_BASE}/{self.split}/{{{0:04d}..{n - 1:04d}}}.tar"

        ds = wds.WebDataset(url, shardshuffle=self.shardshuffle)
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(self.shuffle_buffer)
        ds = ds.to_tuple("jpg", "json")

        import json as _json
        for jpg_bytes, meta_bytes in ds:
            try:
                meta = _json.loads(meta_bytes)
                image = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
                lat = float(meta["latitude"])
                lon = float(meta["longitude"])
            except Exception:
                continue
            if self.transform:
                image = self.transform(image)
            yield image, torch.tensor([lat, lon], dtype=torch.float32)
