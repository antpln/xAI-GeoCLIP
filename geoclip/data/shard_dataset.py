from __future__ import annotations

import gc
import os
import queue
import shutil
import threading
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class ShardedOSV5MDataset(Dataset):
    """
    Rotates OSV-5M shards with background prefetching.

    While one shard is being used for training, the next one is downloaded
    in a background thread. Calling next_shard() blocks only if the prefetch
    is not ready yet, then swaps atomically and deletes the old shard.

    Each shard is downloaded into its own isolated cache directory so that
    deleting a consumed shard never touches data still being downloaded.

    Disk usage: (shards_per_step + prefetch) * ~5 GB.

    Args:
        split:           "train" or "test".
        transform:       torchvision transform applied to each image.
        shards_per_step: number of shard files loaded per epoch (default 1).
        prefetch:        number of future shards to download in the background.
        hf_home:         HuggingFace home dir (defaults to HF_HOME env var).
        start_shard:     index of the first shard to load.
    """

    REPO_ID = "osv5m/osv5m"

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        shards_per_step: int = 1,
        prefetch: int = 1,
        hf_home: Optional[str] = None,
        start_shard: int = 0,
    ):
        from huggingface_hub import list_repo_files

        self.split = split
        self.transform = transform
        self.shards_per_step = shards_per_step
        self.hf_home = hf_home or os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )

        print(f"[ShardedDataset] Discovering shards for split='{split}' ...")
        all_files = list(list_repo_files(self.REPO_ID, repo_type="dataset"))

        skip_exts = {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".sh"}
        self.shard_files = sorted([
            f for f in all_files
            if split in f.lower()
            and os.path.splitext(f)[1] not in skip_exts
            and not f.startswith(".")
        ])

        if not self.shard_files:
            raise RuntimeError(f"No shard files found for split='{split}' in {self.REPO_ID}")

        n = len(self.shard_files)
        disk_gb = (shards_per_step + prefetch) * 5
        print(
            f"[ShardedDataset] {n} shards (~{n * 5} GB total). "
            f"Disk usage: ~{disk_gb} GB "
            f"({shards_per_step} active + {prefetch} prefetch)."
        )

        self._current_idx = start_shard % n
        self._current_cache: Optional[str] = None
        self._hf_dataset = None

        # Queue holds (hf_dataset, cache_dir) pairs ready for consumption.
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()

        # Load the first shard synchronously so the dataset is usable immediately.
        cache, ds = self._download(self._current_idx)
        self._current_cache = cache
        self._hf_dataset = ds

        # Start background thread that prefetches subsequent shards.
        if prefetch > 0:
            next_idx = (self._current_idx + shards_per_step) % n
            threading.Thread(
                target=self._worker, args=(next_idx,), daemon=True
            ).start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_shard(self) -> None:
        """
        Swap to the next shard. Blocks until the prefetch is ready, then
        deletes the old shard's cache directory to free disk space.
        """
        print("[ShardedDataset] Waiting for prefetched shard ...")
        new_cache, new_ds = self._queue.get(block=True)

        old_cache = self._current_cache
        self._current_cache = new_cache
        self._hf_dataset = new_ds
        self._current_idx = (self._current_idx + self.shards_per_step) % len(self.shard_files)

        if old_cache and os.path.exists(old_cache):
            shutil.rmtree(old_cache)
            print(f"[ShardedDataset] Deleted: {os.path.basename(old_cache)}")

        gc.collect()

    def stop(self) -> None:
        """Signal the background thread to exit cleanly."""
        self._stop.set()

    @property
    def shard_progress(self) -> str:
        n = len(self.shard_files)
        end = (self._current_idx + self.shards_per_step - 1) % n
        return f"shards {self._current_idx}–{end} / {n}"

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._hf_dataset)

    def __getitem__(self, idx: int):
        try:
            item = self._hf_dataset[idx]
            image: Image.Image = item["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            lat = float(item["latitude"])
            lon = float(item["longitude"])
            if self.transform is not None:
                image = self.transform(image)
            return image, torch.tensor([lat, lon], dtype=torch.float32)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shard_cache_dir(self, idx: int) -> str:
        return os.path.join(self.hf_home, f"osv5m_shard_{idx:05d}")

    def _download(self, start_idx: int) -> Tuple[str, object]:
        """Download shards_per_step files into an isolated cache dir and return the dataset.

        Uses snapshot_download with allow_patterns so the HuggingFace Hub layer
        physically fetches *only* the requested zip(s) before the loading script
        runs. This is necessary because the OSV-5M loading script ignores the
        data_files parameter during its download step and would otherwise pull all
        98 shards (~490 GB).
        """
        from huggingface_hub import snapshot_download
        from datasets import load_dataset

        indices = [
            (start_idx + i) % len(self.shard_files)
            for i in range(self.shards_per_step)
        ]
        files = [self.shard_files[i] for i in indices]
        cache_dir = self._shard_cache_dir(start_idx)

        print(f"[ShardedDataset] Downloading shards {indices} → {os.path.basename(cache_dir)}")

        # Download only the target shard zip(s) plus repo metadata/scripts.
        # snapshot_download enforces the pattern at the HTTP level, so no other
        # shard zips are transferred regardless of what the loading script requests.
        snapshot_download(
            repo_id=self.REPO_ID,
            repo_type="dataset",
            allow_patterns=files + ["*.py", "*.json", "*.yaml", "*.yml", "*.md", "*.txt"],
            local_dir=cache_dir,
        )

        # Load from the local snapshot. Pass data_files as absolute local paths so
        # the loading script uses only the file(s) we just downloaded.
        local_files = {self.split: [os.path.join(cache_dir, f) for f in files]}
        ds = load_dataset(
            cache_dir,
            data_files=local_files,
            split=self.split,
            trust_remote_code=True,
        )

        print(f"[ShardedDataset] Ready: {len(ds)} samples in {os.path.basename(cache_dir)}")
        return cache_dir, ds

    def _worker(self, start_idx: int) -> None:
        """Background thread: continuously download shards and push to the queue."""
        idx = start_idx
        while not self._stop.is_set():
            try:
                result = self._download(idx)
            except Exception as e:
                print(f"[ShardedDataset] Prefetch error shard {idx}: {e}")
                idx = (idx + self.shards_per_step) % len(self.shard_files)
                continue

            # Push to queue; retry with timeout so the stop event is checked.
            while not self._stop.is_set():
                try:
                    self._queue.put(result, timeout=1)
                    break
                except queue.Full:
                    continue

            idx = (idx + self.shards_per_step) % len(self.shard_files)


_SHARD_SIZE_ESTIMATE = 51_000  # ~5 M samples / 98 shards
