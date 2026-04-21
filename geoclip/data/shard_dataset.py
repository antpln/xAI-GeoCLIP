from __future__ import annotations

import gc
import os
import queue
import shutil
import threading
import zipfile
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset


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

        # Close any open ZipFile handles pointing at the old cache before
        # deleting the directory, so the OS can release the file descriptors.
        self._close_zip_handles()

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
        # _hf_dataset is a list of (zip_path, member_name, lat, lon) tuples.
        return len(self._hf_dataset)

    def __getitem__(self, idx: int):
        zip_path, member_name, lat, lon = self._hf_dataset[idx]
        try:
            zf = self._get_zip(zip_path)
            with zf.open(member_name) as f:
                image = Image.open(f).convert("RGB")
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

    def _get_zip(self, path: str) -> zipfile.ZipFile:
        """Return a cached ZipFile handle (one per process — safe for DataLoader workers)."""
        if not hasattr(self, "_zip_handles"):
            self._zip_handles: dict = {}
        if path not in self._zip_handles:
            self._zip_handles[path] = zipfile.ZipFile(path, "r")
        return self._zip_handles[path]

    def _close_zip_handles(self) -> None:
        for zf in getattr(self, "_zip_handles", {}).values():
            try:
                zf.close()
            except Exception:
                pass
        self._zip_handles = {}

    def _download(self, start_idx: int) -> Tuple[str, List[Tuple]]:
        """Download one shard zip + metadata CSV without touching the loading script.

        The OSV-5M loading script (osv5m.py) hardcodes all 98 shard URLs in
        _split_generators and ignores the data_files parameter entirely, so any
        path through load_dataset downloads the full ~490 GB.  We bypass it:

          1. hf_hub_download fetches only the target zip(s) and the metadata CSV.
          2. We parse the CSV with pandas and index by image ID.
          3. We enumerate zip members and join with the metadata.
          4. __getitem__ reads images directly from the ZipFile.

        This gives exact shard isolation with no loading-script involvement.
        """
        import pandas as pd
        from huggingface_hub import hf_hub_download

        indices = [
            (start_idx + i) % len(self.shard_files)
            for i in range(self.shards_per_step)
        ]
        files = [self.shard_files[i] for i in indices]
        cache_dir = self._shard_cache_dir(start_idx)
        os.makedirs(cache_dir, exist_ok=True)

        print(f"[ShardedDataset] Downloading shards {indices} → {os.path.basename(cache_dir)}")

        # Metadata CSV is small (~10 MB); cache it once in a shared directory.
        meta_cache = os.path.join(self.hf_home, "meta")
        meta_path = hf_hub_download(
            repo_id=self.REPO_ID,
            repo_type="dataset",
            filename=f"{self.split}.csv",
            local_dir=meta_cache,
        )

        # Download only the requested shard zip(s).
        zip_paths = []
        for fname in files:
            zip_path = hf_hub_download(
                repo_id=self.REPO_ID,
                repo_type="dataset",
                filename=fname,
                local_dir=cache_dir,
            )
            zip_paths.append(zip_path)

        # Build id → (lat, lon) lookup from the metadata CSV.
        df = pd.read_csv(meta_path, dtype={"id": str},
                         usecols=["id", "latitude", "longitude"])
        meta = df.set_index("id")

        # Build sample list: (zip_path, member_name, lat, lon).
        # The loading script matches images by filename stem == row id.
        img_exts = {".jpg", ".jpeg", ".png", ".webp"}
        samples: List[Tuple] = []
        for zip_path in zip_paths:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    if os.path.splitext(name)[1].lower() not in img_exts:
                        continue
                    img_id = os.path.splitext(os.path.basename(name))[0]
                    if img_id not in meta.index:
                        continue
                    row = meta.loc[img_id]
                    samples.append((zip_path, name, float(row["latitude"]), float(row["longitude"])))

        print(f"[ShardedDataset] Ready: {len(samples)} samples in {os.path.basename(cache_dir)}")
        return cache_dir, samples

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


class StreamingOSV5MDataset(IterableDataset):
    """
    OSV-5M via HuggingFace streaming — no pre-download, zero disk management.

    Data is fetched lazily sample-by-sample directly from the Hub. Each
    DataLoader worker receives a disjoint shard slice so there is no overlap.

    Compared to ShardedOSV5MDataset:
      - Pro: no disk usage, no shard rotation logic, simpler setup.
      - Con: random-access shuffle is replaced by a bounded shuffle buffer;
             throughput is bottlenecked by network + JPEG decode rather than
             local I/O.  Use num_workers >= 4 and prefetch_factor=2 to hide
             latency.

    DataLoader requirements
    -----------------------
    MUST use num_workers=0.  HuggingFace streaming internally uses
    requests.Session objects that are not fork-safe; any num_workers > 0
    will deadlock on Colab (and most Linux environments that use fork).
    Network I/O releases the GIL so a single worker is not as slow as it
    sounds — the bottleneck is bandwidth, not the Python thread.

        DataLoader(ds, num_workers=0, pin_memory=True)

    Args:
        split:          "train" or "test".
        transform:      torchvision transform applied to each PIL image.
        shuffle_buffer: size of the in-memory shuffle reservoir (0 = no shuffle).
        seed:           random seed for the shuffle buffer.
        hf_home:        if set, overrides the HF_HOME environment variable.
    """

    REPO_ID = "osv5m/osv5m"

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        shuffle_buffer: int = 2048,
        seed: int = 42,
        hf_home: Optional[str] = None,
    ):
        self.split = split
        self.transform = transform
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        if hf_home:
            os.environ["HF_HOME"] = hf_home

    def __iter__(self):
        from datasets import load_dataset

        # Guard: warn if called from a DataLoader worker.  HF streaming is not
        # fork-safe; the caller must use num_workers=0.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise RuntimeError(
                "StreamingOSV5MDataset is not compatible with num_workers > 0. "
                "HuggingFace streaming uses requests.Session which is not fork-safe. "
                "Use DataLoader(..., num_workers=0)."
            )

        ds = load_dataset(
            self.REPO_ID,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )

        if self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)

        for sample in ds:
            img = sample["image"]  # PIL.Image decoded by the loading script
            coords = torch.tensor(
                [sample["latitude"], sample["longitude"]], dtype=torch.float32
            )
            if self.transform:
                img = self.transform(img)
            yield img, coords
