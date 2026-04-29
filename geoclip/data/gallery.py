"""
GPS gallery construction and retrieval.

The gallery is a fixed set of GPS coordinates used as the retrieval database
at inference time. Predicted location = gallery point most similar to image embedding.
"""
from __future__ import annotations

import os
import torch
import numpy as np
from typing import Optional


def build_uniform_gallery(size: int = 10_000) -> torch.Tensor:
    """
    Sample GPS points uniformly at random over the globe.
    Returns:
        [size, 2] tensor of (lat, lon) in degrees.
    """
    lat = torch.FloatTensor(size).uniform_(-90, 90)
    lon = torch.FloatTensor(size).uniform_(-180, 180)
    return torch.stack([lat, lon], dim=1)


def build_train_sample_gallery(
    dataset,
    size: int = 10_000,
    seed: int = 42,
) -> torch.Tensor:
    """
    Sample GPS points from the training dataset.
    Covers only land regions actually present in OSV-5M.

    Args:
        dataset: OSV5MDataset instance (or any dataset returning (image, coords)).
        size:    Number of gallery points to sample.
        seed:    RNG seed for reproducibility.

    Returns:
        [size, 2] tensor of (lat, lon) in degrees.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = rng.choice(n, size=min(size, n), replace=False)

    coords_list = []
    for idx in indices:
        _, coords = dataset[int(idx)]
        coords_list.append(coords)

    return torch.stack(coords_list)  # [size, 2]


def load_or_build_gallery(
    strategy: str,
    size: int,
    cache_path: str,
    dataset=None,
) -> torch.Tensor:
    """Load gallery from cache if available, else build and cache it."""
    if os.path.exists(cache_path):
        print(f"[Gallery] Loading from cache: {cache_path}")
        return torch.load(cache_path)

    print(f"[Gallery] Building gallery (strategy='{strategy}', size={size}) ...")
    if strategy == "uniform":
        gallery = build_uniform_gallery(size)
    elif strategy == "train_sample":
        if dataset is None:
            raise ValueError("dataset required for 'train_sample' strategy")
        gallery = build_train_sample_gallery(dataset, size)
    else:
        raise ValueError(f"Unknown gallery strategy: {strategy}")

    torch.save(gallery, cache_path)
    print(f"[Gallery] Saved to {cache_path}")
    return gallery


@torch.no_grad()
def compute_gallery_embeddings(
    model,
    gallery_coords: torch.Tensor,
    batch_size: int = 512,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Pre-compute GPS embeddings for all gallery points.

    Args:
        model:          GeoCLIP model.
        gallery_coords: [G, 2] gallery GPS coordinates.
        batch_size:     Batch size for embedding computation.
        device:         Target device.

    Returns:
        [G, D] L2-normalized gallery embeddings.
    """
    model.eval()
    gallery_coords = gallery_coords.to(device)
    embeddings = []
    print(f"[Gallery] Computing embeddings for {len(gallery_coords):,} points ...", flush=True)

    for i in range(0, len(gallery_coords), batch_size):
        batch = gallery_coords[i : i + batch_size]
        emb = model.encode_gps(batch)
        embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)
