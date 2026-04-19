"""
Evaluation utilities: gallery-based retrieval and GCD metrics.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from geoclip.utils.geo_math import haversine_distance
from geoclip.data.gallery import compute_gallery_embeddings


GCD_THRESHOLDS_KM: List[int] = [1, 25, 200, 750, 2500]


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    gallery_coords: torch.Tensor,
    device: str = "cpu",
    thresholds_km: List[int] = GCD_THRESHOLDS_KM,
    gallery_batch_size: int = 512,
) -> Dict[str, float]:
    """
    Evaluate geo-localization accuracy using gallery-based retrieval.

    For each image, the predicted location is the gallery point whose GPS
    embedding is most similar to the image embedding (cosine similarity).

    Args:
        model:             GeoCLIP model.
        dataloader:        Evaluation DataLoader (images, true_coords).
        gallery_coords:    [G, 2] gallery GPS coordinates in degrees.
        device:            Device string.
        thresholds_km:     List of distance thresholds for recall@km metrics.
        gallery_batch_size: Batch size for pre-computing gallery embeddings.

    Returns:
        Dict with keys: mean_gcd, median_gcd, recall@{t}km for each threshold.
    """
    model.eval()
    gallery_coords = gallery_coords.to(device)

    # Pre-compute gallery embeddings
    print("[Eval] Pre-computing gallery embeddings ...")
    gallery_embs = compute_gallery_embeddings(
        model, gallery_coords, batch_size=gallery_batch_size, device=device
    )
    gallery_embs = gallery_embs.to(device)  # [G, D]
    gallery_coords_dev = gallery_coords.to(device)

    all_distances = []
    for images, true_coords in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        true_coords = true_coords.to(device)

        img_embs = model.encode_image(images)    # [B, D]
        sims = img_embs @ gallery_embs.T         # [B, G]
        best_idx = sims.argmax(dim=-1)           # [B]
        pred_coords = gallery_coords_dev[best_idx]  # [B, 2]

        dist = haversine_distance(pred_coords, true_coords)  # [B]
        all_distances.append(dist.cpu())

    distances = torch.cat(all_distances)  # [N]

    metrics: Dict[str, float] = {
        "mean_gcd_km": distances.mean().item(),
        "median_gcd_km": distances.median().item(),
    }
    for thresh in thresholds_km:
        recall = (distances <= thresh).float().mean().item()
        metrics[f"recall@{thresh}km"] = recall

    return metrics


@torch.no_grad()
def evaluate_by_zone(
    model,
    dataloader: DataLoader,
    gallery_coords: torch.Tensor,
    classifier,
    device: str = "cpu",
    thresholds_km: List[int] = GCD_THRESHOLDS_KM,
    gallery_batch_size: int = 512,
) -> Dict[str, Dict]:
    """
    Gallery-based evaluation broken down by Köppen major climate group.

    Args:
        classifier: a KoppenClassifier instance (geoclip.utils.koppen).

    Returns:
        Dict keyed by group letter (A/B/C/D/E/?).  Each value is a metrics
        dict with the same keys as evaluate() plus 'count'.
    """
    model.eval()
    gallery_coords = gallery_coords.to(device)

    gallery_embs = compute_gallery_embeddings(
        model, gallery_coords, batch_size=gallery_batch_size, device=device
    )
    gallery_embs = gallery_embs.to(device)
    gallery_coords_dev = gallery_coords.to(device)

    # Collect distances grouped by true Köppen group
    group_distances: Dict[str, list] = defaultdict(list)

    for images, true_coords in tqdm(dataloader, desc="Evaluating by zone"):
        images     = images.to(device)
        true_coords = true_coords.to(device)

        img_embs   = model.encode_image(images)
        best_idx   = (img_embs @ gallery_embs.T).argmax(dim=-1)
        pred_coords = gallery_coords_dev[best_idx]
        dists      = haversine_distance(pred_coords, true_coords).cpu()

        true_np = true_coords.cpu().numpy()
        groups  = classifier.get_group(true_np[:, 0], true_np[:, 1])
        if isinstance(groups, str):
            groups = [groups]

        for g, d in zip(groups, dists.tolist()):
            group_distances[g].append(d)

    results: Dict[str, Dict] = {}
    for group, dists_list in group_distances.items():
        d = torch.tensor(dists_list)
        m: Dict = {
            "count":        len(d),
            "mean_gcd_km":  d.mean().item(),
            "median_gcd_km": d.median().item(),
        }
        for thresh in thresholds_km:
            m[f"recall@{thresh}km"] = (d <= thresh).float().mean().item()
        results[group] = m

    return results
