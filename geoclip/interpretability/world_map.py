"""
World map similarity heatmap for GeoCLIP.

Plots the model's full geographic belief distribution: rather than discarding
all but the argmax, we colour every gallery point by its cosine similarity
to the query image embedding.  This reveals:

  - Confidence: a sharp peak = certain prediction; spread = uncertain.
  - Geographic confusion: mass on two distant continents suggests the model
    is responding to a spurious visual cue shared by both regions.
  - Systematic bias: if mass concentrates on a wrong continent, the GPS
    encoder and/or training distribution are misaligned.

No extra dependencies beyond matplotlib.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_similarity_map(
    gallery_coords: torch.Tensor,
    similarities: torch.Tensor,
    true_coords: Optional[torch.Tensor] = None,
    pred_coords: Optional[torch.Tensor] = None,
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot gallery GPS points coloured by their cosine similarity to one image.

    Args:
        gallery_coords: [G, 2] (lat, lon) in degrees.
        similarities:   [G] similarity scores (will be min-max normalised).
        true_coords:    [2] true (lat, lon) — plotted as a green star.
        pred_coords:    [2] predicted (lat, lon) — plotted as a red cross.
        title:          Axes title string.
        ax:             Existing axes to draw on; creates one if None.

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    lons = gallery_coords[:, 1].numpy()
    lats = gallery_coords[:, 0].numpy()
    sims = similarities.numpy()

    # Normalise to [0, 1] for colour mapping
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

    ax.set_facecolor("#d0e8f0")          # ocean colour
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.tick_params(labelsize=6)
    ax.grid(color="white", linewidth=0.4, alpha=0.6)
    ax.set_xlabel("Longitude", fontsize=7)
    ax.set_ylabel("Latitude", fontsize=7)

    sc = ax.scatter(
        lons, lats,
        c=sims,
        cmap="hot",
        s=6,
        alpha=0.8,
        linewidths=0,
        zorder=2,
    )
    plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.02, label="Similarity")

    if true_coords is not None:
        ax.scatter(
            true_coords[1].item(), true_coords[0].item(),
            marker="*", s=180, color="lime", edgecolors="black",
            linewidths=0.5, zorder=5, label="True",
        )
    if pred_coords is not None:
        ax.scatter(
            pred_coords[1].item(), pred_coords[0].item(),
            marker="X", s=120, color="red", edgecolors="black",
            linewidths=0.5, zorder=5, label="Pred",
        )

    if true_coords is not None or pred_coords is not None:
        ax.legend(fontsize=6, loc="lower left")

    ax.set_title(title, fontsize=8)
    return ax


def plot_similarity_grid(
    model,
    images: torch.Tensor,
    gallery_coords: torch.Tensor,
    gallery_embs: torch.Tensor,
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    distances: torch.Tensor,
    output_path: str,
) -> None:
    """
    Save a figure with one world-map panel per image in the batch.

    Args:
        model:          GeoCLIP model (used to encode images).
        images:         [B, 3, H, W] CLIP-normalized images.
        gallery_coords: [G, 2] gallery GPS coordinates.
        gallery_embs:   [G, D] pre-computed gallery embeddings.
        true_coords:    [B, 2] ground-truth coordinates.
        pred_coords:    [B, 2] predicted coordinates.
        distances:      [B] haversine distances in km.
        output_path:    Where to save the figure.
    """
    device = next(model.parameters()).device
    B = images.shape[0]

    model.eval()
    with torch.no_grad():
        img_embs = model.encode_image(images.to(device))       # [B, D]
        sims = img_embs @ gallery_embs.to(device).T            # [B, G]
    sims = sims.cpu()

    cols = min(B, 4)
    rows = (B + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(B):
        dist_km = distances[i].item()
        title = f"Dist: {dist_km:.0f} km"
        plot_similarity_map(
            gallery_coords,
            sims[i],
            true_coords=true_coords[i],
            pred_coords=pred_coords[i],
            title=title,
            ax=axes[i],
        )

    for ax in axes[B:]:
        ax.set_visible(False)

    plt.suptitle("Geographic Belief Distribution (brighter = higher similarity)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[WorldMap] Saved to {output_path}")
