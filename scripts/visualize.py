"""
Visualization script for GeoCLIP — covers all interpretability and analysis
sections from the notebook.

Interpretability methods (patch/pixel-level):
    gradcam      — Grad-CAM on the last transformer block
    rollout      — Attention Rollout across all layers
    ig           — Integrated Gradients (completeness axiom)
    layerwise    — Grad-CAM at blocks 2, 5, 8, 11
    heads        — Per-head attention at the last layer
    worldmap     — Geographic belief distribution over the gallery
    all_methods  — Side-by-side comparison of all five methods
    good_vs_bad  — Grad-CAM + IG on correct vs. wrong predictions
    topk         — Top-K gallery candidates on a world map

Analysis (full validation set):
    koppen       — Köppen-Geiger climate-zone error analysis
    error_dist   — Histogram + CDF of prediction errors
    tsne         — t-SNE of image and GPS embeddings
    calibration  — Similarity–error calibration
    zone_perf    — Per-climate-zone recall and median GCD

Training diagnostics (requires checkpoint directory):
    training_curves — Loss, GCD, and recall over training epochs

Usage:
    # Interpretability on a small batch
    python scripts/visualize.py \\
        --checkpoint outputs/checkpoints/best.pt \\
        --config configs/default.yaml \\
        --hf_home /data/hf_cache \\
        --methods gradcam rollout ig all_methods good_vs_bad topk worldmap \\
        --num_samples 4 \\
        --output_dir outputs/viz/

    # Full-set analysis
    python scripts/visualize.py \\
        --checkpoint outputs/checkpoints/best.pt \\
        --config configs/default.yaml \\
        --hf_home /data/hf_cache \\
        --methods koppen error_dist tsne calibration zone_perf \\
        --eval_split val --eval_subset 2000 \\
        --output_dir outputs/viz/

    # Training diagnostics
    python scripts/visualize.py \\
        --checkpoint outputs/checkpoints/best.pt \\
        --config configs/default.yaml \\
        --methods training_curves \\
        --checkpoint_dir outputs/checkpoints/ \\
        --output_dir outputs/viz/
"""
import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from tqdm import tqdm

from geoclip.models.geoclip import GeoCLIP
from geoclip.data.dataset import OSV5MDataset, LocalZipOSV5MDataset
from geoclip.data.transforms import get_eval_transform
from geoclip.data.gallery import load_or_build_gallery, compute_gallery_embeddings
from geoclip.interpretability.gradcam import gradcam_context, gradcam_layerwise
from geoclip.interpretability.attention_rollout import attention_rollout_context, PerHeadAttention
from geoclip.interpretability.integrated_gradients import IntegratedGradients
from geoclip.interpretability.world_map import plot_similarity_map, plot_similarity_grid
from geoclip.training.evaluator import evaluate, evaluate_by_zone
from geoclip.utils.checkpoint import load_checkpoint
from geoclip.utils.config import load_config
from geoclip.utils.geo_math import haversine_distance
from geoclip.utils.koppen import (
    KoppenClassifier, classify_error,
    KOPPEN_GROUPS, GROUP_COLORS,
)

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

BATCH_METHODS = {
    "gradcam", "rollout", "ig", "layerwise", "heads",
    "worldmap", "all_methods", "good_vs_bad", "topk",
}
FULLSET_METHODS = {"koppen", "error_dist", "tsne", "calibration", "zone_perf"}
CKPT_METHODS    = {"training_curves"}
TRAIN_METHODS   = {"contamination"}   # require --train_zip_dir + --train_csv
ALL_METHODS     = sorted(BATCH_METHODS | FULLSET_METHODS | CKPT_METHODS | TRAIN_METHODS)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def denormalize(t: torch.Tensor) -> np.ndarray:
    img = t.cpu() * CLIP_STD[:, None, None] + CLIP_MEAN[:, None, None]
    return (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay(img_np: np.ndarray, hmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    colored = (cm.jet(hmap)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * colored + (1 - alpha) * img_np).astype(np.uint8)


def save_grid(rows_data, col_titles, output_path, row_labels=None, suptitle=""):
    n_rows = len(rows_data)
    n_cols = len(rows_data[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    for r, row in enumerate(rows_data):
        for c, img in enumerate(row):
            ax = axes[r][c]
            if img is not None:
                ax.imshow(img)
            ax.axis("off")
            if r == 0 and col_titles:
                ax.set_title(col_titles[c], fontsize=8, fontweight="bold")
        if row_labels:
            axes[r][0].set_ylabel(row_labels[r], fontsize=8, rotation=90, labelpad=4)
    if suptitle:
        plt.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved {output_path}")


# ---------------------------------------------------------------------------
# Batch-level interpretability
# ---------------------------------------------------------------------------

def run_gradcam(model, images, true_coords, img_nps, out_dir):
    with gradcam_context(model) as gc:
        maps = gc.compute(images, true_coords)
    rows = [[img_nps[i], overlay(img_nps[i], maps[i].numpy())]
            for i in tqdm(range(len(images)), desc="gradcam samples", leave=False)]
    save_grid(rows, ["Input", "Grad-CAM"], os.path.join(out_dir, "gradcam.png"),
              suptitle="Grad-CAM")
    return maps


def run_rollout(model, images, img_nps, out_dir):
    with attention_rollout_context(model, head_fusion="mean", discard_ratio=0.1) as ar:
        maps = ar.compute(images)
    rows = [[img_nps[i], overlay(img_nps[i], maps[i].numpy())]
            for i in tqdm(range(len(images)), desc="rollout samples", leave=False)]
    save_grid(rows, ["Input", "Attention Rollout"], os.path.join(out_dir, "rollout.png"),
              suptitle="Attention Rollout")
    return maps


def run_ig(model, images, true_coords, img_nps, out_dir, n_steps=50):
    ig = IntegratedGradients(model, n_steps=n_steps)
    maps = ig.compute(images, true_coords)
    rows = [[img_nps[i], overlay(img_nps[i], maps[i].numpy())]
            for i in tqdm(range(len(images)), desc="ig samples", leave=False)]
    save_grid(rows, ["Input", "Integrated Gradients"],
              os.path.join(out_dir, "ig.png"), suptitle="Integrated Gradients")
    return maps


def run_layerwise(model, images, true_coords, img_nps, out_dir,
                  layer_indices=(2, 5, 8, 11)):
    lw = gradcam_layerwise(model, images, true_coords, layer_indices=layer_indices)
    rows = [
        [img_nps[i]] + [overlay(img_nps[i], lw[idx][i].numpy()) for idx in layer_indices]
        for i in tqdm(range(len(images)), desc="layerwise samples", leave=False)
    ]
    col_titles = ["Input"] + [f"Block {idx}" for idx in layer_indices]
    save_grid(rows, col_titles, os.path.join(out_dir, "layerwise_gradcam.png"),
              suptitle="Layer-wise Grad-CAM")
    return lw


def run_heads(model, images, img_nps, out_dir):
    pha = PerHeadAttention(model, layer_idx=-1)
    head_maps = pha.compute(images)
    num_heads = head_maps.shape[1]
    rows = [
        [img_nps[i]] + [overlay(img_nps[i], head_maps[i, h].numpy()) for h in range(num_heads)]
        for i in tqdm(range(len(images)), desc="head samples", leave=False)
    ]
    col_titles = ["Input"] + [f"Head {h} (rank {r+1})" for r, h in enumerate(range(num_heads))]
    save_grid(rows, col_titles, os.path.join(out_dir, "per_head_attention.png"),
              suptitle="Per-head Attention (last layer)")
    return head_maps


def run_worldmap(model, images, gallery_coords, gallery_embs,
                 true_coords, pred_coords, distances, out_dir):
    plot_similarity_grid(
        model=model,
        images=images,
        gallery_coords=gallery_coords,
        gallery_embs=gallery_embs,
        true_coords=true_coords,
        pred_coords=pred_coords,
        distances=distances,
        output_path=os.path.join(out_dir, "worldmap.png"),
    )


def run_all_methods(model, images, true_coords, img_nps,
                    gallery_embs, gallery_coords, distances, out_dir, n_steps=50):
    """Section 12 — all five methods side by side for each sample."""
    device = next(model.parameters()).device
    N = len(images)

    with gradcam_context(model) as gc:
        gcam_maps = gc.compute(images, true_coords)
    with attention_rollout_context(model, head_fusion="mean", discard_ratio=0.1) as ar:
        roll_maps = ar.compute(images)
    ig_maps = IntegratedGradients(model, n_steps=n_steps).compute(images, true_coords)
    lw = gradcam_layerwise(model, images, true_coords, layer_indices=(2, 5, 8, 11))
    ph = PerHeadAttention(model, layer_idx=-1).compute(images)
    ph_mean = ph.mean(dim=1)   # [N, H, W]

    method_names = ["Grad-CAM", "Rollout", "Integ. Grad", "GC Block 11", "Heads (avg)"]
    col_titles   = ["Input"] + method_names

    rows = []
    for i in tqdm(range(N), desc="all_methods samples", leave=False):
        rows.append([
            img_nps[i],
            overlay(img_nps[i], gcam_maps[i].numpy()),
            overlay(img_nps[i], roll_maps[i].numpy()),
            overlay(img_nps[i], ig_maps[i].numpy()),
            overlay(img_nps[i], lw[11][i].numpy()),
            overlay(img_nps[i], ph_mean[i].numpy()),
        ])

    row_labels = [f"{distances[i]:.0f} km" for i in range(N)]
    save_grid(rows, col_titles, os.path.join(out_dir, "all_methods.png"),
              row_labels=row_labels, suptitle="All methods — sample comparison")


def run_good_vs_bad(model, good_images, good_coords, good_dists,
                    bad_images, bad_coords, bad_dists, out_dir, n_steps=50):
    """Grad-CAM + IG comparison across good and bad predictions."""
    n_good, n_bad = len(good_images), len(bad_images)
    all_images = torch.cat([good_images, bad_images])
    all_coords = torch.cat([good_coords, bad_coords])
    all_dists  = torch.cat([good_dists,  bad_dists])

    with gradcam_context(model) as gc:
        gcam_all = gc.compute(all_images, all_coords)
    ig_all = IntegratedGradients(model, n_steps=n_steps).compute(all_images, all_coords)

    labels = ([f"Good — {d:.0f} km" for d in good_dists.tolist()] +
              [f"Bad  — {d:.0f} km" for d in bad_dists.tolist()])
    colors = ["green"] * n_good + ["red"] * n_bad

    fig, axes = plt.subplots(n_good + n_bad, 3, figsize=(10, 3.5 * (n_good + n_bad)))
    for row, (img_t, gc_map, ig_map, label, color) in enumerate(
        zip(all_images, gcam_all, ig_all, labels, colors)
    ):
        p = denormalize(img_t)
        axes[row][0].imshow(p);              axes[row][0].axis("off")
        axes[row][0].set_title(label, fontsize=9, color=color)
        axes[row][1].imshow(overlay(p, gc_map.numpy())); axes[row][1].axis("off")
        axes[row][1].set_title("Grad-CAM", fontsize=9)
        axes[row][2].imshow(overlay(p, ig_map.numpy())); axes[row][2].axis("off")
        axes[row][2].set_title("Integrated Gradients", fontsize=9)

    plt.suptitle("Good vs. Bad Predictions", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "good_vs_bad.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved {path}")


def run_topk(model, images, true_coords, img_nps, gallery_coords, gallery_embs,
             distances, out_dir, topk=5):
    """Section 19 — Top-K gallery candidates on a world map."""
    device = next(model.parameters()).device
    N = len(images)
    gallery_np = gallery_coords.numpy()

    model.eval()
    with torch.no_grad():
        q_embs = model.encode_image(images.to(device))
        sims_q = (q_embs @ gallery_embs.to(device).T).cpu()

    topk_vals, topk_idx = sims_q.topk(topk, dim=-1)

    fig, axes = plt.subplots(N, 2, figsize=(13, 3.5 * N),
                             gridspec_kw={"width_ratios": [1, 3]})
    if N == 1:
        axes = [axes]

    cmap_k = plt.cm.YlOrRd
    for i in range(N):
        axes[i][0].imshow(img_nps[i])
        axes[i][0].axis("off")
        axes[i][0].set_title(
            f"Query {i}\n({true_coords[i,0]:.1f}°, {true_coords[i,1]:.1f}°)\n"
            f"{distances[i]:.0f} km error", fontsize=8,
        )

        ax = axes[i][1]
        ax.set_facecolor("#d0e8f0")
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30))
        ax.tick_params(labelsize=5); ax.grid(color="white", linewidth=0.3, alpha=0.5)
        ax.scatter(gallery_np[:, 1], gallery_np[:, 0],
                   s=1, color="#aaaaaa", alpha=0.3, linewidths=0, zorder=1)

        for rank in range(topk - 1, -1, -1):
            idx_k = topk_idx[i, rank].item()
            lat_k, lon_k = gallery_np[idx_k]
            ax.scatter(lon_k, lat_k, s=80 - rank * 10,
                       color=cmap_k(1.0 - rank / topk),
                       edgecolors="black", linewidths=0.5, zorder=4 + rank,
                       label=f"#{rank+1}  sim={topk_vals[i,rank]:.3f}")

        tc = true_coords[i].numpy()
        ax.scatter(tc[1], tc[0], marker="*", s=250, color="lime",
                   edgecolors="black", linewidths=0.5, zorder=10, label="True")
        ax.legend(fontsize=6, loc="lower left", framealpha=0.85)
        ax.set_title(f"Top-{topk} gallery candidates", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap_k, norm=plt.Normalize(vmin=1, vmax=topk))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[axes[i][1] for i in range(N)], fraction=0.015, pad=0.02)
    cbar.set_label("Rank (1 = highest similarity)", fontsize=8)
    cbar.set_ticks(np.linspace(1, topk, topk))
    cbar.set_ticklabels([f"#{r}" for r in range(1, topk + 1)])

    plt.suptitle(f"Top-{topk} retrieved gallery points  (★ = true location)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "topk_retrieval.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved {path}")


# ---------------------------------------------------------------------------
# Full validation-set analyses
# ---------------------------------------------------------------------------

def run_koppen(model, val_loader, gallery_coords, gallery_embs, out_dir, device,
               climate_codes=None, pred_classifier=None):
    """Section 14 — Köppen-Geiger climate analysis."""
    kc = KoppenClassifier(require_raster=(climate_codes is None))
    all_true, all_pred, all_dist = [], [], []

    model.eval()
    for imgs, coords in tqdm(val_loader, desc="Köppen"):
        with torch.no_grad():
            e = model.encode_image(imgs.to(device))
            idx = (e @ gallery_embs.to(device).T).argmax(dim=-1).cpu()
        pred_c = gallery_coords[idx]
        all_true.append(coords)
        all_pred.append(pred_c)
        all_dist.append(haversine_distance(pred_c, coords))

    true_np = torch.cat(all_true).numpy()
    pred_np = torch.cat(all_pred).numpy()
    distances = torch.cat(all_dist)

    if climate_codes is not None:
        true_info = kc.classify_from_codes(climate_codes[:len(true_np)])
    else:
        true_info = kc.classify_batch(true_np[:, 0], true_np[:, 1])
    pred_kc = pred_classifier or (kc if kc._grid is not None else None)
    if pred_kc is not None:
        pred_info = pred_kc.classify_batch(pred_np[:, 0], pred_np[:, 1])
    else:
        pred_info = kc.classify_from_codes([0] * len(pred_np))
    coherence  = [classify_error(tg, pg)
                  for tg, pg in zip(true_info["groups"], pred_info["groups"])]

    # Coherence pie
    counts       = Counter(coherence)
    labels_order = ["exact", "adjacent", "distant", "ocean"]
    pie_colors   = ["#2ecc71", "#f39c12", "#e74c3c", "#95a5a6"]
    active = [(l, counts.get(l, 0), c) for l, c in zip(labels_order, pie_colors)
              if counts.get(l, 0) > 0]

    groups_order = ["A", "B", "C", "D", "E", "?"]
    n_g  = len(groups_order)
    conf = np.zeros((n_g, n_g), dtype=int)
    g2i  = {g: i for i, g in enumerate(groups_order)}
    for tg, pg in zip(true_info["groups"], pred_info["groups"]):
        conf[g2i[tg], g2i[pg]] += 1

    # ── Figure 1: coherence pie ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie([s for _, s, _ in active],
           labels=[f"{l}\n({s})" for l, s, _ in active],
           colors=[c for _, _, c in active], autopct="%1.0f%%", startangle=140,
           textprops={"fontsize": 10})
    ax.set_title("Error coherence with Köppen zones", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "koppen_coherence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved {path}")

    # ── Figure 2: confusion matrix (counts + row-normalised %) ───────────────
    row_sums = conf.sum(axis=1, keepdims=True).clip(min=1)
    conf_pct = conf / row_sums * 100          # row-normalised: recall per true zone

    group_labels = [f"{g}\n{KOPPEN_GROUPS.get(g, 'Ocean')}" for g in groups_order]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title, fmt in zip(
        axes,
        [conf, conf_pct],
        ["Counts", "Row-normalised (recall %)"],
        ["{:.0f}", "{:.1f}%"],
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks(range(n_g)); ax.set_xticklabels(group_labels, fontsize=8)
        ax.set_yticks(range(n_g)); ax.set_yticklabels(group_labels, fontsize=8)
        ax.set_xlabel("Predicted group", fontsize=9)
        ax.set_ylabel("True group", fontsize=9)
        ax.set_title(title, fontsize=10)
        threshold = data.max() * 0.5
        for r in range(n_g):
            for c in range(n_g):
                if data[r, c] > 0:
                    ax.text(c, r, fmt.format(data[r, c]), ha="center", va="center",
                            fontsize=8,
                            color="white" if data[r, c] > threshold else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Köppen climate group confusion matrix", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "koppen_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved {path}")

    n_coh = coherence.count("exact") + coherence.count("adjacent")
    n_tot = len(coherence)
    print(f"Climate-coherent: {n_coh}/{n_tot} ({100*n_coh/max(n_tot,1):.0f}%)")
    print(f"Spurious (distant): {coherence.count('distant')}/{n_tot}")
    return {
        "n_total":   n_tot,
        "exact":     coherence.count("exact"),
        "adjacent":  coherence.count("adjacent"),
        "distant":   coherence.count("distant"),
        "ocean":     coherence.count("ocean"),
        "coherent_pct": 100 * n_coh / max(n_tot, 1),
        "confusion": {
            groups_order[r]: {groups_order[c]: int(conf[r, c]) for c in range(n_g)}
            for r in range(n_g)
        },
    }


def run_error_dist(model, val_loader, gallery_coords, gallery_embs, thresholds_km, out_dir, device):
    """Section 17 — histogram + CDF of prediction errors."""
    model.eval()
    all_dists = []
    for imgs, coords in tqdm(val_loader, desc="Error dist"):
        with torch.no_grad():
            e = model.encode_image(imgs.to(device))
            idx = (e @ gallery_embs.to(device).T).argmax(dim=-1).cpu()
        all_dists.append(haversine_distance(gallery_coords[idx], coords))
    all_dists = torch.cat(all_dists)

    bins = np.logspace(0, np.log10(20_000), 60)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.hist(all_dists.numpy(), bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
    ax1.set_xscale("log")
    ax1.set_xlabel("Error (km, log scale)"); ax1.set_ylabel("Count")
    ax1.set_title("Prediction error distribution")
    ax1.grid(axis="y", alpha=0.3)
    for t in thresholds_km:
        ax1.axvline(t, color="red", linewidth=1, linestyle="--", alpha=0.7)
        ax1.text(t * 1.05, ax1.get_ylim()[1] * 0.92, f"{t} km",
                 color="red", fontsize=7, va="top")
    ax1.axvline(all_dists.median().item(), color="orange", linewidth=1.5,
                linestyle="-.", label=f"Median: {all_dists.median():.0f} km")
    ax1.legend(fontsize=8)

    sorted_d = torch.sort(all_dists).values.numpy()
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d) * 100
    ax2.plot(sorted_d, cdf, linewidth=2, color="steelblue")
    ax2.set_xscale("log")
    ax2.set_xlabel("Error (km, log scale)"); ax2.set_ylabel("Cumulative %")
    ax2.set_title("Cumulative error distribution (CDF)")
    ax2.set_ylim(0, 105); ax2.grid(alpha=0.3)
    for t in thresholds_km:
        recall_pct = (all_dists <= t).float().mean().item() * 100
        ax2.axvline(t, color="red", linewidth=1, linestyle="--", alpha=0.7)
        ax2.scatter([t], [recall_pct], color="red", s=40, zorder=5)
        ax2.text(t * 1.08, recall_pct + 1, f"{recall_pct:.0f}%", color="red", fontsize=7)

    plt.suptitle(f"Error over {len(all_dists)} validation images", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "error_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved {path}")
    print(f"Mean: {all_dists.mean():.0f} km | Median: {all_dists.median():.0f} km "
          f"| 90th p: {torch.quantile(all_dists, 0.9):.0f} km")
    return {
        "n_samples":  len(all_dists),
        "mean_km":    all_dists.mean().item(),
        "median_km":  all_dists.median().item(),
        "p10_km":     torch.quantile(all_dists, 0.10).item(),
        "p90_km":     torch.quantile(all_dists, 0.90).item(),
        "p95_km":     torch.quantile(all_dists, 0.95).item(),
        **{f"recall@{t}km": (all_dists <= t).float().mean().item()
           for t in thresholds_km},
    }


def run_tsne(model, val_loader, out_dir, device, n_samples=512, classifier=None):
    """Section 18 — t-SNE of image and GPS embeddings."""
    from sklearn.manifold import TSNE

    img_list, gps_list, coords_list = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, coords in val_loader:
            img_list.append(model.encode_image(imgs.to(device)).cpu())
            gps_list.append(model.encode_gps(coords.to(device)).cpu())
            coords_list.append(coords)
            if sum(len(x) for x in img_list) >= n_samples:
                break

    img_embs = torch.cat(img_list)[:n_samples]
    gps_embs = torch.cat(gps_list)[:n_samples]
    coords_t = torch.cat(coords_list)[:n_samples].numpy()

    all_embs = torch.cat([img_embs, gps_embs]).numpy()
    print(f"[Visualize] Running t-SNE on {len(all_embs)} embeddings …")
    proj = TSNE(n_components=2, perplexity=40, random_state=0,
                max_iter=1000).fit_transform(all_embs)
    img_proj = proj[:n_samples]
    gps_proj = proj[n_samples:]

    kc = classifier or KoppenClassifier(require_raster=False)
    info   = kc.classify_batch(coords_t[:, 0], coords_t[:, 1])
    groups = info["groups"]
    colors = [GROUP_COLORS.get(g, "#aaaaaa") for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, p2d, title, marker in zip(
        axes, [img_proj, gps_proj],
        ["Image embeddings", "GPS embeddings"], ["o", "^"]
    ):
        ax.scatter(p2d[:, 0], p2d[:, 1], c=colors, s=18,
                   marker=marker, alpha=0.75, linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    patches = [mpatches.Patch(color=GROUP_COLORS[g], label=f"{g} — {KOPPEN_GROUPS[g]}")
               for g in ["A", "B", "C", "D", "E"] if g in GROUP_COLORS]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9,
               title="Köppen group (colour = true location)")
    plt.suptitle(f"t-SNE of {n_samples} image & GPS embeddings", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(out_dir, "tsne_embeddings.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved {path}")


def run_calibration(model, val_loader, gallery_coords, gallery_embs, out_dir, device):
    """Section 20 — similarity–error calibration."""
    model.eval()
    cal_sims, cal_errs = [], []
    for imgs, coords in tqdm(val_loader, desc="Calibration"):
        with torch.no_grad():
            e = model.encode_image(imgs.to(device))
            s = (e @ gallery_embs.to(device).T).cpu()
        top_sim, top_idx = s.max(dim=-1)
        cal_sims.append(top_sim)
        cal_errs.append(haversine_distance(gallery_coords[top_idx], coords))

    cal_sims = torch.cat(cal_sims).numpy()
    cal_errs = torch.cat(cal_errs).numpy()
    r = np.corrcoef(cal_sims, cal_errs)[0, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    hb = ax1.hexbin(cal_sims, cal_errs, gridsize=40, yscale="log",
                    cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax1, label="Count")
    m, b = np.polyfit(cal_sims, np.log10(cal_errs + 1), 1)
    xs = np.linspace(cal_sims.min(), cal_sims.max(), 100)
    ax1.plot(xs, 10 ** (m * xs + b) - 1, color="steelblue", linewidth=2, label="Trend")
    ax1.set_xlabel("Max cosine similarity (confidence)")
    ax1.set_ylabel("Prediction error (km, log scale)")
    ax1.set_title(f"Confidence vs. error  |  r = {r:.3f}")
    ax1.legend()

    bins    = np.percentile(cal_sims, np.linspace(0, 100, 11))
    bin_idx = np.digitize(cal_sims, bins[1:-1])
    med_err = [np.median(cal_errs[bin_idx == i]) for i in range(10)]
    pct_mid = [(bins[i] + bins[i + 1]) / 2 for i in range(10)]
    ax2.bar(range(10), med_err, color=plt.cm.RdYlGn(np.linspace(0, 1, 10)),
            edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(10))
    ax2.set_xticklabels([f"{p:.2f}" for p in pct_mid], rotation=30, fontsize=7)
    ax2.set_xlabel("Mean similarity in decile")
    ax2.set_ylabel("Median error (km)")
    ax2.set_title("Median error per confidence decile")
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Similarity–error calibration", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "calibration.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved {path}")
    print(f"Pearson r (sim, error) = {r:.3f}")
    return {
        "pearson_r":    r,
        "sim_range":    (float(cal_sims.min()), float(cal_sims.max())),
        "per_decile":   list(zip([float(p) for p in pct_mid], [float(e) for e in med_err])),
    }


def run_zone_perf(model, val_loader, gallery_coords, thresholds_km, out_dir, device,
                  climate_codes=None):
    """Section 21 — per-Köppen-zone performance."""
    kc = KoppenClassifier(require_raster=(climate_codes is None))
    zone_metrics = evaluate_by_zone(
        model, val_loader, gallery_coords, kc,
        device=device, thresholds_km=thresholds_km,
        climate_codes=climate_codes,
    )

    groups_sorted = sorted(zone_metrics, key=lambda g: -zone_metrics[g]["count"])
    t1, t2 = (thresholds_km[2] if len(thresholds_km) > 2 else 200,
              thresholds_km[-1] if thresholds_km else 2500)
    g_labels   = [f"{g}\n({KOPPEN_GROUPS.get(g,'?')})" for g in groups_sorted]
    r1         = [zone_metrics[g].get(f"recall@{t1}km", 0) * 100 for g in groups_sorted]
    r2         = [zone_metrics[g].get(f"recall@{t2}km", 0) * 100 for g in groups_sorted]
    counts     = [zone_metrics[g]["count"] for g in groups_sorted]
    med_gcds   = [zone_metrics[g]["median_gcd_km"] for g in groups_sorted]
    grp_colors = [GROUP_COLORS.get(g, "#aaaaaa") for g in groups_sorted]

    x, w = np.arange(len(groups_sorted)), 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bars1 = ax1.bar(x - w/2, r1, w, label=f"Recall@{t1}km",
                    color=grp_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    bars2 = ax1.bar(x + w/2, r2, w, label=f"Recall@{t2}km",
                    color=grp_colors, alpha=0.45, edgecolor="black", linewidth=0.5, hatch="//")
    for bar, v in zip(list(bars1) + list(bars2), r1 + r2):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.8, f"{v:.0f}%",
                 ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(x); ax1.set_xticklabels(g_labels, fontsize=8)
    ax1.set_ylabel("Recall (%)"); ax1.set_ylim(0, 115)
    ax1.set_title("Recall@km by Köppen zone")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    for i, n in enumerate(counts):
        ax1.text(i, -7, f"n={n}", ha="center", fontsize=7, color="grey")

    bars_gcd = ax2.bar(x, med_gcds, color=grp_colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars_gcd, med_gcds):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 10, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(g_labels, fontsize=8)
    ax2.set_ylabel("Median GCD (km)")
    ax2.set_title("Median error by Köppen zone")
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Per-climate-zone performance", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "zone_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved {path}")
    return {"zones": zone_metrics}


def run_training_curves(checkpoint_dir, out_dir):
    """Section 16 — loss, GCD, and recall over epochs."""
    import glob

    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))
    if not ckpt_files:
        print(f"[Visualize] training_curves: no epoch_*.pt files found in {checkpoint_dir}")
        return

    history = []
    for path in ckpt_files:
        ckpt = torch.load(path, map_location="cpu")
        row  = {"epoch": ckpt["epoch"] + 1}
        if "train_loss" in ckpt:
            row["train_loss"] = ckpt["train_loss"]
        row.update(ckpt.get("metrics", {}))
        history.append(row)

    epochs     = [r["epoch"] for r in history]
    mean_gcd   = [r.get("mean_gcd_km",   float("nan")) for r in history]
    median_gcd = [r.get("median_gcd_km", float("nan")) for r in history]
    train_losses = [r.get("train_loss", None) for r in history]
    thresholds = sorted(
        [k.replace("recall@", "").replace("km", "") for k in history[0] if k.startswith("recall")],
        key=int,
    )
    recalls = {t: [r.get(f"recall@{t}km", float("nan")) * 100 for r in history]
               for t in thresholds}

    has_loss = any(v is not None for v in train_losses)
    n_plots  = 4 if has_loss else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4))
    ax_idx = 0

    if has_loss:
        valid = [(e, l) for e, l in zip(epochs, train_losses) if l is not None]
        axes[ax_idx].plot(*zip(*valid), marker="o", linewidth=2, color="tomato")
        axes[ax_idx].set_xlabel("Epoch"); axes[ax_idx].set_ylabel("Loss")
        axes[ax_idx].set_title("Training loss"); axes[ax_idx].grid(alpha=0.3)
        ax_idx += 1

    axes[ax_idx].plot(epochs, mean_gcd,   label="Mean GCD",   marker="o", linewidth=2)
    axes[ax_idx].plot(epochs, median_gcd, label="Median GCD", marker="s", linewidth=2, linestyle="--")
    best_epoch = epochs[median_gcd.index(min(median_gcd))]
    axes[ax_idx].axvline(best_epoch, color="red", linestyle=":", linewidth=1,
                         label=f"Best (ep {best_epoch})")
    axes[ax_idx].set_xlabel("Epoch"); axes[ax_idx].set_ylabel("km")
    axes[ax_idx].set_title("GCD over training")
    axes[ax_idx].legend(fontsize=8); axes[ax_idx].grid(alpha=0.3)
    ax_idx += 1

    cmap = plt.cm.plasma
    for j, t in enumerate(thresholds):
        axes[ax_idx].plot(epochs, recalls[t], label=f"@{t} km", marker="o",
                          linewidth=2, color=cmap(j / max(len(thresholds) - 1, 1)))
    axes[ax_idx].set_xlabel("Epoch"); axes[ax_idx].set_ylabel("Recall (%)")
    axes[ax_idx].set_ylim(0, 105); axes[ax_idx].set_title("Recall@km over training")
    axes[ax_idx].legend(fontsize=7); axes[ax_idx].grid(alpha=0.3)
    ax_idx += 1

    final      = history[-1]
    bar_labels = [f"@{t} km" for t in thresholds]
    bar_vals   = [final.get(f"recall@{t}km", 0) * 100 for t in thresholds]
    bar_colors = [cmap(j / max(len(thresholds) - 1, 1)) for j in range(len(thresholds))]
    bars = axes[ax_idx].bar(bar_labels, bar_vals, color=bar_colors, width=0.6,
                             edgecolor="black", linewidth=0.5)
    axes[ax_idx].set_ylabel("Recall (%)"); axes[ax_idx].set_ylim(0, 110)
    axes[ax_idx].set_title(f"Final recall — epoch {final['epoch']}\n"
                            f"Median GCD: {final.get('median_gcd_km',0):.0f} km")
    axes[ax_idx].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, bar_vals):
        axes[ax_idx].text(bar.get_x() + bar.get_width()/2, v + 1.5,
                          f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Training history", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved {path}")
    print(f"Best median GCD: {min(median_gcd):.0f} km  (epoch {best_epoch})")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_report(out_dir: str, args, stats: dict) -> None:
    """Write a markdown summary report of all computed stats."""
    lines = []
    add = lines.append

    add("# GeoCLIP Evaluation Report\n")
    add(f"**Checkpoint**: `{args.checkpoint}`  ")
    add(f"**Eval set**: `{args.eval_csv}`  ")
    add(f"**Eval subset**: {args.eval_subset:,} samples\n")

    if "error_dist" in stats:
        s = stats["error_dist"]
        add("## Overall Error\n")
        add(f"| Metric | Value |")
        add(f"|--------|-------|")
        add(f"| Samples evaluated | {s['n_samples']:,} |")
        add(f"| Mean GCD | {s['mean_km']:.0f} km |")
        add(f"| Median GCD | {s['median_km']:.0f} km |")
        add(f"| 10th percentile | {s['p10_km']:.0f} km |")
        add(f"| 90th percentile | {s['p90_km']:.0f} km |")
        add(f"| 95th percentile | {s['p95_km']:.0f} km |")
        add("")
        add("### Recall @ thresholds\n")
        add("| Threshold | Recall |")
        add("|-----------|--------|")
        for k, v in s.items():
            if k.startswith("recall@"):
                add(f"| {k.replace('recall@', '')} | {v*100:.2f}% |")
        add("")

    if "calibration" in stats:
        s = stats["calibration"]
        add("## Calibration\n")
        add(f"**Pearson r** (confidence vs error): **{s['pearson_r']:.3f}**  ")
        add(f"Similarity range: [{s['sim_range'][0]:.3f}, {s['sim_range'][1]:.3f}]\n")
        add("### Median error per confidence decile\n")
        add("| Decile mid-sim | Median error (km) |")
        add("|----------------|-------------------|")
        for sim, err in s["per_decile"]:
            add(f"| {sim:.3f} | {err:.0f} |")
        add("")

    if "koppen" in stats:
        s = stats["koppen"]
        n = s["n_total"]
        add("## Köppen Climate Coherence\n")
        add(f"| Category | Count | % |")
        add(f"|----------|-------|---|")
        for cat in ["exact", "adjacent", "distant", "ocean"]:
            cnt = s[cat]
            add(f"| {cat.capitalize()} | {cnt} | {100*cnt/max(n,1):.1f}% |")
        add(f"\n**Climate-coherent** (exact + adjacent): "
            f"**{s['coherent_pct']:.1f}%**\n")
        add("### Confusion matrix (true → predicted major group)\n")
        groups = [g for g in s["confusion"] if g != "?"]
        add("| True \\ Pred | " + " | ".join(groups + ["?"]) + " |")
        add("|" + "---|" * (len(groups) + 2))
        for tg in groups + ["?"]:
            row = s["confusion"].get(tg, {})
            vals = " | ".join(str(row.get(pg, 0)) for pg in groups + ["?"])
            add(f"| **{tg}** | {vals} |")
        add("")

    if "zone_perf" in stats:
        s = stats["zone_perf"]["zones"]
        add("## Per-Zone Performance\n")
        sorted_zones = sorted(s, key=lambda g: -s[g]["count"])
        recall_keys = sorted(
            [k for k in next(iter(s.values())) if k.startswith("recall@")],
            key=lambda k: int(k.replace("recall@", "").replace("km", ""))
        )
        header = "| Zone | Name | n | Median GCD | " + " | ".join(recall_keys) + " |"
        sep    = "|------|------|---|------------|" + "---|" * len(recall_keys)
        add(header); add(sep)
        for g in sorted_zones:
            m = s[g]
            recalls = " | ".join(f"{m.get(k, 0)*100:.1f}%" for k in recall_keys)
            add(f"| {g} | {KOPPEN_GROUPS.get(g, 'Ocean/Unk')} | {m['count']} "
                f"| {m['median_gcd_km']:.0f} km | {recalls} |")
        add("")

    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Visualize] Report saved to {path}")


# ---------------------------------------------------------------------------
# Contamination check
# ---------------------------------------------------------------------------

def run_contamination_check(
    model, test_images, test_coords, test_img_nps, test_distances,
    train_zip_dir, train_csv, out_dir, device,
    n_train_samples: int = 10_000, top_k: int = 3,
):
    """
    For each good test image find the most visually similar training images.

    High cosine similarity + large GPS gap  → model learned real features.
    High cosine similarity + tiny GPS gap   → possible data contamination
                                              (near-duplicate in both splits).
    """
    from torch.utils.data import Subset, DataLoader as DL

    # ── Sample & encode training images ──────────────────────────────────────
    train_ds = LocalZipOSV5MDataset(train_zip_dir, train_csv,
                                    transform=get_eval_transform())
    n = min(n_train_samples, len(train_ds))
    idx = random.sample(range(len(train_ds)), n)
    train_subset = Subset(train_ds, idx)
    loader = DL(train_subset, batch_size=64, shuffle=False, num_workers=0)

    train_embs, train_coords_all = [], []
    model.eval()
    with torch.no_grad():
        for imgs, coords in tqdm(loader, desc="Encoding train sample", leave=False):
            train_embs.append(model.encode_image(imgs.to(device)).cpu())
            train_coords_all.append(coords)
    train_embs   = torch.cat(train_embs)        # [N, D]
    train_coords = torch.cat(train_coords_all)  # [N, 2]

    # ── Query each test image ─────────────────────────────────────────────────
    with torch.no_grad():
        test_embs = model.encode_image(test_images.to(device)).cpu()  # [B, D]

    sims     = test_embs @ train_embs.T                   # [B, N]
    top_vals, top_idx = sims.topk(top_k, dim=-1)          # [B, top_k]

    # ── Load train images for display ────────────────────────────────────────
    # Collect all unique train indices needed
    flat_idx = top_idx.flatten().tolist()
    train_imgs_needed = {}
    for ti in set(flat_idx):
        img_t, coord_t = train_ds[idx[ti]]
        train_imgs_needed[ti] = (denormalize(img_t), coord_t)

    # ── Build figure ─────────────────────────────────────────────────────────
    B = len(test_images)
    col_titles = ["Test image (good)"] + [f"Train match #{k+1}" for k in range(top_k)]
    rows = []
    contamination_flags = []

    for b in range(B):
        tc = test_coords[b].numpy()
        row = [test_img_nps[b]]
        flagged = False
        for k in range(top_k):
            ti       = top_idx[b, k].item()
            sim      = top_vals[b, k].item()
            tr_img, tr_coord = train_imgs_needed[ti]
            gps_dist = haversine_distance(
                torch.tensor(tc).unsqueeze(0),
                tr_coord.unsqueeze(0),
            ).item()
            # Annotate: red border hint if very similar AND close (<50 km)
            flagged = flagged or (sim > 0.90 and gps_dist < 50)
            # Overlay similarity + distance as text on the train image
            ann = tr_img.copy()
            import cv2 as _cv2
            label = f"sim={sim:.3f}  {gps_dist:.0f}km"
            _cv2.putText(ann, label, (4, 18), _cv2.FONT_HERSHEY_SIMPLEX,
                         0.45, (255, 255, 0), 1, _cv2.LINE_AA)
            row.append(ann)
        rows.append(row)
        contamination_flags.append(flagged)

    row_labels = [
        f"{'⚠ ' if contamination_flags[b] else ''}"
        f"{test_distances[b]:.0f} km err"
        for b in range(B)
    ]
    save_grid(rows, col_titles,
              os.path.join(out_dir, "contamination_check.png"),
              row_labels=row_labels,
              suptitle=f"Contamination check — top-{top_k} train matches "
                       f"(⚠ = sim>0.90 and GPS<50 km)")

    n_flagged = sum(contamination_flags)
    print(f"[Contamination] Flagged {n_flagged}/{B} test images "
          f"(sim>0.90 and GPS distance <50 km to nearest train image)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate all GeoCLIP visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (required for most methods)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--hf_home", default=None,
                        help="HuggingFace cache directory (overrides HF_HOME env var)")
    parser.add_argument("--methods", nargs="+", default=["gradcam", "ig", "worldmap"],
                        metavar="METHOD",
                        help=f"One or more of: {', '.join(ALL_METHODS)}, or 'all'")
    # Batch options
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of images for batch-level methods")
    parser.add_argument("--ig_steps", type=int, default=50)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    # Eval dataset options
    parser.add_argument("--eval_split", default="val")
    parser.add_argument("--eval_subset", type=int, default=2000,
                        help="Validation subset size for full-set analyses")
    # Training curves
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Directory with epoch_*.pt files (for training_curves)")
    # Local dataset
    parser.add_argument("--zip_dir", default=None,
                        help="Local zip directory (overrides HF dataset loading)")
    parser.add_argument("--eval_csv", default=None,
                        help="CSV metadata file for local zip dataset")
    parser.add_argument("--train_csv", default=None,
                        help="Optional: denser reference CSV for Köppen KD-tree (defaults to eval_csv)")
    parser.add_argument("--train_zip_dir", default=None,
                        help="[contamination] Directory of train zip files")
    parser.add_argument("--n_train_samples", type=int, default=10_000,
                        help="[contamination] Number of train images to sample for the index")
    # Output
    parser.add_argument("--output_dir", default="outputs/visualizations/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hf_home = args.hf_home or os.environ.get("HF_HOME")
    if hf_home:
        os.makedirs(hf_home, exist_ok=True)
        os.environ["HF_HOME"] = hf_home

    methods = (BATCH_METHODS | FULLSET_METHODS) if "all" in args.methods else set(args.methods)
    device  = args.device

    # Training curves needs no model
    if "training_curves" in methods:
        ckpt_dir = args.checkpoint_dir or (
            os.path.dirname(args.checkpoint) if args.checkpoint else None
        )
        if not ckpt_dir:
            print("[Visualize] training_curves: provide --checkpoint_dir")
        else:
            run_training_curves(ckpt_dir, args.output_dir)
        methods.discard("training_curves")

    if not methods:
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required for all methods except training_curves")

    cfg = load_config(args.config)

    model = GeoCLIP(
        clip_model_name=cfg.model.clip_backbone,
        freeze_layers=cfg.model.freeze_layers,
        rff_num_scales=cfg.model.rff_num_scales,
        rff_dim=cfg.model.rff_dim,
        mlp_hidden=cfg.model.mlp_hidden,
        embedding_dim=cfg.model.embedding_dim,
    )
    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device).eval()

    # Dataset + gallery (shared by all remaining methods)
    if args.zip_dir and args.eval_csv:
        dataset = LocalZipOSV5MDataset(
            zip_dir=args.zip_dir,
            csv_path=args.eval_csv,
            transform=get_eval_transform(),
        )
        if len(dataset) > max(args.eval_subset, args.num_samples * 4):
            from torch.utils.data import Subset
            import random
            dataset = Subset(dataset, random.sample(range(len(dataset)),
                                                    max(args.eval_subset, args.num_samples * 4)))
    else:
        dataset = OSV5MDataset(
            split=args.eval_split,
            subset_size=max(args.eval_subset, args.num_samples * 4),
            transform=get_eval_transform(),
            cache_dir=hf_home,
        )
    gallery_coords = load_or_build_gallery(
        strategy=cfg.gallery.strategy,
        size=cfg.gallery.size,
        cache_path=cfg.gallery.cache_path,
        dataset=dataset,
    )
    gallery_embs = compute_gallery_embeddings(model, gallery_coords, device=device)

    # ── Full-set analyses ──────────────────────────────────────────────────────
    fullset_requested = methods & FULLSET_METHODS
    if fullset_requested:
        full_loader = DataLoader(
            dataset, batch_size=64, shuffle=False,
            num_workers=cfg.data.num_workers,
        )
        # Unwrap Subset to reach climate_codes; filter by subset indices if needed
        _base = getattr(dataset, "dataset", dataset)
        _codes = getattr(_base, "climate_codes", None)
        if _codes is not None and hasattr(dataset, "indices"):
            _codes = [_codes[i] for i in dataset.indices]
        climate_codes = _codes

        # Build a KD-tree classifier so predicted coords can be classified
        # without the raster. Prefer train_csv (denser global coverage) but
        # fall back to eval_csv — its full split covers the globe well enough.
        _kc_for_pred = None
        _ref_csv = (args.train_csv if args.train_csv and os.path.exists(args.train_csv)
                    else args.eval_csv if args.eval_csv and os.path.exists(args.eval_csv)
                    else None)
        if _ref_csv:
            _kc_for_pred = KoppenClassifier.from_csv(_ref_csv)

        def _timed_full(label, fn, *a, **kw):
            print(f"\n[Visualize] ── {label.upper()} ──", flush=True)
            t0 = time.time()
            result = fn(*a, **kw)
            print(f"[Visualize] {label} done ({time.time()-t0:.1f}s)", flush=True)
            return result

        report_stats = {}

        if "error_dist"  in fullset_requested:
            report_stats["error_dist"] = _timed_full(
                "error_dist", run_error_dist, model, full_loader, gallery_coords,
                gallery_embs, cfg.evaluation.thresholds_km, args.output_dir, device)

        if "calibration" in fullset_requested:
            report_stats["calibration"] = _timed_full(
                "calibration", run_calibration, model, full_loader, gallery_coords,
                gallery_embs, args.output_dir, device)

        if "koppen"      in fullset_requested:
            report_stats["koppen"] = _timed_full(
                "koppen", run_koppen, model, full_loader, gallery_coords,
                gallery_embs, args.output_dir, device, climate_codes=climate_codes,
                pred_classifier=_kc_for_pred)

        if "zone_perf"   in fullset_requested:
            report_stats["zone_perf"] = _timed_full(
                "zone_perf", run_zone_perf, model, full_loader, gallery_coords,
                cfg.evaluation.thresholds_km, args.output_dir, device,
                climate_codes=climate_codes)

        if "tsne"        in fullset_requested:
            _timed_full("tsne", run_tsne, model, full_loader, args.output_dir, device,
                        n_samples=min(512, args.eval_subset), classifier=_kc_for_pred)

        if report_stats:
            write_report(args.output_dir, args, report_stats)

    # ── Batch-level + train-requiring methods ─────────────────────────────────
    batch_requested = methods & BATCH_METHODS
    train_requested = methods & TRAIN_METHODS
    if not batch_requested and not train_requested:
        return

    # Sample a large pool so stratification has enough candidates in each group
    n = args.num_samples
    pool_size = max(n * 30, 300)
    torch.manual_seed(args.seed)
    pool_loader = DataLoader(dataset, batch_size=pool_size, shuffle=True)
    pool_images, pool_coords = next(iter(pool_loader))

    model.eval()
    with torch.no_grad():
        pool_embs = model.encode_image(pool_images.to(device))
        pool_idx  = (pool_embs @ gallery_embs.to(device).T).argmax(dim=-1).cpu()
    pool_pred = gallery_coords[pool_idx]
    pool_dist = haversine_distance(pool_pred, pool_coords)

    # Stratify: good < 200 km | mixed 200–2500 km | bad > 2500 km
    def pick(mask, count):
        idx = mask.nonzero().squeeze(1).tolist()
        return idx[:count]

    groups = {
        "good":  pick(pool_dist <  200,                          n),
        "mixed": pick((pool_dist >= 200) & (pool_dist <= 2500),  n),
        "bad":   pick(pool_dist >  2500,                         n),
    }
    print("[Visualize] Group sizes: " +
          " | ".join(f"{k}={len(v)}" for k, v in groups.items()))

    def _timed(label, fn, *a, **kw):
        print(f"  → {label} ...", flush=True)
        t0 = time.time()
        fn(*a, **kw)
        print(f"  ✓ {label} done ({time.time()-t0:.1f}s)", flush=True)

    def run_batch_methods(images, true_coords, pred_coords, distances, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        img_nps = [denormalize(images[i]) for i in range(len(images))]
        if "gradcam"     in batch_requested:
            _timed("gradcam",     run_gradcam, model, images, true_coords, img_nps, out_dir)
        if "rollout"     in batch_requested:
            _timed("rollout",     run_rollout, model, images, img_nps, out_dir)
        if "ig"          in batch_requested:
            _timed("ig",          run_ig, model, images, true_coords, img_nps, out_dir, args.ig_steps)
        if "layerwise"   in batch_requested:
            _timed("layerwise",   run_layerwise, model, images, true_coords, img_nps, out_dir)
        if "heads"       in batch_requested:
            _timed("heads",       run_heads, model, images, img_nps, out_dir)
        if "worldmap"    in batch_requested:
            _timed("worldmap",    run_worldmap, model, images, gallery_coords, gallery_embs,
                                  true_coords, pred_coords, distances, out_dir)
        if "all_methods" in batch_requested:
            _timed("all_methods", run_all_methods, model, images, true_coords, img_nps,
                                  gallery_embs, gallery_coords, distances, out_dir, args.ig_steps)
        if "topk"        in batch_requested:
            _timed("topk",        run_topk, model, images, true_coords, img_nps, gallery_coords,
                                  gallery_embs, distances, out_dir, args.topk)

    for group_name, indices in groups.items():
        if not indices:
            print(f"[Visualize] No samples for group '{group_name}', skipping")
            continue
        g_images = pool_images[indices]
        g_coords = pool_coords[indices]
        g_pred   = pool_pred[indices]
        g_dist   = pool_dist[indices]
        out_dir  = os.path.join(args.output_dir, group_name)
        print(f"\n[Visualize] ── {group_name.upper()} "
              f"(errors: {g_dist.min():.0f}–{g_dist.max():.0f} km) ──")
        run_batch_methods(g_images, g_coords, g_pred, g_dist, out_dir)

    # good_vs_bad: compare good and bad groups side by side
    if "good_vs_bad" in batch_requested and groups["good"] and groups["bad"]:
        run_good_vs_bad(
            model,
            pool_images[groups["good"]], pool_coords[groups["good"]], pool_dist[groups["good"]],
            pool_images[groups["bad"]],  pool_coords[groups["bad"]],  pool_dist[groups["bad"]],
            args.output_dir, args.ig_steps,
        )

    # Contamination check: run only on the "good" group
    if "contamination" in methods and groups["good"]:
        if not args.train_zip_dir or not args.train_csv:
            print("[Visualize] contamination: requires --train_zip_dir and --train_csv")
        else:
            good_idx   = groups["good"]
            good_imgs  = pool_images[good_idx]
            good_nps   = [denormalize(good_imgs[i]) for i in range(len(good_idx))]
            print(f"\n[Visualize] ── CONTAMINATION CHECK ──", flush=True)
            t0 = time.time()
            run_contamination_check(
                model, good_imgs, pool_coords[good_idx],
                good_nps, pool_dist[good_idx],
                args.train_zip_dir, args.train_csv,
                args.output_dir, device,
                n_train_samples=args.n_train_samples,
            )
            print(f"[Visualize] contamination done ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
