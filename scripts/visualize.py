"""
Visualization script for GeoCLIP interpretability methods.

Methods:
    gradcam      — Grad-CAM on the last transformer block
    rollout      — Attention Rollout across all layers
    ig           — Integrated Gradients (pixel-level, completeness axiom)
    layerwise    — Grad-CAM at blocks 2, 5, 8, 11 (evolution across depth)
    heads        — Per-head attention at the last layer
    worldmap     — Geographic belief distribution over the gallery

Usage:
    python scripts/visualize.py --checkpoint checkpoints/best.pt \\
                                 --config configs/default.yaml \\
                                 --methods gradcam ig worldmap \\
                                 --num_samples 4 \\
                                 --output_dir outputs/
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader

from geoclip.models.geoclip import GeoCLIP
from geoclip.data.dataset import OSV5MDataset
from geoclip.data.transforms import get_eval_transform
from geoclip.data.gallery import load_or_build_gallery, compute_gallery_embeddings
from geoclip.interpretability.gradcam import gradcam_context, gradcam_layerwise
from geoclip.interpretability.attention_rollout import attention_rollout_context, PerHeadAttention
from geoclip.interpretability.integrated_gradients import IntegratedGradients
from geoclip.interpretability.world_map import plot_similarity_grid
from geoclip.utils.checkpoint import load_checkpoint
from geoclip.utils.config import load_config
from geoclip.utils.geo_math import haversine_distance

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

ALL_METHODS = ["gradcam", "rollout", "ig", "layerwise", "heads", "worldmap"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denormalize(t: torch.Tensor) -> np.ndarray:
    """[3, H, W] CLIP-normalized → [H, W, 3] uint8."""
    img = t.cpu() * CLIP_STD[:, None, None] + CLIP_MEAN[:, None, None]
    return (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay(img_np: np.ndarray, hmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a [0,1] heatmap onto a uint8 image."""
    colored = (cm.jet(hmap)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * colored + (1 - alpha) * img_np).astype(np.uint8)


def save_grid(rows_data: list, col_titles: list, output_path: str) -> None:
    """
    Save a (n_samples × n_cols) grid of images.

    rows_data: list of lists of (np.ndarray or None) — one inner list per sample.
    """
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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved {output_path}")


# ---------------------------------------------------------------------------
# Per-method rendering
# ---------------------------------------------------------------------------

def render_gradcam(model, images, true_coords, img_nps):
    with gradcam_context(model) as gc:
        maps = gc.compute(images, true_coords)
    return [[overlay(img_nps[i], maps[i].numpy())] for i in range(len(images))]


def render_rollout(model, images, img_nps):
    with attention_rollout_context(model, head_fusion="mean", discard_ratio=0.1) as ar:
        maps = ar.compute(images)
    return [[overlay(img_nps[i], maps[i].numpy())] for i in range(len(images))]


def render_ig(model, images, true_coords, img_nps, n_steps=50):
    ig = IntegratedGradients(model, n_steps=n_steps)
    maps = ig.compute(images, true_coords)
    return [[overlay(img_nps[i], maps[i].numpy())] for i in range(len(images))]


def render_layerwise(model, images, true_coords, img_nps, layer_indices=(2, 5, 8, 11)):
    results = gradcam_layerwise(model, images, true_coords, layer_indices=layer_indices)
    # Each sample gets one cell per layer — return as (samples × layers)
    rows = []
    for i in range(len(images)):
        row = [overlay(img_nps[i], results[idx][i].numpy()) for idx in layer_indices]
        rows.append(row)
    return rows, [f"Layer {idx}" for idx in layer_indices]


def render_heads(model, images, img_nps, layer_idx=-1):
    pha = PerHeadAttention(model, layer_idx=layer_idx)
    maps = pha.compute(images)          # [B, num_heads, H, W]
    num_heads = maps.shape[1]
    rows = []
    for i in range(len(images)):
        row = [overlay(img_nps[i], maps[i, h].numpy()) for h in range(num_heads)]
        rows.append(row)
    return rows, [f"Head {h}" for h in range(num_heads)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument(
        "--methods", nargs="+",
        default=["gradcam", "ig", "worldmap"],
        choices=ALL_METHODS,
    )
    parser.add_argument("--ig_steps", type=int, default=50)
    parser.add_argument("--output_dir", default="outputs/visualizations/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args.config)
    device = args.device

    # Load model
    model = GeoCLIP(
        clip_model_name=cfg.model.clip_backbone,
        freeze_layers=cfg.model.freeze_layers,
        rff_num_scales=cfg.model.rff_num_scales,
        rff_dim=cfg.model.rff_dim,
        mlp_hidden=cfg.model.mlp_hidden,
        embedding_dim=cfg.model.embedding_dim,
    )
    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)

    # Data
    dataset = OSV5MDataset(
        split="test",
        subset_size=args.num_samples * 2,
        transform=get_eval_transform(),
    )
    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True)
    images, true_coords = next(iter(loader))
    images, true_coords = images[:args.num_samples], true_coords[:args.num_samples]

    # Gallery
    gallery_coords = load_or_build_gallery(
        strategy=cfg.gallery.strategy,
        size=cfg.gallery.size,
        cache_path=cfg.gallery.cache_path,
        dataset=dataset,
    )
    gallery_embs = compute_gallery_embeddings(model, gallery_coords, device=device)

    # Predictions
    model.eval()
    with torch.no_grad():
        img_embs = model.encode_image(images.to(device))
        sims_all = img_embs @ gallery_embs.to(device).T
        best_idx = sims_all.argmax(dim=-1)
    pred_coords = gallery_coords[best_idx.cpu()]
    distances = haversine_distance(pred_coords, true_coords)

    img_nps = [denormalize(images[i]) for i in range(args.num_samples)]

    # ------------------------------------------------------------------
    # World map — separate figure, one panel per sample
    # ------------------------------------------------------------------
    if "worldmap" in args.methods:
        plot_similarity_grid(
            model=model,
            images=images,
            gallery_coords=gallery_coords,
            gallery_embs=gallery_embs,
            true_coords=true_coords,
            pred_coords=pred_coords,
            distances=distances,
            output_path=os.path.join(args.output_dir, "worldmap.png"),
        )

    # ------------------------------------------------------------------
    # Patch-level heatmap methods — all share the same grid layout
    # ------------------------------------------------------------------
    heatmap_methods = [m for m in args.methods if m != "worldmap"]

    for method in heatmap_methods:
        if method == "gradcam":
            rows = render_gradcam(model, images, true_coords, img_nps)
            col_titles = ["Input", "Grad-CAM"]
            rows = [[img_nps[i]] + rows[i] for i in range(len(rows))]

        elif method == "rollout":
            rows = render_rollout(model, images, img_nps)
            col_titles = ["Input", "Attention Rollout"]
            rows = [[img_nps[i]] + rows[i] for i in range(len(rows))]

        elif method == "ig":
            rows = render_ig(model, images, true_coords, img_nps, n_steps=args.ig_steps)
            col_titles = ["Input", "Integrated Gradients"]
            rows = [[img_nps[i]] + rows[i] for i in range(len(rows))]

        elif method == "layerwise":
            rows, layer_cols = render_layerwise(model, images, true_coords, img_nps)
            col_titles = ["Input"] + layer_cols
            rows = [[img_nps[i]] + rows[i] for i in range(len(rows))]

        elif method == "heads":
            rows, head_cols = render_heads(model, images, img_nps)
            col_titles = ["Input"] + head_cols
            rows = [[img_nps[i]] + rows[i] for i in range(len(rows))]

        # Annotate first column with distance info
        for i, row in enumerate(rows):
            dist_km = distances[i].item()
            # Replace first cell with annotated version (matplotlib title handles this)
            pass

        out = os.path.join(args.output_dir, f"{method}.png")
        save_grid(rows, col_titles, out)


if __name__ == "__main__":
    main()
