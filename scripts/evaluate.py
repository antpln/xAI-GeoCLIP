"""
Standalone evaluation script.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/default.yaml
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from geoclip.models.geoclip import GeoCLIP
from geoclip.data.dataset import OSV5MDataset
from geoclip.data.transforms import get_eval_transform
from geoclip.data.gallery import load_or_build_gallery
from geoclip.training.evaluator import evaluate
from geoclip.utils.checkpoint import load_checkpoint
from geoclip.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate GeoCLIP")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset_size", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device

    dataset = OSV5MDataset(
        split=args.split,
        subset_size=args.subset_size,
        transform=get_eval_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    gallery_coords = load_or_build_gallery(
        strategy=cfg.gallery.strategy,
        size=cfg.gallery.size,
        cache_path=cfg.gallery.cache_path,
        dataset=dataset,
    )

    model = GeoCLIP(
        clip_model_name=cfg.model.clip_backbone,
        freeze_layers=cfg.model.freeze_layers,
        rff_num_scales=cfg.model.rff_num_scales,
        rff_dim=cfg.model.rff_dim,
        mlp_hidden=cfg.model.mlp_hidden,
        embedding_dim=cfg.model.embedding_dim,
    )
    load_checkpoint(args.checkpoint, model, device=device)

    metrics = evaluate(
        model,
        loader,
        gallery_coords,
        device=device,
        thresholds_km=cfg.evaluation.thresholds_km,
    )

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
