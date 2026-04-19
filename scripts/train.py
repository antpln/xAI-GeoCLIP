"""
Main training entry point for GeoCLIP.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/small_experiment.yaml --device cuda
    python scripts/train.py --config configs/default.yaml --resume checkpoints/epoch_010.pt
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from geoclip.models.geoclip import GeoCLIP
from geoclip.data.dataset import OSV5MDataset
from geoclip.data.transforms import get_train_transform, get_eval_transform
from geoclip.data.gallery import load_or_build_gallery
from geoclip.training.trainer import Trainer
from geoclip.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train GeoCLIP")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device
    print(f"[Train] Using device: {device}")

    # Datasets
    train_dataset = OSV5MDataset(
        split="train",
        subset_size=cfg.data.subset_size,
        transform=get_train_transform(),
    )
    val_dataset = OSV5MDataset(
        split="test",
        subset_size=min(cfg.data.subset_size or 5000, 5000),
        transform=get_eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    # Gallery
    gallery_coords = load_or_build_gallery(
        strategy=cfg.gallery.strategy,
        size=cfg.gallery.size,
        cache_path=cfg.gallery.cache_path,
        dataset=train_dataset,
    )

    # Model
    model = GeoCLIP(
        clip_model_name=cfg.model.clip_backbone,
        freeze_layers=cfg.model.freeze_layers,
        rff_num_scales=cfg.model.rff_num_scales,
        rff_dim=cfg.model.rff_dim,
        mlp_hidden=cfg.model.mlp_hidden,
        embedding_dim=cfg.model.embedding_dim,
    )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        gallery_coords=gallery_coords,
        cfg=cfg,
        device=device,
        resume_path=args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
