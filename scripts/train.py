"""
Main training entry point for GeoCLIP.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --device cuda
    python scripts/train.py --config configs/default.yaml --resume checkpoints/epoch_010.pt
    python scripts/train.py --config configs/local_partial.yaml
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from geoclip.models.geoclip import GeoCLIP
from geoclip.data.dataset import OSV5MDataset, LocalZipOSV5MDataset
from geoclip.data.shard_dataset import ShardedOSV5MDataset, StreamingOSV5MDataset
from geoclip.data.transforms import get_train_transform, get_eval_transform
from geoclip.data.gallery import load_or_build_gallery
from geoclip.training.trainer import Trainer
from geoclip.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train GeoCLIP")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--hf_home", default=None,
        help="Root directory for HuggingFace downloads (overrides HF_HOME env var).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device

    hf_home = args.hf_home or os.environ.get("HF_HOME")
    if hf_home:
        os.makedirs(hf_home, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        print(f"[Train] HF home: {hf_home}")

    print(f"[Train] Device: {device} | Dataset mode: {cfg.data.mode}")

    # ── Train dataset ──────────────────────────────────────────────────────────
    if cfg.data.mode == "local":
        if not cfg.data.zip_dir or not cfg.data.train_csv:
            raise ValueError("data.mode=local requires data.zip_dir and data.train_csv in config")

        train_dataset = LocalZipOSV5MDataset(
            zip_dir=cfg.data.zip_dir,
            csv_path=cfg.data.train_csv,
            transform=get_train_transform(),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.training.batch_size,
            shuffle=True, num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory, drop_last=True,
        )

    elif cfg.data.mode in ("hf", "subset"):
        train_dataset = OSV5MDataset(
            split="train",
            subset_size=cfg.data.subset_size,
            transform=get_train_transform(),
            cache_dir=hf_home,
            local_files_only=cfg.data.local_files_only,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.training.batch_size,
            shuffle=True, num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory, drop_last=True,
        )

    elif cfg.data.mode == "streaming":
        train_dataset = StreamingOSV5MDataset(
            split="train",
            num_shards=cfg.data.num_shards,
            transform=get_train_transform(),
            shuffle_buffer=4096,
            hf_home=hf_home,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.training.batch_size,
            shuffle=False, num_workers=cfg.data.num_workers,
            prefetch_factor=2 if cfg.data.num_workers > 0 else None,
            pin_memory=cfg.data.pin_memory,
        )

    else:  # sharded
        train_dataset = ShardedOSV5MDataset(
            split="train",
            num_shards=cfg.data.num_shards,
            transform=get_train_transform(),
            shards_per_step=1,
            hf_home=hf_home,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.training.batch_size,
            shuffle=True, num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory, drop_last=True,
        )

    # ── Validation dataset ────────────────────────────────────────────────────
    if cfg.data.mode == "local" and cfg.data.val_csv:
        val_zip_dir = cfg.data.val_zip_dir or cfg.data.zip_dir
        val_dataset = LocalZipOSV5MDataset(
            zip_dir=val_zip_dir,
            csv_path=cfg.data.val_csv,
            transform=get_eval_transform(),
        )
        if len(val_dataset) > 5000:
            indices = random.sample(range(len(val_dataset)), 5000)
            val_dataset = Subset(val_dataset, indices)
    elif cfg.data.mode == "local":
        # No separate val data — hold out 5 % of training samples
        n_val = min(5000, max(1, int(0.05 * len(train_dataset))))
        indices = random.sample(range(len(train_dataset)), n_val)
        val_dataset = Subset(train_dataset, indices)
        print(f"[Train] Using {n_val} held-out train samples as validation set")
    else:
        val_dataset = OSV5MDataset(
            split="val",
            subset_size=min(cfg.data.subset_size or 5000, 5000),
            transform=get_eval_transform(),
            cache_dir=hf_home,
            local_files_only=cfg.data.local_files_only,
        )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    # ── Gallery ────────────────────────────────────────────────────────────────
    gallery_coords = load_or_build_gallery(
        strategy=cfg.gallery.strategy,
        size=cfg.gallery.size,
        cache_path=cfg.gallery.cache_path,
        dataset=train_dataset,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = GeoCLIP(
        clip_model_name=cfg.model.clip_backbone,
        freeze_layers=cfg.model.freeze_layers,
        rff_num_scales=cfg.model.rff_num_scales,
        rff_dim=cfg.model.rff_dim,
        mlp_hidden=cfg.model.mlp_hidden,
        embedding_dim=cfg.model.embedding_dim,
    )

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
