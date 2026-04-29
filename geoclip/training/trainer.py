"""
Training loop for GeoCLIP.

Combines InfoNCE with hard geographic negative mining as the primary loss,
plus optional attention entropy regularization every few steps to encourage
focused, interpretable attention maps.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from geoclip.losses.infonce import info_nce_loss
from geoclip.losses.attention_entropy import attention_entropy_loss
from geoclip.training.hard_negatives import info_nce_with_hard_negatives
from geoclip.training.scheduler import get_cosine_schedule_with_warmup
from geoclip.training.evaluator import evaluate
from geoclip.utils.checkpoint import save_checkpoint, load_checkpoint, load_pretrained_geoclip_weights
from geoclip.utils.config import Config


class Trainer:
    """
    Manages the full training loop for GeoCLIP.

    Supports:
    - Mixed precision (AMP) training
    - Cosine LR schedule with linear warmup
    - Separate LR for ViT backbone vs GPS encoder vs temperature
    - Hard geographic negative mining (every step)
    - Attention entropy regularization (every attn_reg_every steps)
    - Gallery-based evaluation every N epochs
    - Checkpoint saving (latest + best)
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        gallery_coords: torch.Tensor,
        cfg: Config,
        device: str = "cuda",
        resume_path: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gallery_coords = gallery_coords
        self.cfg = cfg
        self.device = device

        if cfg.training.pretrained_weights_dir:
            load_pretrained_geoclip_weights(model, cfg.training.pretrained_weights_dir, device)

        self.optimizer = self._build_optimizer()
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * cfg.training.epochs
        warmup_steps = steps_per_epoch * cfg.training.warmup_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        self.scaler = GradScaler("cuda", enabled=cfg.training.amp)
        self.start_epoch = 0
        self.best_median_gcd = float("inf")

        if resume_path:
            ckpt = load_checkpoint(resume_path, model, self.optimizer, self.scheduler, device)
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_median_gcd = ckpt.get("best_median_gcd", float("inf"))

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.cfg.training
        param_groups = [
            {"params": self.model.image_encoder.parameters(), "lr": cfg.lr_clip,  "name": "vit"},
            {"params": self.model.gps_encoder.parameters(),   "lr": cfg.lr_gps,   "name": "gps"},
            {"params": [self.model.log_logit_scale],           "lr": cfg.lr_temp,  "name": "temperature"},
        ]
        return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    def train(self) -> None:
        cfg = self.cfg.training
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        history: list[dict] = []

        for epoch in range(self.start_epoch, cfg.epochs):
            train_loss = self._train_epoch(epoch)
            self._rotate_shard()

            if (epoch + 1) % cfg.eval_every == 0:
                metrics = evaluate(
                    self.model,
                    self.val_loader,
                    self.gallery_coords,
                    device=self.device,
                    thresholds_km=self.cfg.evaluation.thresholds_km,
                )
                median_gcd = metrics["median_gcd_km"]
                is_best = median_gcd < self.best_median_gcd
                if is_best:
                    self.best_median_gcd = median_gcd

                print(
                    f"[Epoch {epoch+1}/{cfg.epochs}] "
                    f"loss={train_loss:.4f} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                )

                history.append({"epoch": epoch + 1, "loss": train_loss, **metrics})
                self._save_figures(history, cfg.checkpoint_dir)

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict(),
                        "best_median_gcd": self.best_median_gcd,
                        "metrics": metrics,
                    },
                    checkpoint_dir=cfg.checkpoint_dir,
                    filename=f"epoch_{epoch+1:03d}.pt",
                    is_best=is_best,
                )

    def _save_figures(self, history: list[dict], checkpoint_dir: str) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        epochs = [h["epoch"] for h in history]
        acc_keys = [k for k in history[0] if k.startswith("recall@")]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epochs, [h["loss"] for h in history], marker="o")
        axes[0].set(title="Train loss", xlabel="Epoch", ylabel="Loss")
        axes[0].grid(True)

        for k in acc_keys:
            axes[1].plot(epochs, [h[k] for h in history], marker="o", label=k)
        axes[1].set(title="Accuracy @ thresholds", xlabel="Epoch", ylabel="Accuracy")
        axes[1].legend(fontsize=8)
        axes[1].grid(True)

        axes[2].plot(epochs, [h["median_gcd_km"] for h in history], marker="o", color="tab:red")
        axes[2].set(title="Median GCD (km)", xlabel="Epoch", ylabel="km")
        axes[2].grid(True)

        fig.tight_layout()
        fig.savefig(os.path.join(checkpoint_dir, "training_curves.png"), dpi=120)
        plt.close(fig)

    def _rotate_shard(self) -> None:
        """If the training dataset supports shard rotation, advance to the next shard."""
        dataset = self.train_loader.dataset
        if not hasattr(dataset, "next_shard"):
            return
        dataset.next_shard()
        self.train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=self.train_loader.num_workers,
            pin_memory=self.train_loader.pin_memory,
        )
        print(f"[Trainer] Shard rotated → {dataset.shard_progress}")

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        cfg = self.cfg.training

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for step, (images, coords) in enumerate(pbar):
            images = images.to(self.device)
            coords = coords.to(self.device)

            self.optimizer.zero_grad()

            # Decide whether to run attention regularization this step.
            # We do it every `attn_reg_every` steps to limit memory overhead
            # (returning all attention maps requires storing L × [B, H, S, S] tensors).
            use_attn_reg = (
                cfg.lambda_attn > 0.0
                and (step % cfg.attn_reg_every == 0)
            )

            with autocast("cuda", enabled=cfg.amp):
                if use_attn_reg:
                    img_emb, extras = self.model.image_encoder(
                        images, output_attentions=True
                    )
                    gps_emb = self.model.encode_gps(coords)
                    logit_scale = self.model.log_logit_scale.exp().clamp(max=100.0)
                else:
                    img_emb, gps_emb, logit_scale = self.model(images, coords)

                # Primary loss: InfoNCE with hard geographic negatives
                loss = info_nce_with_hard_negatives(
                    img_emb, gps_emb, coords, logit_scale,
                    swap_prob=cfg.hard_neg_swap_prob,
                    min_distance_km=cfg.hard_neg_min_dist_km,
                )

                # Auxiliary loss: attention entropy regularization
                if use_attn_reg:
                    attn_loss = attention_entropy_loss(extras["attentions"], layer_idx=-1)
                    loss = loss + cfg.lambda_attn * attn_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.model.clamp_temperature()
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()

            total_loss += loss.item()

            if (step + 1) % cfg.log_every == 0:
                lr_vit = self.optimizer.param_groups[0]["lr"]
                temp   = self.model.log_logit_scale.exp().item()
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    temp=f"{temp:.2f}",
                    lr=f"{lr_vit:.2e}",
                )

        return total_loss / len(self.train_loader)
