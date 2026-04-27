import os
import torch
from typing import Optional


def load_pretrained_geoclip_weights(model, weights_dir: str, device: str = "cpu") -> None:
    """
    Warm-start a GeoCLIP model from the original GeoCLIP pre-trained weights.
    skip gps_encoder because architectures are incompatible.
    """
    # --- logit scale ---
    ls_path = os.path.join(weights_dir, "logit_scale_weights.pth")
    ls = torch.load(ls_path, map_location=device)
    with torch.no_grad():
        model.log_logit_scale.copy_(ls)
    print(f"[Pretrained] Loaded logit_scale = {ls.item():.4f} (exp={ls.exp().item():.2f})")

    # --- image encoder projection head ---
    proj_path = os.path.join(weights_dir, "image_encoder_mlp_weights.pth")
    proj_weights = torch.load(proj_path, map_location=device)
    proj = model.image_encoder.projection
    # original Sequential: Linear(768,768)[0] -> act[1] -> Linear(768,512)[2]
    # our Sequential:      Linear(768,768)[0] -> GELU[1] -> LayerNorm[2] -> Linear(768,512)[3]
    with torch.no_grad():
        proj[0].weight.copy_(proj_weights["0.weight"])
        proj[0].bias.copy_(proj_weights["0.bias"])
        proj[3].weight.copy_(proj_weights["2.weight"])
        proj[3].bias.copy_(proj_weights["2.bias"])
    print("[Pretrained] Loaded image_encoder projection head (layers 0 and 3).")


def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
) -> None:
    """Save training state to disk. If is_best, also overwrite best.pt."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        torch.save(state, best_path)
        print(f"[Checkpoint] New best model saved to {best_path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
) -> dict:
    """
    Load a checkpoint. Always restores model weights; optionally restores
    optimizer and scheduler state if provided and present in the checkpoint.
    Returns the raw checkpoint dict (contains epoch, metrics, etc.).
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    print(f"[Checkpoint] Loaded from {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint
