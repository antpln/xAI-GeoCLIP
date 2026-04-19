import os
import torch
from typing import Optional


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
