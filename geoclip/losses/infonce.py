import torch
import torch.nn.functional as F


def info_nce_loss(
    img_emb: torch.Tensor,
    gps_emb: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric InfoNCE (NT-Xent) loss for image-GPS contrastive learning.

    Args:
        img_emb:     [B, D] L2-normalized image embeddings.
        gps_emb:     [B, D] L2-normalized GPS embeddings.
        logit_scale: scalar temperature (already exponentiated, not log).

    Returns:
        Scalar loss.
    """
    logits = logit_scale * img_emb @ gps_emb.T   # [B, B]
    labels = torch.arange(len(img_emb), device=img_emb.device)
    loss_i2g = F.cross_entropy(logits, labels)
    loss_g2i = F.cross_entropy(logits.T, labels)
    return (loss_i2g + loss_g2i) / 2.0
