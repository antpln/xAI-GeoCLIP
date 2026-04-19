"""
Attention entropy regularization loss.

Penalizes high-entropy (diffuse) attention distributions in the last
transformer block. Low-entropy attention = the CLS token concentrates
on a small number of patches = spatially interpretable predictions.

This directly ties the training objective to interpretability: a model
that attends diffusely over the whole image is penalized, pushing it
toward learning focused, geographically meaningful visual cues.

Usage in trainer:
    _, extras = model.image_encoder(images, output_attentions=True)
    attn_loss  = attention_entropy_loss(extras["attentions"], layer_idx=-1)
    total_loss = infonce_loss + lambda_attn * attn_loss
"""
import torch
import torch.nn.functional as F


def attention_entropy(
    attn_weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-sample mean attention entropy for a single layer.

    Args:
        attn_weights: [B, num_heads, seq_len, seq_len] softmax attention weights.
        eps:          Numerical stability for log.

    Returns:
        [B] mean entropy across heads, computed on the CLS token row.
    """
    # CLS token (row 0) attends to all tokens — shape [B, num_heads, seq_len]
    cls_attn = attn_weights[:, :, 0, :]                          # [B, H, S]
    # Entropy: -sum(p * log(p))
    entropy = -(cls_attn * (cls_attn + eps).log()).sum(dim=-1)   # [B, H]
    return entropy.mean(dim=-1)                                   # [B]


def attention_entropy_loss(
    all_attentions: tuple,
    layer_idx: int = -1,
) -> torch.Tensor:
    """
    Mean attention entropy loss over the batch, for a chosen layer.

    Minimizing this loss during training encourages focused attention maps,
    making Grad-CAM and Attention Rollout explanations more localized and
    easier to interpret.

    Args:
        all_attentions: Tuple of [B, H, S, S] tensors, one per layer.
                        Obtained from model.image_encoder(images, output_attentions=True).
        layer_idx:      Which layer to regularize (-1 = last layer).
                        The last layer's attention most directly influences
                        the final CLS token used for prediction.

    Returns:
        Scalar loss (mean entropy across batch).
    """
    attn = all_attentions[layer_idx]   # [B, H, S, S]
    return attention_entropy(attn).mean()
