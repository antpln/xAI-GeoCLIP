"""
Hard geographic negative mining for contrastive training.

Standard InfoNCE treats all other samples in the batch as equally valid
negatives. This is too easy: many negatives are geographically and visually
dissimilar, and the model can learn to separate them via spurious cues
(brightness, camera type, compression artifacts).

Hard negative mining forces the model to separate:
  - Visually similar images at very different locations  (hard geographic negatives)
  - Geographically close images that look different      (hard visual negatives, optional)

By mixing hard negatives into the training batch, the InfoNCE loss
can only be reduced by learning genuine geographic visual cues.

Strategy implemented here:
  Given a batch of (image, gps) pairs, we replace a fraction of the GPS
  negatives with "geographically hard" ones: GPS points close to the anchor
  but belonging to a different image.  Concretely, for each anchor i we find
  the j ≠ i in the batch whose GPS is closest, and swap that GPS embedding
  into the negative pool with probability `swap_prob`.

  This is an in-batch approximation that adds zero data-loading overhead.
"""
import torch
import torch.nn.functional as F

from geoclip.utils.geo_math import haversine_distance


def build_hard_negative_gps_embs(
    gps_emb: torch.Tensor,
    coords: torch.Tensor,
    swap_prob: float = 0.5,
    min_distance_km: float = 500.0,
) -> torch.Tensor:
    """
    Construct a modified GPS embedding matrix where some "easy" negatives
    (geographically distant) are replaced by "hard" ones (geographically close
    to the anchor but belonging to a different sample).

    Args:
        gps_emb:          [B, D] GPS embeddings for the current batch.
        coords:           [B, 2] raw (lat, lon) coordinates.
        swap_prob:        Probability of swapping each negative with its hardest one.
        min_distance_km:  A pair is only a "hard" negative if their geographic
                          distance is below this threshold.

    Returns:
        [B, D] modified GPS embeddings to use as the negative column in InfoNCE.
        For unswapped rows this is identical to the original gps_emb.
    """
    B = gps_emb.shape[0]

    # Pairwise geographic distances [B, B]
    coords_i = coords.unsqueeze(1).expand(B, B, 2).reshape(B * B, 2)
    coords_j = coords.unsqueeze(0).expand(B, B, 2).reshape(B * B, 2)
    dist_matrix = haversine_distance(coords_i, coords_j).reshape(B, B)  # [B, B]

    # Mask out self-pairs (set diagonal to inf so they are never chosen)
    dist_matrix.fill_diagonal_(float("inf"))

    # For each anchor, find its nearest neighbor in the batch
    nearest_dist, nearest_idx = dist_matrix.min(dim=1)   # [B], [B]

    hard_emb = gps_emb.clone()
    for i in range(B):
        if nearest_dist[i].item() < min_distance_km:
            if torch.rand(1).item() < swap_prob:
                # Replace row i with the GPS embedding of its nearest neighbor
                hard_emb[i] = gps_emb[nearest_idx[i]].detach()

    return hard_emb


def info_nce_with_hard_negatives(
    img_emb: torch.Tensor,
    gps_emb: torch.Tensor,
    coords: torch.Tensor,
    logit_scale: torch.Tensor,
    swap_prob: float = 0.5,
    min_distance_km: float = 500.0,
) -> torch.Tensor:
    """
    InfoNCE loss augmented with hard geographic negatives.

    The image-to-GPS direction uses a modified negative set that includes
    geographically close (but incorrect) GPS embeddings.  The GPS-to-image
    direction remains unchanged.

    Args:
        img_emb:          [B, D] L2-normalized image embeddings.
        gps_emb:          [B, D] L2-normalized GPS embeddings.
        coords:           [B, 2] raw (lat, lon) in degrees.
        logit_scale:      Scalar temperature.
        swap_prob:        Probability of injecting a hard negative per sample.
        min_distance_km:  Distance threshold below which a pair is considered hard.

    Returns:
        Scalar loss.
    """
    hard_gps_emb = build_hard_negative_gps_embs(
        gps_emb, coords, swap_prob=swap_prob, min_distance_km=min_distance_km
    )

    # Image → GPS: use hard negatives in the column (GPS) side
    logits_i2g = logit_scale * img_emb @ hard_gps_emb.T    # [B, B]
    # GPS → Image: standard negatives
    logits_g2i = logit_scale * gps_emb @ img_emb.T         # [B, B]

    labels = torch.arange(B := img_emb.shape[0], device=img_emb.device)
    loss_i2g = F.cross_entropy(logits_i2g, labels)
    loss_g2i = F.cross_entropy(logits_g2i, labels)
    return (loss_i2g + loss_g2i) / 2.0
