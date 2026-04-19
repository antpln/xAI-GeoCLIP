import torch


_EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
) -> torch.Tensor:
    """
    Computes great-circle distance (km) between predicted and true coordinates.

    Args:
        pred_coords: [B, 2] (lat, lon) in degrees.
        true_coords: [B, 2] (lat, lon) in degrees.

    Returns:
        [B] distances in km.
    """
    pred = torch.deg2rad(pred_coords)
    true = torch.deg2rad(true_coords)

    lat1, lon1 = pred[:, 0], pred[:, 1]
    lat2, lon2 = true[:, 0], true[:, 1]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))
    return _EARTH_RADIUS_KM * c
