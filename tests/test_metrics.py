import torch
import pytest
from geoclip.utils.geo_math import haversine_distance


def test_same_location_zero_distance():
    coords = torch.tensor([[48.8566, 2.3522]])
    dist = haversine_distance(coords, coords)
    assert dist.item() < 0.01, f"Same location should give ~0 km, got {dist.item()}"


def test_paris_to_london():
    paris = torch.tensor([[48.8566, 2.3522]])
    london = torch.tensor([[51.5074, -0.1278]])
    dist = haversine_distance(paris, london)
    # True distance ~341 km
    assert 300 < dist.item() < 380, f"Paris-London distance out of range: {dist.item():.1f} km"


def test_antipodal_points():
    p1 = torch.tensor([[0.0, 0.0]])
    p2 = torch.tensor([[0.0, 180.0]])
    dist = haversine_distance(p1, p2)
    # Half circumference of Earth = pi * 6371 ≈ 20015 km
    assert 19000 < dist.item() < 21000, f"Antipodal distance out of range: {dist.item():.1f} km"


def test_batch_distances():
    pred = torch.tensor([[48.8566, 2.3522], [51.5074, -0.1278]])
    true = torch.tensor([[51.5074, -0.1278], [48.8566, 2.3522]])
    dist = haversine_distance(pred, true)
    # Should be symmetric
    assert dist.shape == (2,)
    assert abs(dist[0].item() - dist[1].item()) < 0.1
