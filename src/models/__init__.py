"""Recommendation model implementations for Freak AI."""

from .two_tower import TwoTowerModel, UserTower, ItemTower
from .losses import InfoNCELoss, BPRLoss, SampledSoftmaxLoss

__all__ = [
    "TwoTowerModel",
    "UserTower",
    "ItemTower",
    "InfoNCELoss",
    "BPRLoss",
    "SampledSoftmaxLoss",
]
