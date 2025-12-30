"""
Evaluation Module
=================

Metrics and evaluation utilities for recommendation systems.
"""

from .metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mrr_at_k,
    hit_rate_at_k,
    map_at_k,
    coverage,
    diversity,
    novelty,
    RecommenderEvaluator,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "mrr_at_k",
    "hit_rate_at_k",
    "map_at_k",
    "coverage",
    "diversity",
    "novelty",
    "RecommenderEvaluator",
]
