"""Tests for evaluation metrics."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    hit_rate_at_k,
    coverage,
    diversity
)


class TestRankingMetrics:
    """Tests for ranking metrics."""
    
    def test_precision_at_k_perfect(self):
        """Test precision@k with perfect predictions."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3, 4, 5}
        
        p5 = precision_at_k(recommended, relevant, k=5)
        assert p5 == 1.0
    
    def test_precision_at_k_partial(self):
        """Test precision@k with partial overlap."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        
        p5 = precision_at_k(recommended, relevant, k=5)
        assert p5 == 0.6
    
    def test_precision_at_k_no_overlap(self):
        """Test precision@k with no overlap."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {6, 7, 8}
        
        p5 = precision_at_k(recommended, relevant, k=5)
        assert p5 == 0.0
    
    def test_recall_at_k_perfect(self):
        """Test recall@k with perfect predictions."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        
        r5 = recall_at_k(recommended, relevant, k=5)
        assert r5 == 1.0
    
    def test_recall_at_k_partial(self):
        """Test recall@k with partial recall."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 6, 7, 8}
        
        r5 = recall_at_k(recommended, relevant, k=5)
        assert r5 == 0.4  # 2 out of 5
    
    def test_recall_at_k_empty_relevant(self):
        """Test recall@k with no relevant items."""
        recommended = [1, 2, 3, 4, 5]
        relevant = set()
        
        r5 = recall_at_k(recommended, relevant, k=5)
        assert r5 == 0.0
    
    def test_ndcg_at_k_perfect(self):
        """Test NDCG@k with perfect ranking."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}  # All at top
        
        ndcg = ndcg_at_k(recommended, relevant, k=5)
        assert ndcg == 1.0
    
    def test_ndcg_at_k_reversed(self):
        """Test NDCG@k with worst ranking."""
        recommended = [4, 5, 6, 1, 2]
        relevant = {1, 2}  # At bottom
        
        ndcg = ndcg_at_k(recommended, relevant, k=5)
        assert 0 < ndcg < 1  # Should be less than 1
    
    def test_ndcg_at_k_no_relevant(self):
        """Test NDCG@k with no relevant items in list."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {6, 7, 8}
        
        ndcg = ndcg_at_k(recommended, relevant, k=5)
        assert ndcg == 0.0
    
    def test_mrr_first_position(self):
        """Test MRR when first item is relevant."""
        recommendations = [[1, 2, 3], [4, 5, 6]]
        ground_truth = [{1, 7}, {4, 8}]
        
        mrr = mean_reciprocal_rank(recommendations, ground_truth)
        assert mrr == 1.0
    
    def test_mrr_second_position(self):
        """Test MRR when first relevant is at position 2."""
        recommendations = [[1, 2, 3]]
        ground_truth = [{2}]
        
        mrr = mean_reciprocal_rank(recommendations, ground_truth)
        assert mrr == 0.5
    
    def test_mrr_no_relevant(self):
        """Test MRR when no relevant items found."""
        recommendations = [[1, 2, 3]]
        ground_truth = [{4, 5}]
        
        mrr = mean_reciprocal_rank(recommendations, ground_truth)
        assert mrr == 0.0


class TestDiversityMetrics:
    """Tests for diversity and coverage metrics."""
    
    def test_coverage_full(self):
        """Test coverage when all items recommended."""
        recommendations = [[1, 2], [3, 4], [5]]
        total_items = 5
        
        cov = coverage(recommendations, total_items)
        assert cov == 1.0
    
    def test_coverage_partial(self):
        """Test coverage with partial item set."""
        recommendations = [[1, 2], [1, 3], [2, 3]]
        total_items = 10
        
        cov = coverage(recommendations, total_items)
        assert cov == 0.3  # 3 unique items out of 10
    
    def test_hit_rate_all_hits(self):
        """Test hit rate when all users have hits."""
        recommendations = [[1, 2, 3], [4, 5, 6]]
        ground_truth = [{1}, {5}]
        
        hr = hit_rate_at_k(recommendations, ground_truth, k=3)
        assert hr == 1.0
    
    def test_hit_rate_no_hits(self):
        """Test hit rate when no users have hits."""
        recommendations = [[1, 2, 3], [4, 5, 6]]
        ground_truth = [{7}, {8}]
        
        hr = hit_rate_at_k(recommendations, ground_truth, k=3)
        assert hr == 0.0


class TestMetricsEdgeCases:
    """Tests for edge cases in metrics."""
    
    def test_empty_recommendations(self):
        """Test metrics with empty recommendations."""
        recommended = []
        relevant = {1, 2, 3}
        
        assert precision_at_k(recommended, relevant, k=5) == 0.0
        assert recall_at_k(recommended, relevant, k=5) == 0.0
        assert ndcg_at_k(recommended, relevant, k=5) == 0.0
    
    def test_k_larger_than_list(self):
        """Test metrics when k is larger than list."""
        recommended = [1, 2, 3]
        relevant = {1, 2, 3, 4, 5}
        
        p10 = precision_at_k(recommended, relevant, k=10)
        assert p10 == 1.0  # All 3 items are relevant
    
    def test_negative_k_raises(self):
        """Test that negative k raises error."""
        recommended = [1, 2, 3]
        relevant = {1}
        
        with pytest.raises((ValueError, AssertionError)):
            precision_at_k(recommended, relevant, k=-1)
