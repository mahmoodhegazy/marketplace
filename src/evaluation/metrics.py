"""
Recommendation System Evaluation Metrics
========================================

Implements standard IR and recommendation metrics:
- Precision@K, Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Hit Rate@K
- MAP@K (Mean Average Precision)
- Coverage
- Diversity (Intra-list diversity)
- Novelty
"""

import numpy as np
from typing import List, Dict, Set, Optional, Union
from collections import defaultdict
import pandas as pd
from scipy.spatial.distance import pdist


def precision_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int,
) -> float:
    """
    Compute Precision@K.
    
    Precision@K = |recommended ∩ relevant| / K
    
    Parameters
    ----------
    recommended : List[int]
        Ordered list of recommended item IDs.
    relevant : Set[int]
        Set of relevant (ground truth) item IDs.
    k : int
        Number of recommendations to consider.
    
    Returns
    -------
    float
        Precision@K score.
    """
    if k <= 0:
        return 0.0
    
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant)
    return hits / k


def recall_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int,
) -> float:
    """
    Compute Recall@K.
    
    Recall@K = |recommended ∩ relevant| / |relevant|
    
    Parameters
    ----------
    recommended : List[int]
        Ordered list of recommended item IDs.
    relevant : Set[int]
        Set of relevant (ground truth) item IDs.
    k : int
        Number of recommendations to consider.
    
    Returns
    -------
    float
        Recall@K score.
    """
    if len(relevant) == 0:
        return 0.0
    
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int,
    relevance_scores: Optional[Dict[int, float]] = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.
    
    NDCG@K = DCG@K / IDCG@K
    
    DCG@K = Σ (rel_i / log2(i + 1))
    
    Parameters
    ----------
    recommended : List[int]
        Ordered list of recommended item IDs.
    relevant : Set[int]
        Set of relevant item IDs.
    k : int
        Number of recommendations to consider.
    relevance_scores : Dict[int, float], optional
        Graded relevance scores. If None, binary relevance is used.
    
    Returns
    -------
    float
        NDCG@K score between 0 and 1.
    """
    if k <= 0 or len(relevant) == 0:
        return 0.0
    
    recommended_k = recommended[:k]
    
    # Compute DCG
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            rel = relevance_scores.get(item, 1.0) if relevance_scores else 1.0
            dcg += rel / np.log2(i + 2)  # i + 2 because i is 0-indexed
    
    # Compute IDCG (ideal DCG)
    if relevance_scores:
        ideal_rels = sorted(
            [relevance_scores.get(item, 1.0) for item in relevant],
            reverse=True
        )[:k]
    else:
        ideal_rels = [1.0] * min(k, len(relevant))
    
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mrr_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int,
) -> float:
    """
    Compute Mean Reciprocal Rank at K.
    
    MRR = 1 / rank_of_first_relevant_item
    
    Parameters
    ----------
    recommended : List[int]
        Ordered list of recommended item IDs.
    relevant : Set[int]
        Set of relevant item IDs.
    k : int
        Number of recommendations to consider.
    
    Returns
    -------
    float
        MRR score (reciprocal rank of first hit).
    """
    recommended_k = recommended[:k]
    
    for i, item in enumerate(recommended_k):
        if item in relevant:
            return 1.0 / (i + 1)
    
    return 0.0


def hit_rate_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int,
) -> float:
    """
    Compute Hit Rate at K (binary indicator).
    
    Hit Rate@K = 1 if any relevant item in top K, else 0
    
    Parameters
    ----------
    recommended : List[int]
        Ordered list of recommended item IDs.
    relevant : Set[int]
        Set of relevant item IDs.
    k : int
        Number of recommendations to consider.
    
    Returns
    -------
    float
        1.0 if hit, 0.0 otherwise.
    """
    recommended_k = set(recommended[:k])
    return 1.0 if len(recommended_k & relevant) > 0 else 0.0


def map_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int,
) -> float:
    """
    Compute Average Precision at K.
    
    AP@K = (1/min(K, |relevant|)) * Σ Precision@i * rel_i
    
    Parameters
    ----------
    recommended : List[int]
        Ordered list of recommended item IDs.
    relevant : Set[int]
        Set of relevant item IDs.
    k : int
        Number of recommendations to consider.
    
    Returns
    -------
    float
        Average Precision@K score.
    """
    if len(relevant) == 0:
        return 0.0
    
    recommended_k = recommended[:k]
    
    score = 0.0
    hits = 0
    
    for i, item in enumerate(recommended_k):
        if item in relevant:
            hits += 1
            precision_at_i = hits / (i + 1)
            score += precision_at_i
    
    return score / min(k, len(relevant))


def coverage(
    all_recommendations: List[List[int]],
    catalog_size: int,
) -> float:
    """
    Compute catalog coverage.
    
    Coverage = |unique recommended items| / |catalog|
    
    Parameters
    ----------
    all_recommendations : List[List[int]]
        List of recommendation lists for all users.
    catalog_size : int
        Total number of items in catalog.
    
    Returns
    -------
    float
        Coverage ratio between 0 and 1.
    """
    if catalog_size == 0:
        return 0.0
    
    unique_items = set()
    for recs in all_recommendations:
        unique_items.update(recs)
    
    return len(unique_items) / catalog_size


def diversity(
    recommendations: List[int],
    item_embeddings: np.ndarray,
    item_to_idx: Dict[int, int],
) -> float:
    """
    Compute intra-list diversity using cosine distance.
    
    Diversity = average pairwise distance between recommended items
    
    Parameters
    ----------
    recommendations : List[int]
        List of recommended item IDs.
    item_embeddings : np.ndarray
        Item embedding matrix, shape (num_items, dim).
    item_to_idx : Dict[int, int]
        Mapping from item ID to embedding index.
    
    Returns
    -------
    float
        Diversity score (higher = more diverse).
    """
    if len(recommendations) < 2:
        return 0.0
    
    # Get embeddings for recommended items
    valid_indices = []
    for item_id in recommendations:
        if item_id in item_to_idx:
            valid_indices.append(item_to_idx[item_id])
    
    if len(valid_indices) < 2:
        return 0.0
    
    embeddings = item_embeddings[valid_indices]
    
    # Compute pairwise cosine distances
    distances = pdist(embeddings, metric='cosine')
    
    return float(np.mean(distances))


def novelty(
    recommendations: List[int],
    item_popularity: Dict[int, int],
    num_users: int,
) -> float:
    """
    Compute novelty (inverse popularity).
    
    Novelty = average self-information of recommended items
    Self-information = -log2(popularity)
    
    Parameters
    ----------
    recommendations : List[int]
        List of recommended item IDs.
    item_popularity : Dict[int, int]
        Number of interactions per item.
    num_users : int
        Total number of users.
    
    Returns
    -------
    float
        Novelty score (higher = more novel/obscure items).
    """
    if len(recommendations) == 0 or num_users == 0:
        return 0.0
    
    total_novelty = 0.0
    valid_count = 0
    
    for item_id in recommendations:
        popularity = item_popularity.get(item_id, 1)
        # Probability of interaction
        prob = popularity / num_users
        # Self-information (capped to avoid inf)
        self_info = -np.log2(max(prob, 1e-10))
        total_novelty += self_info
        valid_count += 1
    
    if valid_count == 0:
        return 0.0
    
    return total_novelty / valid_count


class RecommenderEvaluator:
    """
    Comprehensive evaluator for recommendation systems.
    
    Computes multiple metrics across all users and aggregates results.
    
    Parameters
    ----------
    k_values : List[int]
        List of K values to evaluate (e.g., [5, 10, 20]).
    metrics : List[str]
        Metrics to compute. Options: precision, recall, ndcg, mrr, hit_rate, map.
    """
    
    def __init__(
        self,
        k_values: List[int] = [5, 10, 20],
        metrics: List[str] = None,
    ):
        self.k_values = k_values
        self.metrics = metrics or ['precision', 'recall', 'ndcg', 'mrr', 'hit_rate']
        
        self.metric_funcs = {
            'precision': precision_at_k,
            'recall': recall_at_k,
            'ndcg': ndcg_at_k,
            'mrr': mrr_at_k,
            'hit_rate': hit_rate_at_k,
            'map': map_at_k,
        }
    
    def evaluate_user(
        self,
        recommended: List[int],
        relevant: Set[int],
        relevance_scores: Optional[Dict[int, float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate recommendations for a single user.
        
        Parameters
        ----------
        recommended : List[int]
            Ordered list of recommended item IDs.
        relevant : Set[int]
            Set of relevant item IDs.
        relevance_scores : Dict[int, float], optional
            Graded relevance scores for NDCG.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of metric@k scores.
        """
        results = {}
        
        for metric_name in self.metrics:
            func = self.metric_funcs[metric_name]
            
            for k in self.k_values:
                key = f"{metric_name}@{k}"
                
                if metric_name == 'ndcg':
                    results[key] = func(recommended, relevant, k, relevance_scores)
                else:
                    results[key] = func(recommended, relevant, k)
        
        return results
    
    def evaluate_all(
        self,
        recommendations: Dict[int, List[int]],
        ground_truth: Dict[int, Set[int]],
        relevance_scores: Optional[Dict[int, Dict[int, float]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate recommendations for all users.
        
        Parameters
        ----------
        recommendations : Dict[int, List[int]]
            Mapping from user ID to ordered list of recommended items.
        ground_truth : Dict[int, Set[int]]
            Mapping from user ID to set of relevant items.
        relevance_scores : Dict[int, Dict[int, float]], optional
            Mapping from user ID to item relevance scores.
        
        Returns
        -------
        Dict[str, float]
            Averaged metrics across all users.
        """
        all_results = defaultdict(list)
        
        for user_id, recs in recommendations.items():
            if user_id not in ground_truth:
                continue
            
            relevant = ground_truth[user_id]
            
            if len(relevant) == 0:
                continue
            
            user_relevance = None
            if relevance_scores and user_id in relevance_scores:
                user_relevance = relevance_scores[user_id]
            
            user_results = self.evaluate_user(recs, relevant, user_relevance)
            
            for key, value in user_results.items():
                all_results[key].append(value)
        
        # Average across users
        averaged = {}
        for key, values in all_results.items():
            averaged[key] = np.mean(values) if values else 0.0
        
        # Add user count
        averaged['num_evaluated_users'] = len(all_results.get(f'{self.metrics[0]}@{self.k_values[0]}', []))
        
        return averaged
    
    def evaluate_with_embeddings(
        self,
        recommendations: Dict[int, List[int]],
        ground_truth: Dict[int, Set[int]],
        item_embeddings: np.ndarray,
        item_to_idx: Dict[int, int],
        item_popularity: Dict[int, int],
        catalog_size: int,
        num_users: int,
    ) -> Dict[str, float]:
        """
        Full evaluation including beyond-accuracy metrics.
        
        Parameters
        ----------
        recommendations : Dict[int, List[int]]
            User recommendations.
        ground_truth : Dict[int, Set[int]]
            Ground truth relevant items.
        item_embeddings : np.ndarray
            Item embedding matrix.
        item_to_idx : Dict[int, int]
            Item ID to index mapping.
        item_popularity : Dict[int, int]
            Item interaction counts.
        catalog_size : int
            Total items in catalog.
        num_users : int
            Total number of users.
        
        Returns
        -------
        Dict[str, float]
            All metrics including accuracy, diversity, coverage, novelty.
        """
        # Standard accuracy metrics
        results = self.evaluate_all(recommendations, ground_truth)
        
        # Coverage
        all_recs = list(recommendations.values())
        results['coverage'] = coverage(all_recs, catalog_size)
        
        # Average diversity and novelty
        diversities = []
        novelties = []
        
        for user_id, recs in recommendations.items():
            if len(recs) > 0:
                div = diversity(recs, item_embeddings, item_to_idx)
                nov = novelty(recs, item_popularity, num_users)
                diversities.append(div)
                novelties.append(nov)
        
        results['diversity'] = np.mean(diversities) if diversities else 0.0
        results['novelty'] = np.mean(novelties) if novelties else 0.0
        
        return results
    
    def evaluate_cold_warm_split(
        self,
        recommendations: Dict[int, List[int]],
        ground_truth: Dict[int, Set[int]],
        user_interaction_counts: Dict[int, int],
        cold_threshold: int = 5,
        warm_threshold: int = 20,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate separately for cold, warm, and hot users.
        
        Parameters
        ----------
        recommendations : Dict[int, List[int]]
            User recommendations.
        ground_truth : Dict[int, Set[int]]
            Ground truth items.
        user_interaction_counts : Dict[int, int]
            Number of interactions per user.
        cold_threshold : int
            Max interactions for cold users.
        warm_threshold : int
            Max interactions for warm users.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Metrics split by user segment.
        """
        cold_recs = {}
        warm_recs = {}
        hot_recs = {}
        
        cold_gt = {}
        warm_gt = {}
        hot_gt = {}
        
        for user_id, recs in recommendations.items():
            count = user_interaction_counts.get(user_id, 0)
            
            if user_id not in ground_truth:
                continue
            
            gt = ground_truth[user_id]
            
            if count < cold_threshold:
                cold_recs[user_id] = recs
                cold_gt[user_id] = gt
            elif count < warm_threshold:
                warm_recs[user_id] = recs
                warm_gt[user_id] = gt
            else:
                hot_recs[user_id] = recs
                hot_gt[user_id] = gt
        
        return {
            'cold': self.evaluate_all(cold_recs, cold_gt),
            'warm': self.evaluate_all(warm_recs, warm_gt),
            'hot': self.evaluate_all(hot_recs, hot_gt),
        }
    
    def to_dataframe(self, results: Dict[str, float]) -> pd.DataFrame:
        """
        Convert results to a formatted DataFrame.
        
        Parameters
        ----------
        results : Dict[str, float]
            Evaluation results.
        
        Returns
        -------
        pd.DataFrame
            Formatted results table.
        """
        # Separate metrics by K value
        rows = []
        
        for metric in self.metrics:
            row = {'metric': metric}
            for k in self.k_values:
                key = f"{metric}@{k}"
                if key in results:
                    row[f'@{k}'] = results[key]
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Add beyond-accuracy metrics
        beyond_accuracy = []
        for key in ['coverage', 'diversity', 'novelty']:
            if key in results:
                beyond_accuracy.append({
                    'metric': key,
                    '@all': results[key],
                })
        
        if beyond_accuracy:
            df_beyond = pd.DataFrame(beyond_accuracy)
            return df, df_beyond
        
        return df


def offline_ab_test(
    model_a_results: Dict[str, float],
    model_b_results: Dict[str, float],
    primary_metric: str = 'ndcg@10',
) -> Dict[str, any]:
    """
    Compare two models offline.
    
    Parameters
    ----------
    model_a_results : Dict[str, float]
        Evaluation results for model A.
    model_b_results : Dict[str, float]
        Evaluation results for model B.
    primary_metric : str
        Primary metric for comparison.
    
    Returns
    -------
    Dict
        Comparison summary.
    """
    comparison = {
        'primary_metric': primary_metric,
        'model_a': model_a_results.get(primary_metric, 0),
        'model_b': model_b_results.get(primary_metric, 0),
    }
    
    comparison['winner'] = 'A' if comparison['model_a'] > comparison['model_b'] else 'B'
    comparison['lift'] = (
        (comparison['model_b'] - comparison['model_a']) / comparison['model_a'] * 100
        if comparison['model_a'] > 0 else float('inf')
    )
    
    # Compare all metrics
    metric_comparison = {}
    all_keys = set(model_a_results.keys()) | set(model_b_results.keys())
    
    for key in all_keys:
        if key == 'num_evaluated_users':
            continue
        
        a_val = model_a_results.get(key, 0)
        b_val = model_b_results.get(key, 0)
        
        metric_comparison[key] = {
            'model_a': a_val,
            'model_b': b_val,
            'diff': b_val - a_val,
            'pct_change': (b_val - a_val) / a_val * 100 if a_val > 0 else float('inf'),
        }
    
    comparison['all_metrics'] = metric_comparison
    
    return comparison
