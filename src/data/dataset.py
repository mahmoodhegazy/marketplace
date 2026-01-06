"""
Dataset Classes for Freak AI
=============================

PyTorch and TensorFlow dataset implementations for training
recommendation models with negative sampling.
"""

import random
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ..utils.logger import get_logger

logger = get_logger(__name__)


class InteractionDataset(Dataset):
    """
    PyTorch Dataset for user-item interactions with negative sampling.
    
    Implements in-batch negative sampling and hard negative mining
    for efficient training of two-tower models.
    
    Parameters
    ----------
    interactions_df : pd.DataFrame
        DataFrame with user_idx, item_idx, interaction_strength columns.
    items_df : pd.DataFrame
        DataFrame with item features.
    num_negatives : int
        Number of negative samples per positive.
    hard_negative_ratio : float
        Proportion of hard negatives (same category).
    item_features : List[str]
        List of item feature columns to include.
    """
    
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        items_df: pd.DataFrame,
        num_negatives: int = 4,
        hard_negative_ratio: float = 0.3,
        item_features: Optional[List[str]] = None,
    ):
        self.interactions = interactions_df.reset_index(drop=True)
        self.items = items_df.set_index('item_idx')
        self.num_negatives = num_negatives
        self.hard_negative_ratio = hard_negative_ratio
        
        # Default item features
        self.item_features = item_features or [
            'category_idx', 'brand_idx', 'condition_idx', 
            'size_idx', 'price_normalized'
        ]
        
        # Build lookup structures
        self._build_lookups()
        
        logger.info(f"Created dataset with {len(self.interactions)} interactions")
    
    def _build_lookups(self):
        """Build data structures for efficient negative sampling."""
        # User -> positive items set
        self.user_positives: Dict[int, Set[int]] = defaultdict(set)
        for _, row in self.interactions.iterrows():
            self.user_positives[row['user_idx']].add(row['item_idx'])
        
        # All item indices
        self.all_items = set(self.items.index.tolist())
        
        # Category -> items mapping (for hard negatives)
        self.category_items: Dict[int, List[int]] = defaultdict(list)
        if 'category_idx' in self.items.columns:
            for item_idx, row in self.items.iterrows():
                self.category_items[row['category_idx']].append(item_idx)
        
        # Item -> category mapping
        self.item_category = {}
        if 'category_idx' in self.items.columns:
            self.item_category = self.items['category_idx'].to_dict()
    
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample with negative samples.
        
        Returns
        -------
        Dict with:
            - user_idx: User index
            - pos_item_idx: Positive item index
            - neg_item_indices: Negative item indices
            - pos_item_features: Features for positive item
            - neg_item_features: Features for negative items
            - interaction_weight: Weight for the interaction
        """
        row = self.interactions.iloc[idx]
        user_idx = int(row['user_idx'])
        pos_item_idx = int(row['item_idx'])
        
        # Sample negatives
        neg_items = self._sample_negatives(user_idx, pos_item_idx)
        
        # Get item features
        pos_features = self._get_item_features(pos_item_idx)
        neg_features = [self._get_item_features(neg_idx) for neg_idx in neg_items]
        
        # Stack negative features
        neg_features_stacked = {
            key: torch.stack([nf[key] for nf in neg_features])
            for key in neg_features[0].keys()
        }
        
        return {
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'pos_item_idx': torch.tensor(pos_item_idx, dtype=torch.long),
            'neg_item_indices': torch.tensor(neg_items, dtype=torch.long),
            'pos_item_features': pos_features,
            'neg_item_features': neg_features_stacked,
            'interaction_weight': torch.tensor(
                row.get('interaction_strength', 1.0), 
                dtype=torch.float32
            ),
        }
    
    def _sample_negatives(
        self, 
        user_idx: int, 
        pos_item_idx: int
    ) -> List[int]:
        """Sample negative items for a user-item pair."""
        positives = self.user_positives[user_idx]
        candidates = self.all_items - positives
        
        if len(candidates) == 0:
            # Fallback: use items not equal to positive
            candidates = self.all_items - {pos_item_idx}
        
        candidates_list = list(candidates)
        
        # Determine hard vs random negatives
        num_hard = int(self.num_negatives * self.hard_negative_ratio)
        num_random = self.num_negatives - num_hard
        
        negatives = []
        
        # Hard negatives (same category)
        if num_hard > 0 and pos_item_idx in self.item_category:
            pos_category = self.item_category[pos_item_idx]
            category_items = set(self.category_items.get(pos_category, []))
            hard_candidates = list(category_items - positives - {pos_item_idx})
            
            if hard_candidates:
                hard_sample_size = min(num_hard, len(hard_candidates))
                negatives.extend(random.sample(hard_candidates, hard_sample_size))
        
        # Random negatives
        remaining = self.num_negatives - len(negatives)
        if remaining > 0:
            available = list(set(candidates_list) - set(negatives))
            random_sample_size = min(remaining, len(available))
            negatives.extend(random.sample(available, random_sample_size))
        
        # Pad if needed
        while len(negatives) < self.num_negatives:
            negatives.append(random.choice(candidates_list))
        
        return negatives
    
    def _get_item_features(self, item_idx: int) -> Dict[str, torch.Tensor]:
        """Get features for an item."""
        if item_idx not in self.items.index:
            # Return zeros for missing items
            return {
                feat: torch.tensor(0, dtype=torch.long if 'idx' in feat else torch.float32)
                for feat in self.item_features
            }
        
        item = self.items.loc[item_idx]
        features = {}
        
        for feat in self.item_features:
            if feat in item.index:
                val = item[feat]
                if 'idx' in feat:
                    features[feat] = torch.tensor(int(val), dtype=torch.long)
                else:
                    features[feat] = torch.tensor(float(val), dtype=torch.float32)
            else:
                features[feat] = torch.tensor(0, dtype=torch.long if 'idx' in feat else torch.float32)
        
        return features


class FreakDataset(Dataset):
    """
    Simplified dataset for two-tower model training.
    
    Returns triplets of (user, positive_item, negative_item) for
    contrastive learning.
    """
    
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        num_items: int,
        num_negatives: int = 1,
    ):
        self.interactions = interactions_df.reset_index(drop=True)
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # Build user positive sets
        self.user_positives: Dict[int, Set[int]] = defaultdict(set)
        for _, row in self.interactions.iterrows():
            self.user_positives[row['user_idx']].add(row['item_idx'])
    
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        row = self.interactions.iloc[idx]
        user_idx = int(row['user_idx'])
        pos_item_idx = int(row['item_idx'])
        
        # Sample negative
        negatives = []
        positives = self.user_positives[user_idx]
        
        for _ in range(self.num_negatives):
            neg_idx = random.randint(0, self.num_items - 1)
            while neg_idx in positives:
                neg_idx = random.randint(0, self.num_items - 1)
            negatives.append(neg_idx)
        
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(pos_item_idx, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long),
            torch.tensor(row.get('interaction_strength', 1.0), dtype=torch.float32),
        )


class InferenceBatchDataset(Dataset):
    """
    Dataset for batch inference/embedding generation.
    
    Used for generating embeddings for all users or items.
    """
    
    def __init__(
        self,
        indices: List[int],
        features_df: Optional[pd.DataFrame] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        self.indices = indices
        self.features_df = features_df
        self.feature_columns = feature_columns or []
        
        if features_df is not None:
            self.features_df = features_df.set_index(
                features_df.columns[0] if features_df.index.name is None 
                else features_df.index
            )
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        index = self.indices[idx]
        result = {'index': torch.tensor(index, dtype=torch.long)}
        
        if self.features_df is not None and index in self.features_df.index:
            row = self.features_df.loc[index]
            for col in self.feature_columns:
                if col in row.index:
                    val = row[col]
                    dtype = torch.long if 'idx' in col else torch.float32
                    result[col] = torch.tensor(val, dtype=dtype)
        
        return result


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    items_df: pd.DataFrame,
    batch_size: int = 1024,
    num_negatives: int = 4,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training interactions.
    val_df : pd.DataFrame
        Validation interactions.
    items_df : pd.DataFrame
        Item features.
    batch_size : int
        Batch size.
    num_negatives : int
        Number of negative samples.
    num_workers : int
        Number of data loading workers.
    
    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and validation DataLoaders.
    """
    train_dataset = InteractionDataset(
        train_df, items_df, num_negatives=num_negatives
    )
    val_dataset = InteractionDataset(
        val_df, items_df, num_negatives=num_negatives
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def collate_with_negatives(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batches with negative samples.
    
    Handles variable-length negative features.
    """
    result = {
        'user_idx': torch.stack([b['user_idx'] for b in batch]),
        'pos_item_idx': torch.stack([b['pos_item_idx'] for b in batch]),
        'neg_item_indices': torch.stack([b['neg_item_indices'] for b in batch]),
        'interaction_weight': torch.stack([b['interaction_weight'] for b in batch]),
    }
    
    # Stack positive item features
    pos_features = {}
    for key in batch[0]['pos_item_features'].keys():
        pos_features[key] = torch.stack([b['pos_item_features'][key] for b in batch])
    result['pos_item_features'] = pos_features
    
    # Stack negative item features
    neg_features = {}
    for key in batch[0]['neg_item_features'].keys():
        neg_features[key] = torch.stack([b['neg_item_features'][key] for b in batch])
    result['neg_item_features'] = neg_features
    
    return result
