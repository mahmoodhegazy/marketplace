"""
Feature Engineering for Freak AI
=================================

Comprehensive feature engineering for users and items.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for recommendation system.
    
    Generates user-level and item-level features for model input.
    
    Features include:
    - User: interaction patterns, category preferences, recency
    - Item: visual embeddings, price tier, popularity, freshness
    """
    
    def __init__(self):
        self.user_features: Optional[pd.DataFrame] = None
        self.item_features: Optional[pd.DataFrame] = None
        self.scalers: Dict[str, StandardScaler] = {}
    
    def compute_user_features(
        self,
        interactions_df: pd.DataFrame,
        items_df: pd.DataFrame,
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Compute user-level features from interaction history.
        
        Parameters
        ----------
        interactions_df : pd.DataFrame
            User-item interactions.
        items_df : pd.DataFrame
            Item catalog.
        reference_date : datetime, optional
            Reference date for recency calculations.
        
        Returns
        -------
        pd.DataFrame
            User features indexed by user_idx.
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        # Merge interactions with items
        interactions = interactions_df.merge(
            items_df[['item_idx', 'category_idx', 'brand_idx', 'price', 'condition_idx']],
            on='item_idx',
            how='left'
        )
        
        features_list = []
        
        for user_idx, user_data in interactions.groupby('user_idx'):
            features = {'user_idx': user_idx}
            
            # Interaction count features
            features['total_interactions'] = len(user_data)
            features['unique_items'] = user_data['item_idx'].nunique()
            
            # Event type counts
            if 'event_types' in user_data.columns:
                all_events = []
                for events in user_data['event_types']:
                    if isinstance(events, list):
                        all_events.extend(events)
                features['save_count'] = all_events.count('save')
                features['cart_count'] = all_events.count('cart')
                features['order_count'] = all_events.count('order')
            
            # Category preferences (top 5 as one-hot would be expensive)
            cat_counts = user_data['category_idx'].value_counts()
            features['top_category_1'] = cat_counts.index[0] if len(cat_counts) > 0 else -1
            features['top_category_2'] = cat_counts.index[1] if len(cat_counts) > 1 else -1
            features['top_category_3'] = cat_counts.index[2] if len(cat_counts) > 2 else -1
            features['category_diversity'] = len(cat_counts)
            
            # Brand preferences
            brand_counts = user_data['brand_idx'].value_counts()
            features['top_brand_1'] = brand_counts.index[0] if len(brand_counts) > 0 else -1
            features['top_brand_2'] = brand_counts.index[1] if len(brand_counts) > 1 else -1
            features['brand_diversity'] = len(brand_counts)
            
            # Price features
            features['avg_price'] = user_data['price'].mean()
            features['max_price'] = user_data['price'].max()
            features['min_price'] = user_data['price'].min()
            features['price_std'] = user_data['price'].std() if len(user_data) > 1 else 0
            
            # Condition preferences
            cond_counts = user_data['condition_idx'].value_counts()
            features['preferred_condition'] = cond_counts.index[0] if len(cond_counts) > 0 else -1
            
            # Recency features
            if 'last_interaction' in user_data.columns:
                last_interaction = pd.to_datetime(user_data['last_interaction']).max()
                if pd.notna(last_interaction):
                    days_since = (reference_date - last_interaction.to_pydatetime()).days
                    features['days_since_last'] = days_since
                    features['recency_score'] = np.exp(-days_since / 30)  # Decay over 30 days
                else:
                    features['days_since_last'] = 999
                    features['recency_score'] = 0
            
            # Engagement rate (orders / interactions)
            if features.get('order_count', 0) > 0 and features['total_interactions'] > 0:
                features['conversion_rate'] = features['order_count'] / features['total_interactions']
            else:
                features['conversion_rate'] = 0
            
            features_list.append(features)
        
        user_features = pd.DataFrame(features_list)
        user_features = user_features.fillna(0)
        
        # Normalize numerical features
        num_cols = ['total_interactions', 'unique_items', 'avg_price', 'max_price', 
                    'min_price', 'price_std', 'days_since_last', 'category_diversity',
                    'brand_diversity']
        
        for col in num_cols:
            if col in user_features.columns:
                scaler = MinMaxScaler()
                user_features[f'{col}_norm'] = scaler.fit_transform(
                    user_features[[col]].fillna(0)
                )
                self.scalers[f'user_{col}'] = scaler
        
        self.user_features = user_features.set_index('user_idx')
        logger.info(f"Computed features for {len(self.user_features)} users")
        
        return self.user_features
    
    def compute_item_features(
        self,
        items_df: pd.DataFrame,
        interactions_df: Optional[pd.DataFrame] = None,
        embeddings: Optional[np.ndarray] = None,
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Compute item-level features.
        
        Parameters
        ----------
        items_df : pd.DataFrame
            Item catalog.
        interactions_df : pd.DataFrame, optional
            Interactions for popularity features.
        embeddings : np.ndarray, optional
            Visual embeddings for items.
        reference_date : datetime, optional
            Reference date for freshness calculations.
        
        Returns
        -------
        pd.DataFrame
            Item features indexed by item_idx.
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        features = items_df.copy()
        
        # Freshness features
        if 'created_at' in features.columns:
            features['created_at'] = pd.to_datetime(features['created_at'])
            features['days_listed'] = (
                reference_date - features['created_at']
            ).dt.days.fillna(0)
            features['freshness_score'] = np.exp(-features['days_listed'] / 30)
        
        # Price tier
        price_percentiles = features['price'].quantile([0.25, 0.5, 0.75])
        features['price_tier'] = pd.cut(
            features['price'],
            bins=[0, price_percentiles[0.25], price_percentiles[0.5], 
                  price_percentiles[0.75], float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Popularity features (if interactions available)
        if interactions_df is not None:
            # Interaction counts
            interaction_counts = interactions_df.groupby('item_idx').agg({
                'interaction_strength': ['sum', 'count']
            }).reset_index()
            interaction_counts.columns = ['item_idx', 'total_strength', 'interaction_count']
            
            features = features.merge(interaction_counts, on='item_idx', how='left')
            features['total_strength'] = features['total_strength'].fillna(0)
            features['interaction_count'] = features['interaction_count'].fillna(0)
            
            # Popularity score (log-normalized)
            features['popularity_score'] = np.log1p(features['interaction_count'])
            
            # Unique users
            user_counts = interactions_df.groupby('item_idx')['user_idx'].nunique().reset_index()
            user_counts.columns = ['item_idx', 'unique_users']
            features = features.merge(user_counts, on='item_idx', how='left')
            features['unique_users'] = features['unique_users'].fillna(0)
        
        # Image count feature
        if 'num_images' in features.columns:
            features['has_multiple_images'] = (features['num_images'] > 1).astype(int)
        
        # Normalize numerical features
        num_cols = ['price', 'days_listed', 'interaction_count', 'popularity_score', 
                    'unique_users', 'num_images']
        
        for col in num_cols:
            if col in features.columns:
                scaler = MinMaxScaler()
                features[f'{col}_norm'] = scaler.fit_transform(
                    features[[col]].fillna(0)
                )
                self.scalers[f'item_{col}'] = scaler
        
        # Add embeddings if provided
        if embeddings is not None:
            embedding_dim = embeddings.shape[1]
            for i in range(embedding_dim):
                features[f'emb_{i}'] = embeddings[:, i]
        
        self.item_features = features.set_index('item_idx')
        logger.info(f"Computed features for {len(self.item_features)} items")
        
        return self.item_features
    
    def get_user_feature_vector(
        self,
        user_idx: int,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Get feature vector for a specific user."""
        if self.user_features is None:
            raise ValueError("User features not computed")
        
        if user_idx not in self.user_features.index:
            # Return zeros for unknown user
            return np.zeros(len(feature_names) if feature_names else 10)
        
        row = self.user_features.loc[user_idx]
        
        if feature_names:
            return row[feature_names].values
        
        return row.values
    
    def get_item_feature_vector(
        self,
        item_idx: int,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Get feature vector for a specific item."""
        if self.item_features is None:
            raise ValueError("Item features not computed")
        
        if item_idx not in self.item_features.index:
            return np.zeros(len(feature_names) if feature_names else 10)
        
        row = self.item_features.loc[item_idx]
        
        if feature_names:
            return row[feature_names].values
        
        return row.values
    
    def compute_pairwise_features(
        self,
        user_idx: int,
        item_idx: int,
    ) -> Dict[str, float]:
        """
        Compute features for a user-item pair.
        
        Useful for ranking/re-ranking models.
        """
        features = {}
        
        if self.user_features is not None and user_idx in self.user_features.index:
            user = self.user_features.loc[user_idx]
            
            if self.item_features is not None and item_idx in self.item_features.index:
                item = self.item_features.loc[item_idx]
                
                # Category match
                for i in range(1, 4):
                    cat_col = f'top_category_{i}'
                    if cat_col in user.index and 'category_idx' in item.index:
                        features[f'category_match_{i}'] = float(
                            user[cat_col] == item['category_idx']
                        )
                
                # Brand match
                for i in range(1, 3):
                    brand_col = f'top_brand_{i}'
                    if brand_col in user.index and 'brand_idx' in item.index:
                        features[f'brand_match_{i}'] = float(
                            user[brand_col] == item['brand_idx']
                        )
                
                # Price fit
                if 'avg_price' in user.index and 'price' in item.index:
                    price_diff = abs(item['price'] - user['avg_price'])
                    features['price_diff'] = price_diff
                    features['price_fit'] = np.exp(-price_diff / user.get('price_std', 100))
                
                # Condition match
                if 'preferred_condition' in user.index and 'condition_idx' in item.index:
                    features['condition_match'] = float(
                        user['preferred_condition'] == item['condition_idx']
                    )
        
        return features
    
    def save_features(self, output_dir: str):
        """Save computed features to disk."""
        from pathlib import Path
        import pickle
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.user_features is not None:
            self.user_features.to_parquet(output_path / "user_features.parquet")
        
        if self.item_features is not None:
            self.item_features.to_parquet(output_path / "item_features.parquet")
        
        with open(output_path / "scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        logger.info(f"Saved features to {output_dir}")
    
    def load_features(self, input_dir: str):
        """Load computed features from disk."""
        from pathlib import Path
        import pickle
        
        input_path = Path(input_dir)
        
        user_path = input_path / "user_features.parquet"
        if user_path.exists():
            self.user_features = pd.read_parquet(user_path)
        
        item_path = input_path / "item_features.parquet"
        if item_path.exists():
            self.item_features = pd.read_parquet(item_path)
        
        scalers_path = input_path / "scalers.pkl"
        if scalers_path.exists():
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
        
        logger.info(f"Loaded features from {input_dir}")
