"""
Data Processing for Freak AI
=============================

Handles loading, cleaning, and preprocessing of item and user event data.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from ..utils.logger import get_logger, log_data_stats
from ..utils.config import Config, load_config

logger = get_logger(__name__)


class DataProcessor:
    """
    Main data processing class for Freak AI.
    
    Handles loading raw data, cleaning, feature engineering,
    and preparing train/validation/test splits.
    
    Example:
    --------
        processor = DataProcessor("configs/config.yaml")
        items_df, events_df = processor.load_data()
        train, val, test = processor.prepare_datasets()
    """
    
    def __init__(self, config: Optional[Union[str, Config]] = None):
        """
        Initialize the data processor.
        
        Parameters
        ----------
        config : str or Config, optional
            Configuration file path or Config object.
        """
        if isinstance(config, str):
            self.config = load_config(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            self.config = load_config()
        
        # Encoders and scalers
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.brand_encoder = LabelEncoder()
        self.condition_encoder = LabelEncoder()
        self.size_encoder = LabelEncoder()
        self.price_scaler = MinMaxScaler()
        
        # Data storage
        self.items_df: Optional[pd.DataFrame] = None
        self.events_df: Optional[pd.DataFrame] = None
        self.interactions_df: Optional[pd.DataFrame] = None
        
        # Vocabulary sizes (for embedding layers)
        self.vocab_sizes: Dict[str, int] = {}
    
    def load_data(
        self,
        items_path: Optional[str] = None,
        events_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw item and event data.
        
        Parameters
        ----------
        items_path : str, optional
            Path to items CSV. Uses config if not provided.
        events_path : str, optional
            Path to events CSV. Uses config if not provided.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Items and events DataFrames.
        """
        items_path = items_path or self.config.data.raw_items_path
        events_path = events_path or self.config.data.raw_events_path
        
        logger.info(f"Loading items from: {items_path}")
        logger.info(f"Loading events from: {events_path}")
        
        # Load items
        self.items_df = pd.read_csv(items_path)
        self._clean_items_df()
        log_data_stats("items", self.items_df.shape)
        
        # Load events
        self.events_df = pd.read_csv(events_path)
        self._clean_events_df()
        log_data_stats("events", self.events_df.shape)
        
        return self.items_df, self.events_df
    
    def _clean_items_df(self):
        """Clean and preprocess items DataFrame."""
        df = self.items_df
        
        # Parse item_id (remove commas if present)
        df['item_id'] = df['item_id'].astype(str).str.replace(',', '').astype(int)
        
        # Clean price (remove commas)
        df['price'] = df['price'].astype(str).str.replace(',', '').astype(float)
        
        # Handle missing brands
        df['brand_name'] = df['brand_name'].fillna('Unknown')
        df['custom_brand'] = df['custom_brand'].fillna('')
        
        # Combine brand columns
        df['brand_combined'] = df.apply(
            lambda x: x['brand_name'] if pd.notna(x['brand_name']) and x['brand_name'] != '' 
                      else (x['custom_brand'] if x['custom_brand'] else 'Unknown'),
            axis=1
        )
        
        # Parse image URLs
        df['image_urls'] = df['image_urls'].apply(self._parse_image_urls)
        df['primary_image'] = df['image_urls'].apply(
            lambda x: x[0] if x and len(x) > 0 else None
        )
        df['num_images'] = df['image_urls'].apply(lambda x: len(x) if x else 0)
        
        # Parse dates
        df['created_at'] = pd.to_datetime(df['created_at'], format='mixed')
        
        # Fill missing category/condition names
        df['category_name'] = df['category_name'].fillna('Other')
        df['condition_name'] = df['condition_name'].fillna('Unknown')
        
        self.items_df = df
        logger.info(f"Cleaned items: {len(df)} items, {df['brand_combined'].nunique()} brands, "
                   f"{df['category_name'].nunique()} categories")
    
    def _parse_image_urls(self, urls_str: str) -> List[str]:
        """Parse image URLs from string format."""
        if pd.isna(urls_str) or not urls_str:
            return []
        
        # Handle the specific format: ["url1" "url2" "url3"]
        urls_str = str(urls_str).strip()
        
        # Remove outer brackets if present
        if urls_str.startswith('[') and urls_str.endswith(']'):
            urls_str = urls_str[1:-1]
        
        # Find all URLs using regex
        url_pattern = r'https?://[^\s\"\']+'
        urls = re.findall(url_pattern, urls_str)
        
        return urls
    
    def _clean_events_df(self):
        """Clean and preprocess events DataFrame."""
        df = self.events_df
        
        # Parse IDs (remove commas if present)
        df['user_id'] = df['user_id'].astype(str).str.replace(',', '').astype(int)
        df['item_id'] = df['item_id'].astype(str).str.replace(',', '').astype(int)
        
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        
        # Add event weight based on configuration
        event_weights = self.config.data.event_weights
        df['weight'] = df['event'].map(event_weights).fillna(1.0)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        self.events_df = df
        logger.info(f"Cleaned events: {len(df)} events, {df['user_id'].nunique()} users, "
                   f"event types: {df['event'].value_counts().to_dict()}")
    
    def build_interactions(self) -> pd.DataFrame:
        """
        Build user-item interaction matrix from events.
        
        Aggregates multiple events between same user-item pair,
        weighting by event type.
        
        Returns
        -------
        pd.DataFrame
            Interaction data with user_id, item_id, and interaction strength.
        """
        if self.events_df is None:
            raise ValueError("Events data not loaded. Call load_data() first.")
        
        # Filter to items that exist in items_df
        valid_items = set(self.items_df['item_id'].unique())
        events_filtered = self.events_df[
            self.events_df['item_id'].isin(valid_items)
        ].copy()
        
        logger.info(f"Filtered to {len(events_filtered)} events with valid items")
        
        # Aggregate interactions
        interactions = events_filtered.groupby(['user_id', 'item_id']).agg({
            'weight': 'sum',
            'timestamp': 'max',
            'event': lambda x: list(x)
        }).reset_index()
        
        interactions.columns = ['user_id', 'item_id', 'interaction_strength', 
                               'last_interaction', 'event_types']
        
        # Normalize interaction strength
        interactions['interaction_strength'] = np.log1p(interactions['interaction_strength'])
        
        self.interactions_df = interactions
        log_data_stats("interactions", interactions.shape)
        
        return interactions
    
    def encode_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical features for model input.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Encoded items and interactions DataFrames.
        """
        if self.items_df is None or self.interactions_df is None:
            raise ValueError("Data not prepared. Call load_data() and build_interactions() first.")
        
        items = self.items_df.copy()
        interactions = self.interactions_df.copy()
        
        # Encode user IDs
        all_users = interactions['user_id'].unique()
        self.user_encoder.fit(all_users)
        interactions['user_idx'] = self.user_encoder.transform(interactions['user_id'])
        self.vocab_sizes['user'] = len(all_users)
        
        # Encode item IDs
        all_items = items['item_id'].unique()
        self.item_encoder.fit(all_items)
        items['item_idx'] = self.item_encoder.transform(items['item_id'])
        interactions['item_idx'] = interactions['item_id'].map(
            dict(zip(items['item_id'], items['item_idx']))
        )
        # Drop interactions with items not in catalog
        interactions = interactions.dropna(subset=['item_idx'])
        interactions['item_idx'] = interactions['item_idx'].astype(int)
        self.vocab_sizes['item'] = len(all_items)
        
        # Encode categorical features
        items['category_idx'] = self.category_encoder.fit_transform(
            items['category_name'].fillna('Unknown')
        )
        self.vocab_sizes['category'] = len(self.category_encoder.classes_)
        
        items['brand_idx'] = self.brand_encoder.fit_transform(
            items['brand_combined'].fillna('Unknown')
        )
        self.vocab_sizes['brand'] = len(self.brand_encoder.classes_)
        
        items['condition_idx'] = self.condition_encoder.fit_transform(
            items['condition_name'].fillna('Unknown')
        )
        self.vocab_sizes['condition'] = len(self.condition_encoder.classes_)
        
        # Handle size encoding (may have many unique values)
        items['size_id'] = items['size_id'].fillna(0).astype(int)
        self.size_encoder.fit(items['size_id'])
        items['size_idx'] = self.size_encoder.transform(items['size_id'])
        self.vocab_sizes['size'] = len(self.size_encoder.classes_)
        
        # Normalize price
        items['price_normalized'] = self.price_scaler.fit_transform(
            items[['price']].fillna(0)
        )
        
        # Calculate days since listing
        now = pd.Timestamp.now()
        items['days_listed'] = (now - items['created_at']).dt.days.fillna(0)
        items['days_listed_normalized'] = MinMaxScaler().fit_transform(
            items[['days_listed']]
        )
        
        self.items_df = items
        self.interactions_df = interactions
        
        logger.info(f"Vocabulary sizes: {self.vocab_sizes}")
        
        return items, interactions
    
    def prepare_datasets(
        self,
        temporal_split: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare train/validation/test datasets.
        
        Parameters
        ----------
        temporal_split : bool
            If True, split by time. If False, random split.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Train, validation, and test DataFrames.
        """
        if self.interactions_df is None:
            raise ValueError("Interactions not built. Call build_interactions() first.")
        
        interactions = self.interactions_df.copy()
        
        test_size = self.config.data.test_size
        val_size = self.config.data.validation_size
        random_seed = self.config.data.random_seed
        
        if temporal_split:
            # Sort by timestamp and split
            interactions = interactions.sort_values('last_interaction')
            
            n = len(interactions)
            train_end = int(n * (1 - test_size - val_size))
            val_end = int(n * (1 - test_size))
            
            train_df = interactions.iloc[:train_end]
            val_df = interactions.iloc[train_end:val_end]
            test_df = interactions.iloc[val_end:]
        else:
            # Random split
            train_df, temp_df = train_test_split(
                interactions, 
                test_size=test_size + val_size,
                random_state=random_seed
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=test_size / (test_size + val_size),
                random_state=random_seed
            )
        
        logger.info(f"Dataset splits - Train: {len(train_df)}, "
                   f"Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_user_history(self, user_id: int) -> pd.DataFrame:
        """Get interaction history for a specific user."""
        if self.interactions_df is None:
            return pd.DataFrame()
        
        return self.interactions_df[
            self.interactions_df['user_id'] == user_id
        ].sort_values('last_interaction', ascending=False)
    
    def get_item_features(self, item_id: int) -> Optional[Dict]:
        """Get features for a specific item."""
        if self.items_df is None:
            return None
        
        item = self.items_df[self.items_df['item_id'] == item_id]
        if len(item) == 0:
            return None
        
        return item.iloc[0].to_dict()
    
    def get_popular_items(self, n: int = 100) -> List[int]:
        """Get most popular items by interaction count."""
        if self.interactions_df is None:
            return []
        
        popular = self.interactions_df.groupby('item_id').agg({
            'interaction_strength': 'sum'
        }).sort_values('interaction_strength', ascending=False)
        
        return popular.head(n).index.tolist()
    
    def get_user_preferences(self, user_id: int) -> Dict:
        """
        Compute user preferences based on interaction history.
        
        Returns aggregated preferences for categories, brands, 
        price range, etc.
        """
        history = self.get_user_history(user_id)
        if len(history) == 0:
            return {}
        
        # Get interacted items
        item_ids = history['item_id'].tolist()
        items = self.items_df[self.items_df['item_id'].isin(item_ids)]
        
        # Compute preferences
        preferences = {
            'category_prefs': items['category_name'].value_counts().head(5).to_dict(),
            'brand_prefs': items['brand_combined'].value_counts().head(5).to_dict(),
            'condition_prefs': items['condition_name'].value_counts().to_dict(),
            'avg_price': items['price'].mean(),
            'price_range': (items['price'].min(), items['price'].max()),
            'num_interactions': len(history),
        }
        
        return preferences
    
    def save_processed_data(self, output_dir: Optional[str] = None):
        """Save processed data to disk."""
        output_dir = Path(output_dir or self.config.data.processed_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.items_df is not None:
            self.items_df.to_parquet(output_dir / "items_processed.parquet")
        
        if self.interactions_df is not None:
            self.interactions_df.to_parquet(output_dir / "interactions.parquet")
        
        # Save encoders
        import pickle
        encoders = {
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'category_encoder': self.category_encoder,
            'brand_encoder': self.brand_encoder,
            'condition_encoder': self.condition_encoder,
            'size_encoder': self.size_encoder,
            'price_scaler': self.price_scaler,
            'vocab_sizes': self.vocab_sizes,
        }
        with open(output_dir / "encoders.pkl", 'wb') as f:
            pickle.dump(encoders, f)
        
        logger.info(f"Saved processed data to {output_dir}")
    
    def load_processed_data(self, input_dir: Optional[str] = None):
        """Load processed data from disk."""
        input_dir = Path(input_dir or self.config.data.processed_dir)
        
        self.items_df = pd.read_parquet(input_dir / "items_processed.parquet")
        self.interactions_df = pd.read_parquet(input_dir / "interactions.parquet")
        
        import pickle
        with open(input_dir / "encoders.pkl", 'rb') as f:
            encoders = pickle.load(f)
        
        self.user_encoder = encoders['user_encoder']
        self.item_encoder = encoders['item_encoder']
        self.category_encoder = encoders['category_encoder']
        self.brand_encoder = encoders['brand_encoder']
        self.condition_encoder = encoders['condition_encoder']
        self.size_encoder = encoders['size_encoder']
        self.price_scaler = encoders['price_scaler']
        self.vocab_sizes = encoders['vocab_sizes']
        
        logger.info(f"Loaded processed data from {input_dir}")


def create_sample_data():
    """Create sample data files for testing."""
    # Sample items
    items_data = {
        'item_id': [3, 4, 5, 6, 7, 8],
        'category_id': [110, 135, 50, 135, 133, 168],
        'brand_id': [None, 33, 81, 17, 118, 118],
        'custom_brand': ['Anne Brooks', None, None, None, None, None],
        'condition_id': [3, 7, 11, 3, 3, 3],
        'size_id': [4, 33, 166, 33, 33, 133],
        'price': [400, 200, 10000, 250, 300, 350],
        'created_at': ['June 12, 2024, 7:33 PM'] * 6,
        'category_name': ['Cardigans', 'Crop tops', 'Shoulder bags', 'Crop tops', 'Vest tops', 'Other'],
        'brand_name': [None, 'Guess', 'Okhtein', 'H&M', 'Source Unknown', 'Source Unknown'],
        'condition_name': ['Lightly worn'] * 6,
        'image_urls': ['["https://example.com/1.jpg"]'] * 6,
    }
    
    # Sample events
    events_data = {
        'user_id': [100, 100, 100, 14563, 14563, 22908],
        'item_id': [3, 4, 5, 4, 5, 6],
        'event': ['save', 'save', 'cart', 'cart', 'order', 'order'],
        'timestamp': ['November 29, 2025, 6:37 AM'] * 6,
    }
    
    return pd.DataFrame(items_data), pd.DataFrame(events_data)
