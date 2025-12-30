"""Tests for data processing module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.processor import DataProcessor


class TestDataProcessor:
    """Tests for DataProcessor class."""
    
    def test_init(self, temp_dir, sample_config_yaml):
        """Test DataProcessor initialization."""
        from src.utils.config import Config
        config = Config.from_yaml(sample_config_yaml)
        processor = DataProcessor(config)
        assert processor is not None
    
    def test_parse_image_urls_bracket_format(self):
        """Test parsing image URLs from bracket format."""
        url_str = '["https://example.com/1.jpg" "https://example.com/2.jpg"]'
        urls = DataProcessor._parse_image_urls(url_str)
        assert len(urls) == 2
        assert urls[0] == "https://example.com/1.jpg"
        assert urls[1] == "https://example.com/2.jpg"
    
    def test_parse_image_urls_empty(self):
        """Test parsing empty image URLs."""
        urls = DataProcessor._parse_image_urls("")
        assert urls == []
        
        urls = DataProcessor._parse_image_urls(None)
        assert urls == []
    
    def test_parse_image_urls_single(self):
        """Test parsing single image URL."""
        url_str = '["https://example.com/image.jpg"]'
        urls = DataProcessor._parse_image_urls(url_str)
        assert len(urls) == 1
    
    def test_build_interaction_matrix(self, sample_events_df):
        """Test building interaction matrix with weights."""
        event_weights = {'save': 1.0, 'cart': 3.0, 'order': 5.0}
        
        # Group and apply weights
        interactions = sample_events_df.copy()
        interactions['weight'] = interactions['event'].map(event_weights)
        interactions = interactions.groupby(['user_id', 'item_id'])['weight'].max().reset_index()
        
        assert 'weight' in interactions.columns
        assert interactions['weight'].max() == 5.0
        assert interactions['weight'].min() == 1.0
    
    def test_temporal_split(self, sample_interactions_df):
        """Test temporal data split."""
        df = sample_interactions_df.sort_values('timestamp')
        
        # Split by timestamp
        train_ratio = 0.7
        val_ratio = 0.1
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        assert len(train_df) + len(val_df) + len(test_df) == n
        assert train_df['timestamp'].max() <= val_df['timestamp'].min()
        assert val_df['timestamp'].max() <= test_df['timestamp'].min()


class TestFeatureEncoding:
    """Tests for feature encoding."""
    
    def test_label_encoding(self, sample_items_df):
        """Test label encoding for categorical features."""
        from sklearn.preprocessing import LabelEncoder
        
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(sample_items_df['category_id'])
        
        assert len(np.unique(encoded)) == len(sample_items_df['category_id'].unique())
        assert encoded.min() == 0
    
    def test_price_normalization(self, sample_items_df):
        """Test price normalization."""
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        prices = sample_items_df['price'].values.reshape(-1, 1)
        normalized = scaler.fit_transform(prices)
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
