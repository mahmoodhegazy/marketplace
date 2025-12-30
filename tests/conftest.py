"""Test fixtures for Freak AI tests."""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_items_df():
    """Create sample items DataFrame."""
    return pd.DataFrame({
        'item_id': [1, 2, 3, 4, 5],
        'category_id': [1, 2, 1, 3, 2],
        'brand_id': [10, 20, 10, 30, 20],
        'condition_id': [1, 2, 1, 2, 3],
        'size_id': [3, 4, 2, 5, 4],
        'price': [100.0, 200.0, 150.0, 300.0, 250.0],
        'created_at': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'
        ]),
        'category_name': ['Dresses', 'Tops', 'Dresses', 'Bottoms', 'Tops'],
        'brand_name': ['Zara', 'H&M', 'Zara', 'Mango', 'H&M'],
        'condition_name': ['New', 'Like New', 'New', 'Like New', 'Good'],
        'image_urls': [
            '["https://example.com/1.jpg"]',
            '["https://example.com/2.jpg"]',
            '["https://example.com/3.jpg"]',
            '["https://example.com/4.jpg"]',
            '["https://example.com/5.jpg"]'
        ]
    })


@pytest.fixture
def sample_events_df():
    """Create sample user events DataFrame."""
    return pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5],
        'item_id': [1, 2, 3, 1, 4, 5, 2, 3, 1, 5, 3, 4],
        'event': ['save', 'save', 'cart', 'order', 'save', 'cart', 'save', 'order', 'save', 'save', 'cart', 'order'],
        'timestamp': pd.to_datetime([
            '2024-01-10', '2024-01-11', '2024-01-12',
            '2024-01-10', '2024-01-13', '2024-01-14',
            '2024-01-11', '2024-01-12',
            '2024-01-13', '2024-01-14',
            '2024-01-15', '2024-01-16'
        ])
    })


@pytest.fixture
def sample_interactions_df():
    """Create sample interactions DataFrame with weights."""
    return pd.DataFrame({
        'user_id': [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
        'item_id': [0, 1, 2, 0, 3, 4, 1, 2, 0, 4, 2, 3],
        'weight': [1.0, 1.0, 3.0, 5.0, 1.0, 3.0, 1.0, 5.0, 1.0, 1.0, 3.0, 5.0],
        'timestamp': pd.to_datetime([
            '2024-01-10', '2024-01-11', '2024-01-12',
            '2024-01-10', '2024-01-13', '2024-01-14',
            '2024-01-11', '2024-01-12',
            '2024-01-13', '2024-01-14',
            '2024-01-15', '2024-01-16'
        ])
    })


@pytest.fixture
def sample_embeddings():
    """Create sample visual embeddings."""
    np.random.seed(42)
    num_items = 5
    embedding_dim = 512
    return torch.tensor(np.random.randn(num_items, embedding_dim), dtype=torch.float32)


@pytest.fixture
def vocab_sizes():
    """Create vocabulary sizes for model."""
    return {
        'user': 10,
        'item': 5,
        'category': 3,
        'brand': 4,
        'condition': 3,
        'size': 6
    }


@pytest.fixture
def model_config():
    """Create model configuration."""
    class MockConfig:
        class TwoTower:
            class UserTower:
                embedding_dim = 32
                hidden_layers = [64, 32]
                dropout = 0.1
                l2_reg = 1e-5
            
            class ItemTower:
                embedding_dim = 32
                hidden_layers = [128, 64, 32]
                dropout = 0.1
                l2_reg = 1e-5
                use_visual_features = True
                visual_embedding_dim = 512
            
            user_tower = UserTower()
            item_tower = ItemTower()
            final_embedding_dim = 16
            temperature = 0.1
        
        two_tower = TwoTower()
    
    return MockConfig()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_config_yaml(temp_dir):
    """Create a sample config file."""
    config_content = """
project:
  name: "test-freak-ai"
  version: "0.1.0"

data:
  raw_items_path: "data/raw/items.csv"
  raw_events_path: "data/raw/user_events.csv"
  processed_dir: "data/processed"
  embeddings_dir: "data/embeddings"
  event_weights:
    save: 1.0
    cart: 3.0
    order: 5.0
  test_size: 0.2
  validation_size: 0.1
  random_seed: 42

embeddings:
  model_name: "patrickjohncyh/fashion-clip"
  embedding_dim: 512
  batch_size: 32
  device: "cpu"

two_tower:
  user_tower:
    embedding_dim: 32
    hidden_layers: [64, 32]
    dropout: 0.1
    l2_reg: 1e-5
  item_tower:
    embedding_dim: 32
    hidden_layers: [128, 64, 32]
    dropout: 0.1
    l2_reg: 1e-5
    use_visual_features: true
    visual_embedding_dim: 512
  final_embedding_dim: 16
  temperature: 0.1

training:
  batch_size: 32
  epochs: 5
  learning_rate: 0.001
  optimizer: "adam"
  early_stopping_patience: 3
  num_negatives: 2
  hard_negative_ratio: 0.3
  checkpoint_dir: "checkpoints"

evaluation:
  metrics:
    - "precision_at_k"
    - "recall_at_k"
    - "ndcg_at_k"
  k_values: [5, 10]

serving:
  host: "0.0.0.0"
  port: 8000
  redis:
    host: "localhost"
    port: 6379
    db: 0
    cache_ttl: 3600
  faiss:
    index_type: "Flat"
    nlist: 10
    nprobe: 5
  default_top_k: 10
"""
    config_path = temp_dir / "config.yaml"
    config_path.write_text(config_content)
    return config_path
