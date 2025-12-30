"""
Freak AI - Fashion Resale Recommendation System
================================================

A complete AI platform for fashion resale marketplace recommendations,
featuring two-tower models, FashionCLIP embeddings, and MENA-specific
NLP capabilities.

Modules:
--------
- data: Data loading, preprocessing, and feature engineering
- embeddings: FashionCLIP visual embedding generation
- models: Two-tower and hybrid recommendation models
- training: Training pipelines and experiment tracking
- serving: API endpoints and inference
- evaluation: Metrics and evaluation utilities
- utils: Shared utilities and configurations

Usage:
------
    from freak_ai import TwoTowerModel, FashionCLIPEmbedder
    from freak_ai.data import DataProcessor
    
    # Load and process data
    processor = DataProcessor("configs/config.yaml")
    train_data, val_data, test_data = processor.prepare_datasets()
    
    # Generate embeddings
    embedder = FashionCLIPEmbedder()
    item_embeddings = embedder.embed_catalog(items_df)
    
    # Train model
    model = TwoTowerModel(config)
    model.fit(train_data, val_data)
    
    # Get recommendations
    recommendations = model.recommend(user_id, top_k=20)
"""

__version__ = "0.1.0"
__author__ = "Freak AI Team"

from .utils.config import Config
from .utils.logger import setup_logger

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "TwoTowerModel":
        from .models.two_tower import TwoTowerModel
        return TwoTowerModel
    elif name == "FashionCLIPEmbedder":
        from .embeddings.fashion_clip import FashionCLIPEmbedder
        return FashionCLIPEmbedder
    elif name == "DataProcessor":
        from .data.processor import DataProcessor
        return DataProcessor
    elif name == "RecommendationServer":
        from .serving.api import RecommendationServer
        return RecommendationServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Config",
    "setup_logger",
    "TwoTowerModel",
    "FashionCLIPEmbedder",
    "DataProcessor",
    "RecommendationServer",
]
