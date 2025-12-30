#!/usr/bin/env python3
"""
Training Script for Freak AI Recommendation System
==================================================

End-to-end training pipeline:
1. Load and preprocess data
2. Generate/load visual embeddings
3. Create datasets
4. Train two-tower model
5. Build FAISS index
6. Evaluate model

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --data-path data/raw --epochs 50
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data.processor import DataProcessor
from src.data.dataset import InteractionDataset, collate_interactions
from src.data.features import FeatureEngineer
from src.embeddings.fashion_clip import FashionCLIPEmbedder, EmbeddingCache
from src.models.two_tower import TwoTowerModel
from src.training.trainer import Trainer
from src.serving.retriever import FAISSRetriever
from src.evaluation.metrics import RecommenderEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Freak recommendation model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data path from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (use cached)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (only build index)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name",
    )
    
    return parser.parse_args()


def load_data(config: Config) -> tuple:
    """Load and preprocess data."""
    logger.info("Loading data...")
    
    processor = DataProcessor(config)
    
    # Load raw data
    items_path = Path(config.data.raw_path) / "items.csv"
    events_path = Path(config.data.raw_path) / "events.csv"
    
    if not items_path.exists() or not events_path.exists():
        logger.error(f"Data files not found in {config.data.raw_path}")
        logger.info("Expected files: items.csv, events.csv")
        raise FileNotFoundError("Missing data files")
    
    items_df, events_df = processor.load_data(
        str(items_path),
        str(events_path),
    )
    
    logger.info(f"Loaded {len(items_df)} items and {len(events_df)} events")
    
    # Clean data
    items_df = processor.clean_items(items_df)
    events_df = processor.clean_events(events_df)
    
    # Build interactions
    interactions_df = processor.build_interactions(events_df)
    logger.info(f"Built {len(interactions_df)} interactions")
    
    # Encode features
    items_df = processor.encode_features(items_df)
    
    # Get vocab sizes
    vocab_sizes = processor.get_vocab_sizes(items_df, events_df)
    logger.info(f"Vocab sizes: {vocab_sizes}")
    
    return items_df, events_df, interactions_df, processor, vocab_sizes


def generate_embeddings(
    items_df: pd.DataFrame,
    config: Config,
    skip: bool = False,
) -> np.ndarray:
    """Generate or load visual embeddings."""
    embeddings_path = Path(config.data.embeddings_path) / "visual_embeddings.npy"
    cache_path = Path(config.data.embeddings_path) / "embedding_cache.pkl"
    
    if skip and embeddings_path.exists():
        logger.info(f"Loading cached embeddings from {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True).item()
        return data['embeddings'], data['item_ids']
    
    logger.info("Generating visual embeddings...")
    
    embedder = FashionCLIPEmbedder(
        model_name=config.embedding.model_name,
        device=config.training.device,
    )
    
    # Get image URLs
    item_ids = items_df['item_id'].tolist()
    image_urls = items_df['first_image_url'].tolist()
    
    # Filter items with valid URLs
    valid_items = []
    valid_urls = []
    for item_id, url in zip(item_ids, image_urls):
        if url and isinstance(url, str) and url.startswith('http'):
            valid_items.append(item_id)
            valid_urls.append(url)
    
    logger.info(f"Generating embeddings for {len(valid_urls)} items with valid URLs")
    
    # Load existing cache if available
    cache = None
    if cache_path.exists():
        cache = EmbeddingCache.load(cache_path)
        logger.info(f"Loaded embedding cache with {len(cache.embeddings)} items")
    
    # Generate embeddings
    embeddings, successful_ids = embedder.embed_catalog(
        image_urls=valid_urls,
        item_ids=valid_items,
        batch_size=config.embedding.batch_size,
    )
    
    if len(embeddings) == 0:
        logger.error("No embeddings generated!")
        raise ValueError("Failed to generate any embeddings")
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Save embeddings
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, {
        'embeddings': embeddings,
        'item_ids': successful_ids,
    })
    
    logger.info(f"Saved embeddings to {embeddings_path}")
    
    return embeddings, successful_ids


def create_datasets(
    items_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    config: Config,
) -> tuple:
    """Create train/val/test datasets."""
    logger.info("Creating datasets...")
    
    # Temporal split
    interactions_df = interactions_df.sort_values('timestamp')
    
    n = len(interactions_df)
    train_end = int(n * (1 - config.data.test_size - config.data.val_size))
    val_end = int(n * (1 - config.data.test_size))
    
    train_interactions = interactions_df.iloc[:train_end]
    val_interactions = interactions_df.iloc[train_end:val_end]
    test_interactions = interactions_df.iloc[val_end:]
    
    logger.info(f"Split sizes - Train: {len(train_interactions)}, Val: {len(val_interactions)}, Test: {len(test_interactions)}")
    
    # Create datasets
    train_dataset = InteractionDataset(
        items_df=items_df,
        interactions_df=train_interactions,
        num_negatives=config.training.num_negatives,
        hard_negative_ratio=config.training.hard_negative_ratio,
    )
    
    val_dataset = InteractionDataset(
        items_df=items_df,
        interactions_df=val_interactions,
        num_negatives=config.training.num_negatives,
        hard_negative_ratio=0.0,  # Random negatives for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_interactions,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_interactions,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_interactions


def train_model(
    config: Config,
    vocab_sizes: dict,
    visual_embeddings: np.ndarray,
    embedding_item_ids: list,
    train_loader: DataLoader,
    val_loader: DataLoader,
    experiment_name: str = None,
) -> TwoTowerModel:
    """Train two-tower model."""
    logger.info("Initializing model...")
    
    # Create embedding lookup
    embedding_dim = visual_embeddings.shape[1] if len(visual_embeddings) > 0 else 512
    
    # Create full embedding matrix with zeros for items without embeddings
    num_items = vocab_sizes['num_items']
    full_embeddings = np.zeros((num_items, embedding_dim), dtype=np.float32)
    
    item_id_to_idx = {item_id: i for i, item_id in enumerate(embedding_item_ids)}
    for item_id, embedding in zip(embedding_item_ids, visual_embeddings):
        if item_id < num_items:
            full_embeddings[item_id] = embedding
    
    # Initialize model
    model = TwoTowerModel(
        config=config,
        vocab_sizes=vocab_sizes,
        visual_embeddings=torch.tensor(full_embeddings),
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=config,
        experiment_name=experiment_name,
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    # Save final model
    output_dir = Path(config.serving.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.save_checkpoint(output_dir / "final_model.pt")
    model.save(output_dir / "model.pt")
    
    # Export for serving
    trainer.export_model(output_dir / "model_traced.pt")
    
    logger.info(f"Model saved to {output_dir}")
    
    return model


def build_index(
    model: TwoTowerModel,
    items_df: pd.DataFrame,
    visual_embeddings: np.ndarray,
    embedding_item_ids: list,
    config: Config,
) -> None:
    """Build FAISS indices for serving."""
    logger.info("Building FAISS indices...")
    
    output_dir = Path(config.serving.model_path)
    
    # Build two-tower item embeddings index
    model.eval()
    with torch.no_grad():
        item_embeddings = model.generate_all_item_embeddings()
    
    item_embeddings_np = item_embeddings.cpu().numpy()
    item_ids = list(range(len(item_embeddings_np)))
    
    two_tower_retriever = FAISSRetriever(
        dim=config.two_tower.embedding_dim,
        index_type=config.serving.index_type,
        metric='cosine',
    )
    two_tower_retriever.build(item_embeddings_np, item_ids)
    two_tower_retriever.save(output_dir / "two_tower_index")
    
    logger.info(f"Built two-tower index with {len(item_ids)} items")
    
    # Build visual embeddings index
    visual_retriever = FAISSRetriever(
        dim=visual_embeddings.shape[1],
        index_type=config.serving.index_type,
        metric='cosine',
    )
    visual_retriever.build(visual_embeddings, embedding_item_ids)
    visual_retriever.save(output_dir / "visual_index")
    
    logger.info(f"Built visual index with {len(embedding_item_ids)} items")
    
    # Save embeddings for serving
    np.save(output_dir / "visual_embeddings.npy", {
        int(item_id): emb for item_id, emb in zip(embedding_item_ids, visual_embeddings)
    })
    
    logger.info("FAISS indices saved")


def evaluate_model(
    model: TwoTowerModel,
    test_interactions: pd.DataFrame,
    items_df: pd.DataFrame,
    config: Config,
) -> dict:
    """Evaluate model on test set."""
    logger.info("Evaluating model...")
    
    model.eval()
    
    # Generate all item embeddings
    with torch.no_grad():
        item_embeddings = model.generate_all_item_embeddings()
    
    # Group test interactions by user
    user_ground_truth = test_interactions.groupby('user_idx')['item_idx'].apply(set).to_dict()
    
    # Generate recommendations for each user
    recommendations = {}
    
    unique_users = test_interactions['user_idx'].unique()
    logger.info(f"Evaluating {len(unique_users)} users...")
    
    for user_idx in unique_users:
        user_idx_tensor = torch.tensor([user_idx])
        
        with torch.no_grad():
            recs = model.recommend(
                user_idx=user_idx_tensor,
                item_embeddings=item_embeddings,
                k=100,
            )
        
        recommendations[user_idx] = recs[0].tolist()
    
    # Compute metrics
    evaluator = RecommenderEvaluator(
        k_values=[5, 10, 20, 50],
        metrics=['precision', 'recall', 'ndcg', 'mrr', 'hit_rate'],
    )
    
    results = evaluator.evaluate_all(recommendations, user_ground_truth)
    
    # Log results
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info("=" * 50)
    
    for metric, value in sorted(results.items()):
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")
    
    # Save results
    output_dir = Path(config.serving.model_path)
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_logger(
        log_dir="logs",
        log_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    logger.info("=" * 60)
    logger.info("Freak AI Recommendation System - Training Pipeline")
    logger.info("=" * 60)
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Apply command line overrides
    if args.data_path:
        config.data.raw_path = args.data_path
    if args.output_dir:
        config.serving.model_path = args.output_dir
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.device:
        config.training.device = args.device
    
    logger.info(f"Config: {config}")
    
    try:
        # Step 1: Load data
        items_df, events_df, interactions_df, processor, vocab_sizes = load_data(config)
        
        # Step 2: Generate embeddings
        visual_embeddings, embedding_item_ids = generate_embeddings(
            items_df=items_df,
            config=config,
            skip=args.skip_embeddings,
        )
        
        # Step 3: Create datasets
        train_loader, val_loader, test_interactions = create_datasets(
            items_df=items_df,
            interactions_df=interactions_df,
            config=config,
        )
        
        if not args.skip_training:
            # Step 4: Train model
            model = train_model(
                config=config,
                vocab_sizes=vocab_sizes,
                visual_embeddings=visual_embeddings,
                embedding_item_ids=embedding_item_ids,
                train_loader=train_loader,
                val_loader=val_loader,
                experiment_name=args.experiment_name,
            )
        else:
            # Load existing model
            model_path = Path(config.serving.model_path) / "model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            model = TwoTowerModel.load(model_path)
            logger.info("Loaded existing model")
        
        # Step 5: Build FAISS indices
        build_index(
            model=model,
            items_df=items_df,
            visual_embeddings=visual_embeddings,
            embedding_item_ids=embedding_item_ids,
            config=config,
        )
        
        # Step 6: Evaluate
        results = evaluate_model(
            model=model,
            test_interactions=test_interactions,
            items_df=items_df,
            config=config,
        )
        
        logger.info("=" * 60)
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
