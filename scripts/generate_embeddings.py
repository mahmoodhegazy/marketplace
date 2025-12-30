#!/usr/bin/env python3
"""
Embedding Generation Script for Freak AI
=========================================

Generate FashionCLIP embeddings for the item catalog.

Usage:
    python scripts/generate_embeddings.py --items data/raw/items.csv --output data/embeddings/
    python scripts/generate_embeddings.py --items data/raw/items.csv --incremental
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.embeddings.fashion_clip import FashionCLIPEmbedder, EmbeddingCache


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate FashionCLIP embeddings")
    
    parser.add_argument(
        "--items", 
        type=str, 
        default="data/raw/items.csv",
        help="Path to items CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/embeddings",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to use (cuda/cpu). Auto-detect if not specified"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only generate embeddings for new items"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="patrickjohncyh/fashion-clip",
        choices=["patrickjohncyh/fashion-clip", "marqo-ai/marqo-FashionCLIP"],
        help="FashionCLIP model to use"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max workers for parallel image downloading"
    )
    parser.add_argument(
        "--url-column",
        type=str,
        default="image_urls",
        help="Column name containing image URLs"
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="item_id",
        help="Column name containing item IDs"
    )
    
    return parser.parse_args()


def load_items(items_path: str, id_column: str, url_column: str) -> pd.DataFrame:
    """Load and validate items data."""
    logger.info(f"Loading items from {items_path}")
    
    items_df = pd.read_csv(items_path)
    logger.info(f"Loaded {len(items_df)} items")
    
    # Validate required columns
    if id_column not in items_df.columns:
        raise ValueError(f"ID column '{id_column}' not found in items CSV")
    if url_column not in items_df.columns:
        raise ValueError(f"URL column '{url_column}' not found in items CSV")
    
    # Parse image URLs if they're in string format
    def parse_urls(url_str):
        if pd.isna(url_str):
            return []
        if isinstance(url_str, list):
            return url_str
        # Handle ["url1" "url2"] format
        import re
        urls = re.findall(r'https?://[^\s\"\'\]]+', str(url_str))
        return urls
    
    items_df['parsed_urls'] = items_df[url_column].apply(parse_urls)
    
    # Filter items with valid URLs
    items_with_urls = items_df[items_df['parsed_urls'].apply(len) > 0]
    logger.info(f"{len(items_with_urls)} items have valid image URLs")
    
    return items_with_urls


def get_first_url(urls_list):
    """Get the first URL from a list of URLs."""
    if isinstance(urls_list, list) and len(urls_list) > 0:
        return urls_list[0]
    return None


def main():
    """Main embedding generation pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_logger()
    logger.info("=" * 60)
    logger.info("Freak AI - Embedding Generation")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if args.device:
        device = args.device
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load items
    items_df = load_items(args.items, args.id_column, args.url_column)
    
    # Get image URLs (use first image for each item)
    items_df['primary_image_url'] = items_df['parsed_urls'].apply(get_first_url)
    items_df = items_df[items_df['primary_image_url'].notna()]
    
    item_ids = items_df[args.id_column].tolist()
    image_urls = items_df['primary_image_url'].tolist()
    
    logger.info(f"Processing {len(item_ids)} items with images")
    
    # Check for existing embeddings if incremental
    existing_cache = None
    if args.incremental:
        cache_path = output_dir / "embedding_cache.npz"
        if cache_path.exists():
            logger.info("Loading existing embedding cache...")
            existing_cache = EmbeddingCache.load(cache_path)
            
            # Find new items
            existing_ids = set(existing_cache.item_ids)
            new_mask = [item_id not in existing_ids for item_id in item_ids]
            
            new_item_ids = [item_ids[i] for i in range(len(item_ids)) if new_mask[i]]
            new_image_urls = [image_urls[i] for i in range(len(image_urls)) if new_mask[i]]
            
            logger.info(f"Found {len(new_item_ids)} new items to embed")
            
            if len(new_item_ids) == 0:
                logger.info("No new items to embed. Exiting.")
                return
            
            item_ids = new_item_ids
            image_urls = new_image_urls
    
    # Initialize embedder
    logger.info(f"Loading FashionCLIP model: {args.model}")
    embedder = FashionCLIPEmbedder(
        model_name=args.model,
        device=device
    )
    
    # Generate embeddings
    logger.info(f"Generating embeddings with batch size {args.batch_size}...")
    start_time = datetime.now()
    
    embeddings, valid_indices = embedder.embed_images(
        image_urls=image_urls,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        show_progress=True
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.1f}s")
    logger.info(f"Success rate: {len(embeddings)}/{len(image_urls)} ({100*len(embeddings)/len(image_urls):.1f}%)")
    
    # Get valid item IDs
    valid_item_ids = [item_ids[i] for i in valid_indices]
    
    # Create or update cache
    if args.incremental and existing_cache:
        logger.info("Merging with existing embeddings...")
        # Combine existing and new
        all_embeddings = np.vstack([existing_cache.embeddings, embeddings])
        all_item_ids = existing_cache.item_ids + valid_item_ids
        
        cache = EmbeddingCache(
            embeddings=all_embeddings,
            item_ids=all_item_ids
        )
    else:
        cache = EmbeddingCache(
            embeddings=embeddings,
            item_ids=valid_item_ids
        )
    
    # Save embeddings
    cache_path = output_dir / "embedding_cache.npz"
    cache.save(cache_path)
    logger.info(f"Saved embedding cache to {cache_path}")
    
    # Also save as numpy arrays for direct loading
    np.save(output_dir / "embeddings.npy", cache.embeddings)
    np.save(output_dir / "item_ids.npy", np.array(cache.item_ids))
    logger.info(f"Saved embeddings.npy and item_ids.npy")
    
    # Save metadata
    metadata = {
        "model": args.model,
        "embedding_dim": cache.embeddings.shape[1],
        "num_items": len(cache.item_ids),
        "generated_at": datetime.now().isoformat(),
        "device": device
    }
    
    import json
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata.json")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Embedding Generation Complete!")
    logger.info(f"  Total items: {len(cache.item_ids)}")
    logger.info(f"  Embedding dimension: {cache.embeddings.shape[1]}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
