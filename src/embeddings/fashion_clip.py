"""
FashionCLIP Embedding Generator
===============================

Generates visual embeddings for fashion items using FashionCLIP,
a CLIP model fine-tuned on 800K Farfetch fashion products.

References:
- https://github.com/patrickjohncyh/fashion-clip
- https://github.com/marqo-ai/marqo-FashionCLIP
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import io

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import httpx

from ..utils.logger import get_logger
from ..utils.config import Config, load_config

logger = get_logger(__name__)


class FashionCLIPEmbedder:
    """
    Generate fashion-specific visual embeddings using FashionCLIP.
    
    FashionCLIP is a CLIP model fine-tuned on 800K Farfetch products,
    providing superior fashion understanding compared to general CLIP.
    
    Example:
    --------
        embedder = FashionCLIPEmbedder()
        embeddings = embedder.embed_images(image_urls)
        embedder.embed_catalog(items_df, save_path="embeddings.npy")
    """
    
    def __init__(
        self,
        model_name: str = "patrickjohncyh/fashion-clip",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize FashionCLIP embedder.
        
        Parameters
        ----------
        model_name : str
            HuggingFace model name or path.
        device : str, optional
            Device to use ('cuda' or 'cpu'). Auto-detects if not provided.
        cache_dir : str, optional
            Directory for caching downloaded models.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        self.model = None
        self.processor = None
        self._loaded = False
        
        logger.info(f"FashionCLIPEmbedder initialized (device: {self.device})")
    
    def load_model(self):
        """Load the FashionCLIP model."""
        if self._loaded:
            return
        
        logger.info(f"Loading FashionCLIP model: {self.model_name}")
        
        try:
            # Try loading with transformers (standard HuggingFace approach)
            from transformers import CLIPModel, CLIPProcessor
            
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            ).to(self.device)
            
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            
            self.model.eval()
            self._loaded = True
            logger.info("FashionCLIP model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace model: {e}")
            logger.info("Attempting to load with open_clip...")
            
            try:
                import open_clip
                
                # Use Marqo's FashionCLIP (ViT-B-16)
                self.model, _, self.processor = open_clip.create_model_and_transforms(
                    'ViT-B-16',
                    pretrained='laion2b_s34b_b88k',  # Fallback to standard CLIP
                )
                self.model = self.model.to(self.device)
                self.model.eval()
                self._loaded = True
                self._using_open_clip = True
                logger.info("Loaded with open_clip")
                
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                raise RuntimeError(f"Could not load FashionCLIP model: {e}, {e2}")
    
    def _download_image(self, url: str, timeout: float = 10.0) -> Optional[Image.Image]:
        """Download image from URL."""
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                
                image = Image.open(io.BytesIO(response.content))
                return image.convert('RGB')
        except Exception as e:
            logger.debug(f"Failed to download image {url}: {e}")
            return None
    
    def _download_images_parallel(
        self,
        urls: List[str],
        max_workers: int = 8,
    ) -> List[Optional[Image.Image]]:
        """Download multiple images in parallel."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            images = list(executor.map(self._download_image, urls))
        return images
    
    @torch.no_grad()
    def embed_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Parameters
        ----------
        image : str or PIL.Image
            Image URL or PIL Image object.
        
        Returns
        -------
        np.ndarray
            512-dimensional embedding vector.
        """
        self.load_model()
        
        # Load image if URL provided
        if isinstance(image, str):
            image = self._download_image(image)
            if image is None:
                return np.zeros(512)
        
        # Process image
        if hasattr(self, '_using_open_clip') and self._using_open_clip:
            image_tensor = self.processor(image).unsqueeze(0).to(self.device)
            embedding = self.model.encode_image(image_tensor)
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embedding = self.model.get_image_features(**inputs)
        
        # Normalize
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy().flatten()
    
    @torch.no_grad()
    def embed_images(
        self,
        images: List[Union[str, Image.Image]],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple images.
        
        Parameters
        ----------
        images : list
            List of image URLs or PIL Images.
        batch_size : int
            Batch size for processing.
        show_progress : bool
            Whether to show progress bar.
        
        Returns
        -------
        np.ndarray
            Array of shape (n_images, 512).
        """
        self.load_model()
        
        embeddings = []
        
        # Process in batches
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")
        
        for i in iterator:
            batch_images = images[i:i + batch_size]
            
            # Download images if URLs
            if isinstance(batch_images[0], str):
                batch_images = self._download_images_parallel(batch_images)
            
            # Filter out failed downloads
            valid_images = []
            valid_indices = []
            for j, img in enumerate(batch_images):
                if img is not None:
                    valid_images.append(img)
                    valid_indices.append(j)
            
            if not valid_images:
                # All downloads failed, return zeros
                embeddings.extend([np.zeros(512) for _ in batch_images])
                continue
            
            # Process batch
            if hasattr(self, '_using_open_clip') and self._using_open_clip:
                batch_tensors = torch.stack([
                    self.processor(img) for img in valid_images
                ]).to(self.device)
                batch_embeddings = self.model.encode_image(batch_tensors)
            else:
                inputs = self.processor(images=valid_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                batch_embeddings = self.model.get_image_features(**inputs)
            
            # Normalize
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            batch_embeddings = batch_embeddings.cpu().numpy()
            
            # Reconstruct with zeros for failed downloads
            full_batch = np.zeros((len(batch_images), 512))
            for idx, emb_idx in enumerate(valid_indices):
                full_batch[emb_idx] = batch_embeddings[idx]
            
            embeddings.extend(full_batch)
        
        return np.array(embeddings)
    
    @torch.no_grad()
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate text embeddings (for text-to-image search).
        
        Parameters
        ----------
        texts : str or list
            Text query or list of queries.
        
        Returns
        -------
        np.ndarray
            Text embeddings.
        """
        self.load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        if hasattr(self, '_using_open_clip') and self._using_open_clip:
            import open_clip
            tokenized = open_clip.tokenize(texts).to(self.device)
            embeddings = self.model.encode_text(tokenized)
        else:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embeddings = self.model.get_text_features(**inputs)
        
        # Normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()
    
    def embed_catalog(
        self,
        items_df: pd.DataFrame,
        image_column: str = 'primary_image',
        batch_size: int = 32,
        save_path: Optional[str] = None,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate embeddings for entire item catalog.
        
        Parameters
        ----------
        items_df : pd.DataFrame
            Item catalog with image URLs.
        image_column : str
            Column containing image URLs.
        batch_size : int
            Batch size for processing.
        save_path : str, optional
            Path to save embeddings.
        show_progress : bool
            Whether to show progress bar.
        
        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            Embeddings array and updated DataFrame with embedding indices.
        """
        logger.info(f"Generating embeddings for {len(items_df)} items")
        
        # Get image URLs
        image_urls = items_df[image_column].tolist()
        
        # Generate embeddings
        embeddings = self.embed_images(
            image_urls,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        
        # Track which items have valid embeddings
        valid_mask = ~np.all(embeddings == 0, axis=1)
        logger.info(f"Generated embeddings: {valid_mask.sum()} valid, "
                   f"{(~valid_mask).sum()} failed")
        
        # Add embedding index to DataFrame
        items_df = items_df.copy()
        items_df['embedding_idx'] = range(len(items_df))
        items_df['has_embedding'] = valid_mask
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, embeddings)
            logger.info(f"Saved embeddings to {save_path}")
        
        return embeddings, items_df
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        catalog_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine similarity between query and catalog.
        
        Parameters
        ----------
        query_embedding : np.ndarray
            Query embedding (512,).
        catalog_embeddings : np.ndarray
            Catalog embeddings (n_items, 512).
        top_k : int
            Number of top results to return.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Top-k indices and similarity scores.
        """
        # Ensure normalized
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        catalog_norm = catalog_embeddings / (
            np.linalg.norm(catalog_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Compute similarities
        similarities = np.dot(catalog_norm, query_norm)
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def find_similar_items(
        self,
        query_image: Union[str, Image.Image],
        catalog_embeddings: np.ndarray,
        items_df: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Find similar items given a query image.
        
        Parameters
        ----------
        query_image : str or PIL.Image
            Query image URL or PIL Image.
        catalog_embeddings : np.ndarray
            Pre-computed catalog embeddings.
        items_df : pd.DataFrame
            Item catalog.
        top_k : int
            Number of results to return.
        
        Returns
        -------
        pd.DataFrame
            Top-k similar items with similarity scores.
        """
        # Get query embedding
        query_emb = self.embed_image(query_image)
        
        # Find similar
        top_indices, scores = self.compute_similarity(
            query_emb, catalog_embeddings, top_k=top_k
        )
        
        # Get item details
        results = items_df.iloc[top_indices].copy()
        results['similarity_score'] = scores
        
        return results
    
    def text_search(
        self,
        query: str,
        catalog_embeddings: np.ndarray,
        items_df: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Search items using text query.
        
        Parameters
        ----------
        query : str
            Text search query (e.g., "vintage red dress").
        catalog_embeddings : np.ndarray
            Pre-computed catalog embeddings.
        items_df : pd.DataFrame
            Item catalog.
        top_k : int
            Number of results to return.
        
        Returns
        -------
        pd.DataFrame
            Top-k matching items with similarity scores.
        """
        # Get text embedding
        text_emb = self.embed_text(query).flatten()
        
        # Find similar
        top_indices, scores = self.compute_similarity(
            text_emb, catalog_embeddings, top_k=top_k
        )
        
        # Get item details
        results = items_df.iloc[top_indices].copy()
        results['similarity_score'] = scores
        
        return results


class EmbeddingCache:
    """
    Cache for storing and retrieving embeddings.
    
    Supports incremental updates when new items are added.
    """
    
    def __init__(self, cache_dir: str = "data/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings: Optional[np.ndarray] = None
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
    
    def load(self, name: str = "catalog") -> bool:
        """Load cached embeddings."""
        emb_path = self.cache_dir / f"{name}_embeddings.npy"
        idx_path = self.cache_dir / f"{name}_index.npy"
        
        if emb_path.exists() and idx_path.exists():
            self.embeddings = np.load(emb_path)
            item_ids = np.load(idx_path)
            
            self.item_to_idx = {int(item_id): idx for idx, item_id in enumerate(item_ids)}
            self.idx_to_item = {idx: int(item_id) for idx, item_id in enumerate(item_ids)}
            
            logger.info(f"Loaded {len(self.embeddings)} cached embeddings")
            return True
        
        return False
    
    def save(self, name: str = "catalog"):
        """Save embeddings to cache."""
        if self.embeddings is None:
            return
        
        np.save(self.cache_dir / f"{name}_embeddings.npy", self.embeddings)
        
        item_ids = np.array([self.idx_to_item[i] for i in range(len(self.embeddings))])
        np.save(self.cache_dir / f"{name}_index.npy", item_ids)
        
        logger.info(f"Saved {len(self.embeddings)} embeddings to cache")
    
    def get_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """Get embedding for an item."""
        if self.embeddings is None or item_id not in self.item_to_idx:
            return None
        
        return self.embeddings[self.item_to_idx[item_id]]
    
    def add_embeddings(
        self,
        item_ids: List[int],
        embeddings: np.ndarray,
    ):
        """Add new embeddings to cache."""
        if self.embeddings is None:
            self.embeddings = embeddings
            for idx, item_id in enumerate(item_ids):
                self.item_to_idx[item_id] = idx
                self.idx_to_item[idx] = item_id
        else:
            # Append new embeddings
            start_idx = len(self.embeddings)
            self.embeddings = np.vstack([self.embeddings, embeddings])
            
            for i, item_id in enumerate(item_ids):
                idx = start_idx + i
                self.item_to_idx[item_id] = idx
                self.idx_to_item[idx] = item_id
    
    def get_missing_items(self, item_ids: List[int]) -> List[int]:
        """Get item IDs that don't have cached embeddings."""
        return [item_id for item_id in item_ids if item_id not in self.item_to_idx]
