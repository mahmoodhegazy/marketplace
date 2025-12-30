"""
FAISS-based Retrieval System
============================

Implements efficient approximate nearest neighbor search
using FAISS for recommendation retrieval.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import json
import faiss
from loguru import logger


class FAISSRetriever:
    """
    FAISS-based retriever for efficient similarity search.
    
    Supports multiple index types:
    - Flat: Exact search (small catalogs < 10K)
    - IVF: Inverted file index (10K-1M items)
    - HNSW: Hierarchical navigable small world (low latency)
    
    Parameters
    ----------
    dim : int
        Embedding dimension.
    index_type : str
        Type of index: 'flat', 'ivf', 'hnsw'.
    metric : str
        Distance metric: 'cosine', 'l2', 'ip' (inner product).
    nlist : int
        Number of clusters for IVF index.
    nprobe : int
        Number of clusters to search for IVF.
    m : int
        Number of connections per node for HNSW.
    ef_construction : int
        HNSW construction parameter.
    ef_search : int
        HNSW search parameter.
    """
    
    def __init__(
        self,
        dim: int,
        index_type: str = 'ivf',
        metric: str = 'cosine',
        nlist: int = 100,
        nprobe: int = 10,
        m: int = 32,
        ef_construction: int = 40,
        ef_search: int = 16,
    ):
        self.dim = dim
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        self.index = None
        self.item_ids: List[int] = []
        self.idx_to_item: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.is_trained = False
        
    def _create_index(self, n_items: int = None) -> faiss.Index:
        """Create FAISS index based on configuration."""
        
        # Choose metric
        if self.metric == 'cosine':
            # For cosine, normalize vectors and use inner product
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif self.metric == 'l2':
            metric_type = faiss.METRIC_L2
        else:  # inner product
            metric_type = faiss.METRIC_INNER_PRODUCT
        
        # Create index based on type
        if self.index_type == 'flat':
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(self.dim)
            else:
                index = faiss.IndexFlatL2(self.dim)
                
        elif self.index_type == 'ivf':
            # Adjust nlist based on data size
            actual_nlist = min(self.nlist, max(1, n_items // 10)) if n_items else self.nlist
            
            quantizer = faiss.IndexFlatIP(self.dim) if metric_type == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, actual_nlist, metric_type)
            
        elif self.index_type == 'hnsw':
            index = faiss.IndexHNSWFlat(self.dim, self.m, metric_type)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        return index
    
    def build(
        self,
        embeddings: np.ndarray,
        item_ids: List[int],
    ) -> None:
        """
        Build the index from embeddings.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Item embeddings, shape (n_items, dim).
        item_ids : List[int]
            Corresponding item IDs.
        """
        if len(embeddings) != len(item_ids):
            raise ValueError("Number of embeddings must match number of item IDs")
        
        n_items = len(item_ids)
        logger.info(f"Building FAISS index with {n_items} items, dim={self.dim}")
        
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Convert to float32
        embeddings = embeddings.astype(np.float32)
        
        # Create index
        self.index = self._create_index(n_items)
        
        # Train if needed (IVF requires training)
        if self.index_type == 'ivf' and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe
        
        # Add vectors
        self.index.add(embeddings)
        
        # Store mappings
        self.item_ids = list(item_ids)
        self.idx_to_item = {i: item_id for i, item_id in enumerate(item_ids)}
        self.item_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}
        
        self.is_trained = True
        logger.info(f"FAISS index built successfully: {self.index.ntotal} vectors")
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        exclude_ids: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[float]]:
        """
        Search for nearest neighbors.
        
        Parameters
        ----------
        query : np.ndarray
            Query embedding(s), shape (dim,) or (n_queries, dim).
        k : int
            Number of results to return.
        exclude_ids : List[int], optional
            Item IDs to exclude from results.
        
        Returns
        -------
        Tuple[List[int], List[float]]
            Item IDs and corresponding scores.
        """
        if not self.is_trained:
            raise RuntimeError("Index not built. Call build() first.")
        
        # Handle single query
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Normalize for cosine
        if self.metric == 'cosine':
            query = query / np.linalg.norm(query, axis=1, keepdims=True)
        
        query = query.astype(np.float32)
        
        # Search with extra items if excluding
        search_k = k + len(exclude_ids) if exclude_ids else k
        search_k = min(search_k, self.index.ntotal)
        
        distances, indices = self.index.search(query, search_k)
        
        # Convert to item IDs and filter
        results_ids = []
        results_scores = []
        
        exclude_set = set(exclude_ids) if exclude_ids else set()
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:  # FAISS returns -1 for not found
                continue
            
            item_id = self.idx_to_item[idx]
            
            if item_id in exclude_set:
                continue
            
            results_ids.append(item_id)
            results_scores.append(float(dist))
            
            if len(results_ids) >= k:
                break
        
        return results_ids, results_scores
    
    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for multiple queries.
        
        Parameters
        ----------
        queries : np.ndarray
            Query embeddings, shape (n_queries, dim).
        k : int
            Number of results per query.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Arrays of shape (n_queries, k) for indices and distances.
        """
        if not self.is_trained:
            raise RuntimeError("Index not built. Call build() first.")
        
        if self.metric == 'cosine':
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        
        queries = queries.astype(np.float32)
        
        distances, indices = self.index.search(queries, k)
        
        # Convert indices to item IDs
        item_ids = np.vectorize(lambda x: self.idx_to_item.get(x, -1))(indices)
        
        return item_ids, distances
    
    def add_items(
        self,
        embeddings: np.ndarray,
        item_ids: List[int],
    ) -> None:
        """
        Add new items to existing index (for incremental updates).
        
        Note: This only works well with Flat and HNSW indices.
        IVF indices may need periodic rebuilding.
        """
        if not self.is_trained:
            raise RuntimeError("Index not built. Call build() first.")
        
        if self.metric == 'cosine':
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        embeddings = embeddings.astype(np.float32)
        
        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        # Update mappings
        for i, item_id in enumerate(item_ids):
            idx = start_idx + i
            self.idx_to_item[idx] = item_id
            self.item_to_idx[item_id] = idx
            self.item_ids.append(item_id)
        
        logger.info(f"Added {len(item_ids)} items. Total: {self.index.ntotal}")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save index and mappings to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save metadata
        metadata = {
            'dim': self.dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'item_ids': self.item_ids,
        }
        
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Saved FAISS index to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FAISSRetriever':
        """Load index and mappings from disk."""
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create retriever
        retriever = cls(
            dim=metadata['dim'],
            index_type=metadata['index_type'],
            metric=metadata['metric'],
            nlist=metadata.get('nlist', 100),
            nprobe=metadata.get('nprobe', 10),
        )
        
        # Load FAISS index
        retriever.index = faiss.read_index(str(path / "index.faiss"))
        
        # Restore mappings
        retriever.item_ids = metadata['item_ids']
        retriever.idx_to_item = {i: item_id for i, item_id in enumerate(retriever.item_ids)}
        retriever.item_to_idx = {item_id: i for i, item_id in enumerate(retriever.item_ids)}
        retriever.is_trained = True
        
        logger.info(f"Loaded FAISS index from {path}: {retriever.index.ntotal} vectors")
        
        return retriever


class HybridRetriever:
    """
    Hybrid retriever combining multiple retrieval strategies.
    
    Combines:
    - Two-tower model (personalized)
    - Visual similarity (FashionCLIP)
    - Popularity-based fallback
    
    Parameters
    ----------
    two_tower_retriever : FAISSRetriever
        Retriever using two-tower item embeddings.
    visual_retriever : FAISSRetriever
        Retriever using visual (FashionCLIP) embeddings.
    popularity_items : List[int]
        Pre-computed list of popular items.
    cold_user_threshold : int
        Minimum interactions for personalized recommendations.
    """
    
    def __init__(
        self,
        two_tower_retriever: Optional[FAISSRetriever] = None,
        visual_retriever: Optional[FAISSRetriever] = None,
        popularity_items: Optional[List[int]] = None,
        cold_user_threshold: int = 5,
    ):
        self.two_tower_retriever = two_tower_retriever
        self.visual_retriever = visual_retriever
        self.popularity_items = popularity_items or []
        self.cold_user_threshold = cold_user_threshold
    
    def get_recommendations(
        self,
        user_id: int,
        user_embedding: Optional[np.ndarray] = None,
        user_interaction_count: int = 0,
        session_items: Optional[List[int]] = None,
        item_embeddings_map: Optional[Dict[int, np.ndarray]] = None,
        k: int = 20,
        exclude_items: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[float], str]:
        """
        Get recommendations using appropriate strategy.
        
        Parameters
        ----------
        user_id : int
            User ID.
        user_embedding : np.ndarray, optional
            User embedding from two-tower model.
        user_interaction_count : int
            Number of user interactions.
        session_items : List[int], optional
            Items viewed in current session.
        item_embeddings_map : Dict[int, np.ndarray], optional
            Visual embeddings for session items.
        k : int
            Number of recommendations.
        exclude_items : List[int], optional
            Items to exclude.
        
        Returns
        -------
        Tuple[List[int], List[float], str]
            Item IDs, scores, and strategy used.
        """
        exclude_set = set(exclude_items or [])
        
        # Strategy 1: Personalized (for warm/hot users)
        if (user_interaction_count >= self.cold_user_threshold 
            and user_embedding is not None 
            and self.two_tower_retriever is not None):
            
            item_ids, scores = self.two_tower_retriever.search(
                user_embedding,
                k=k,
                exclude_ids=list(exclude_set),
            )
            return item_ids, scores, 'personalized'
        
        # Strategy 2: Session-based visual similarity (for cold users with session)
        if (session_items 
            and item_embeddings_map 
            and self.visual_retriever is not None):
            
            # Get embeddings for recent session items
            recent_items = session_items[-3:]  # Use last 3 items
            valid_embeddings = []
            
            for item_id in recent_items:
                if item_id in item_embeddings_map:
                    valid_embeddings.append(item_embeddings_map[item_id])
            
            if valid_embeddings:
                # Average session embeddings
                query_embedding = np.mean(valid_embeddings, axis=0)
                
                # Exclude session items
                exclude_with_session = exclude_set | set(session_items)
                
                item_ids, scores = self.visual_retriever.search(
                    query_embedding,
                    k=k,
                    exclude_ids=list(exclude_with_session),
                )
                return item_ids, scores, 'session_visual'
        
        # Strategy 3: Popularity fallback
        if self.popularity_items:
            # Filter and return popular items
            filtered_items = [
                item_id for item_id in self.popularity_items
                if item_id not in exclude_set
            ][:k]
            
            # Assign decreasing scores based on rank
            scores = [1.0 / (i + 1) for i in range(len(filtered_items))]
            
            return filtered_items, scores, 'popularity'
        
        return [], [], 'none'
    
    def get_similar_items(
        self,
        item_id: int,
        item_embedding: np.ndarray,
        k: int = 10,
        use_visual: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """
        Get similar items based on visual or learned embeddings.
        
        Parameters
        ----------
        item_id : int
            Source item ID.
        item_embedding : np.ndarray
            Item embedding (visual or learned).
        k : int
            Number of similar items.
        use_visual : bool
            Use visual embeddings if True, else learned.
        
        Returns
        -------
        Tuple[List[int], List[float]]
            Similar item IDs and scores.
        """
        retriever = self.visual_retriever if use_visual else self.two_tower_retriever
        
        if retriever is None:
            return [], []
        
        return retriever.search(
            item_embedding,
            k=k + 1,  # +1 to exclude self
            exclude_ids=[item_id],
        )
