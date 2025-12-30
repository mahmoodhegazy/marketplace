"""
Caching Infrastructure
======================

Redis-based caching for recommendations and embeddings.
"""

import json
import pickle
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import hashlib
import numpy as np
from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Using in-memory cache fallback.")


@dataclass
class CacheConfig:
    """Cache configuration."""
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ttl: int = 3600  # Default TTL in seconds
    prefix: str = 'freak'


class RedisCache:
    """
    Redis cache for recommendations and features.
    
    Handles:
    - User recommendations cache
    - Item embeddings cache
    - Feature cache
    - Session data
    
    Parameters
    ----------
    config : CacheConfig
        Redis connection configuration.
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.client = None
        self._fallback_cache: Dict[str, Any] = {}
        
        self._connect()
    
    def _connect(self) -> None:
        """Establish Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using in-memory fallback")
            return
        
        try:
            self.client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=False,  # Handle bytes for pickle
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
        except redis.ConnectionError as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using in-memory fallback.")
            self.client = None
    
    def _make_key(self, namespace: str, key: str) -> str:
        """Create namespaced cache key."""
        return f"{self.config.prefix}:{namespace}:{key}"
    
    def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Parameters
        ----------
        namespace : str
            Cache namespace (e.g., 'recs', 'emb', 'features').
        key : str
            Cache key.
        
        Returns
        -------
        Any
            Cached value or None.
        """
        full_key = self._make_key(namespace, key)
        
        if self.client:
            try:
                value = self.client.get(full_key)
                if value:
                    return pickle.loads(value)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        return self._fallback_cache.get(full_key)
    
    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache.
        
        Parameters
        ----------
        namespace : str
            Cache namespace.
        key : str
            Cache key.
        value : Any
            Value to cache.
        ttl : int, optional
            Time to live in seconds.
        
        Returns
        -------
        bool
            Success status.
        """
        full_key = self._make_key(namespace, key)
        ttl = ttl or self.config.ttl
        
        try:
            serialized = pickle.dumps(value)
            
            if self.client:
                self.client.setex(full_key, ttl, serialized)
            else:
                self._fallback_cache[full_key] = value
            
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, namespace: str, key: str) -> bool:
        """Delete key from cache."""
        full_key = self._make_key(namespace, key)
        
        try:
            if self.client:
                self.client.delete(full_key)
            else:
                self._fallback_cache.pop(full_key, None)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def delete_pattern(self, namespace: str, pattern: str) -> int:
        """Delete keys matching pattern."""
        full_pattern = self._make_key(namespace, pattern)
        count = 0
        
        if self.client:
            try:
                for key in self.client.scan_iter(match=full_pattern):
                    self.client.delete(key)
                    count += 1
            except Exception as e:
                logger.error(f"Redis delete pattern error: {e}")
        else:
            keys_to_delete = [
                k for k in self._fallback_cache.keys()
                if k.startswith(full_pattern.replace('*', ''))
            ]
            for key in keys_to_delete:
                del self._fallback_cache[key]
                count += 1
        
        return count
    
    def exists(self, namespace: str, key: str) -> bool:
        """Check if key exists."""
        full_key = self._make_key(namespace, key)
        
        if self.client:
            try:
                return bool(self.client.exists(full_key))
            except Exception:
                pass
        
        return full_key in self._fallback_cache


class CacheManager:
    """
    High-level cache manager for recommendation system.
    
    Provides specialized methods for caching:
    - User recommendations
    - Item embeddings
    - User features
    - Popular items
    
    Parameters
    ----------
    cache : RedisCache
        Underlying cache instance.
    """
    
    def __init__(self, cache: RedisCache = None):
        self.cache = cache or RedisCache()
    
    # ==================== User Recommendations ====================
    
    def get_user_recommendations(
        self,
        user_id: int,
    ) -> Optional[List[int]]:
        """Get cached recommendations for user."""
        return self.cache.get('recs', str(user_id))
    
    def set_user_recommendations(
        self,
        user_id: int,
        item_ids: List[int],
        ttl: int = 3600,
    ) -> bool:
        """Cache recommendations for user."""
        return self.cache.set('recs', str(user_id), item_ids, ttl)
    
    def invalidate_user_recommendations(self, user_id: int) -> bool:
        """Invalidate cached recommendations for user."""
        return self.cache.delete('recs', str(user_id))
    
    def invalidate_all_recommendations(self) -> int:
        """Invalidate all cached recommendations."""
        return self.cache.delete_pattern('recs', '*')
    
    # ==================== Item Embeddings ====================
    
    def get_item_embedding(
        self,
        item_id: int,
        embedding_type: str = 'visual',
    ) -> Optional[np.ndarray]:
        """Get cached item embedding."""
        key = f"{embedding_type}:{item_id}"
        return self.cache.get('emb', key)
    
    def set_item_embedding(
        self,
        item_id: int,
        embedding: np.ndarray,
        embedding_type: str = 'visual',
        ttl: int = 86400,  # 24 hours
    ) -> bool:
        """Cache item embedding."""
        key = f"{embedding_type}:{item_id}"
        return self.cache.set('emb', key, embedding, ttl)
    
    def get_batch_embeddings(
        self,
        item_ids: List[int],
        embedding_type: str = 'visual',
    ) -> Dict[int, np.ndarray]:
        """Get batch of item embeddings."""
        results = {}
        for item_id in item_ids:
            emb = self.get_item_embedding(item_id, embedding_type)
            if emb is not None:
                results[item_id] = emb
        return results
    
    # ==================== User Features ====================
    
    def get_user_features(self, user_id: int) -> Optional[Dict]:
        """Get cached user features."""
        return self.cache.get('features', f'user:{user_id}')
    
    def set_user_features(
        self,
        user_id: int,
        features: Dict,
        ttl: int = 3600,
    ) -> bool:
        """Cache user features."""
        return self.cache.set('features', f'user:{user_id}', features, ttl)
    
    # ==================== Session Data ====================
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data."""
        return self.cache.get('session', session_id)
    
    def set_session(
        self,
        session_id: str,
        data: Dict,
        ttl: int = 1800,  # 30 minutes
    ) -> bool:
        """Set session data."""
        return self.cache.set('session', session_id, data, ttl)
    
    def add_session_item(
        self,
        session_id: str,
        item_id: int,
        max_items: int = 50,
    ) -> bool:
        """Add item to session history."""
        session = self.get_session(session_id) or {'items': []}
        
        items = session.get('items', [])
        if item_id not in items:
            items.append(item_id)
            items = items[-max_items:]  # Keep last N items
        
        session['items'] = items
        return self.set_session(session_id, session)
    
    def get_session_items(self, session_id: str) -> List[int]:
        """Get items from session."""
        session = self.get_session(session_id)
        return session.get('items', []) if session else []
    
    # ==================== Popular Items ====================
    
    def get_popular_items(
        self,
        category: Optional[str] = None,
    ) -> Optional[List[int]]:
        """Get cached popular items."""
        key = f"popular:{category or 'all'}"
        return self.cache.get('trending', key)
    
    def set_popular_items(
        self,
        item_ids: List[int],
        category: Optional[str] = None,
        ttl: int = 3600,
    ) -> bool:
        """Cache popular items."""
        key = f"popular:{category or 'all'}"
        return self.cache.set('trending', key, item_ids, ttl)
    
    # ==================== Similar Items ====================
    
    def get_similar_items(self, item_id: int) -> Optional[List[int]]:
        """Get cached similar items."""
        return self.cache.get('similar', str(item_id))
    
    def set_similar_items(
        self,
        item_id: int,
        similar_ids: List[int],
        ttl: int = 86400,
    ) -> bool:
        """Cache similar items."""
        return self.cache.set('similar', str(item_id), similar_ids, ttl)
    
    # ==================== Stats ====================
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        stats = {
            'backend': 'redis' if self.cache.client else 'memory',
            'connected': self.cache.client is not None,
        }
        
        if self.cache.client:
            try:
                info = self.cache.client.info('stats')
                stats.update({
                    'hits': info.get('keyspace_hits', 0),
                    'misses': info.get('keyspace_misses', 0),
                })
            except Exception:
                pass
        else:
            stats['memory_keys'] = len(self.cache._fallback_cache)
        
        return stats


def cache_key_for_query(
    user_id: int,
    filters: Optional[Dict] = None,
    page: int = 0,
) -> str:
    """
    Generate deterministic cache key for query.
    
    Parameters
    ----------
    user_id : int
        User ID.
    filters : Dict, optional
        Query filters (category, brand, etc.).
    page : int
        Pagination page.
    
    Returns
    -------
    str
        Hashed cache key.
    """
    key_parts = [str(user_id), str(page)]
    
    if filters:
        # Sort for deterministic hashing
        sorted_filters = json.dumps(filters, sort_keys=True)
        key_parts.append(sorted_filters)
    
    key_string = ':'.join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()
