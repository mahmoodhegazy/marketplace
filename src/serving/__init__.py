"""
Serving Module
==============

FastAPI endpoints and serving infrastructure for recommendations.
"""

from .api import app, create_app
from .retriever import FAISSRetriever, HybridRetriever
from .cache import RedisCache, CacheManager

__all__ = [
    "app",
    "create_app",
    "FAISSRetriever",
    "HybridRetriever",
    "RedisCache",
    "CacheManager",
]
