"""
FastAPI Application for Recommendation Serving
==============================================

REST API endpoints for:
- Personalized recommendations
- Similar items
- Trending/popular items
- Health checks

Usage:
    uvicorn src.serving.api:app --reload --port 8000
"""

import os
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from .retriever import FAISSRetriever, HybridRetriever
from .cache import RedisCache, CacheManager, CacheConfig


# ==================== Pydantic Models ====================

class RecommendationRequest(BaseModel):
    """Request model for personalized recommendations."""
    user_id: int
    session_items: Optional[List[int]] = Field(default=None, description="Items viewed in session")
    exclude_items: Optional[List[int]] = Field(default=None, description="Items to exclude")
    category_filter: Optional[int] = Field(default=None, description="Filter by category")
    brand_filter: Optional[int] = Field(default=None, description="Filter by brand")
    k: int = Field(default=20, ge=1, le=100, description="Number of recommendations")


class SimilarItemsRequest(BaseModel):
    """Request model for similar items."""
    item_id: int
    k: int = Field(default=10, ge=1, le=50)
    use_visual: bool = Field(default=True, description="Use visual embeddings")


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    items: List[int]
    scores: List[float]
    strategy: str = Field(description="Strategy used: personalized, session_visual, popularity")
    latency_ms: float
    cached: bool = False


class SimilarItemsResponse(BaseModel):
    """Response model for similar items."""
    source_item: int
    similar_items: List[int]
    scores: List[float]
    latency_ms: float


class TrendingResponse(BaseModel):
    """Response model for trending items."""
    items: List[int]
    category: Optional[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    components: Dict[str, str]


class UserEmbeddingRequest(BaseModel):
    """Request for user embedding update."""
    user_id: int
    embedding: List[float]


# ==================== Application State ====================

class AppState:
    """Application state container."""
    
    def __init__(self):
        self.two_tower_retriever: Optional[FAISSRetriever] = None
        self.visual_retriever: Optional[FAISSRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.cache_manager: Optional[CacheManager] = None
        self.user_embeddings: Dict[int, np.ndarray] = {}
        self.item_visual_embeddings: Dict[int, np.ndarray] = {}
        self.user_interaction_counts: Dict[int, int] = {}
        self.popular_items: List[int] = []
        self.model_version: str = "0.1.0"
        self.loaded: bool = False


app_state = AppState()


# ==================== Lifecycle ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting recommendation API...")
    
    # Initialize cache
    cache_config = CacheConfig(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        ttl=int(os.getenv('CACHE_TTL', 3600)),
    )
    app_state.cache_manager = CacheManager(RedisCache(cache_config))
    
    # Load models if paths are configured
    models_path = Path(os.getenv('MODELS_PATH', 'models'))
    
    if (models_path / 'two_tower_index').exists():
        try:
            app_state.two_tower_retriever = FAISSRetriever.load(
                models_path / 'two_tower_index'
            )
            logger.info("Loaded two-tower retriever")
        except Exception as e:
            logger.warning(f"Failed to load two-tower retriever: {e}")
    
    if (models_path / 'visual_index').exists():
        try:
            app_state.visual_retriever = FAISSRetriever.load(
                models_path / 'visual_index'
            )
            logger.info("Loaded visual retriever")
        except Exception as e:
            logger.warning(f"Failed to load visual retriever: {e}")
    
    # Load embeddings if available
    if (models_path / 'user_embeddings.npy').exists():
        try:
            embeddings = np.load(models_path / 'user_embeddings.npy', allow_pickle=True).item()
            app_state.user_embeddings = embeddings
            logger.info(f"Loaded {len(embeddings)} user embeddings")
        except Exception as e:
            logger.warning(f"Failed to load user embeddings: {e}")
    
    if (models_path / 'visual_embeddings.npy').exists():
        try:
            embeddings = np.load(models_path / 'visual_embeddings.npy', allow_pickle=True).item()
            app_state.item_visual_embeddings = embeddings
            logger.info(f"Loaded {len(embeddings)} item visual embeddings")
        except Exception as e:
            logger.warning(f"Failed to load visual embeddings: {e}")
    
    # Initialize hybrid retriever
    app_state.hybrid_retriever = HybridRetriever(
        two_tower_retriever=app_state.two_tower_retriever,
        visual_retriever=app_state.visual_retriever,
        popularity_items=app_state.popular_items,
        cold_user_threshold=int(os.getenv('COLD_USER_THRESHOLD', 5)),
    )
    
    app_state.loaded = True
    logger.info("Recommendation API ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down recommendation API...")


# ==================== FastAPI App ====================

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    application = FastAPI(
        title="Freak AI Recommendation API",
        description="Fashion resale recommendation system",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return application


app = create_app()


# ==================== Dependencies ====================

def get_cache_manager() -> CacheManager:
    """Dependency for cache manager."""
    if app_state.cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    return app_state.cache_manager


def get_hybrid_retriever() -> HybridRetriever:
    """Dependency for hybrid retriever."""
    if app_state.hybrid_retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return app_state.hybrid_retriever


# ==================== Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {
        "cache": "healthy" if app_state.cache_manager else "unavailable",
        "two_tower": "loaded" if app_state.two_tower_retriever else "not_loaded",
        "visual": "loaded" if app_state.visual_retriever else "not_loaded",
    }
    
    return HealthResponse(
        status="healthy" if app_state.loaded else "initializing",
        version=app_state.model_version,
        timestamp=datetime.utcnow().isoformat(),
        components=components,
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    cache: CacheManager = Depends(get_cache_manager),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
):
    """
    Get personalized recommendations for a user.
    
    Uses three-tier fallback:
    1. Personalized (warm/hot users with embeddings)
    2. Session-based visual similarity (cold users with session)
    3. Popularity (complete cold start)
    """
    start_time = time.time()
    
    # Check cache first
    cached_recs = cache.get_user_recommendations(request.user_id)
    if cached_recs and not request.session_items:
        latency = (time.time() - start_time) * 1000
        return RecommendationResponse(
            items=cached_recs[:request.k],
            scores=[1.0 / (i + 1) for i in range(len(cached_recs[:request.k]))],
            strategy="cached",
            latency_ms=latency,
            cached=True,
        )
    
    # Get user embedding if available
    user_embedding = app_state.user_embeddings.get(request.user_id)
    user_interaction_count = app_state.user_interaction_counts.get(request.user_id, 0)
    
    # Get recommendations
    item_ids, scores, strategy = retriever.get_recommendations(
        user_id=request.user_id,
        user_embedding=user_embedding,
        user_interaction_count=user_interaction_count,
        session_items=request.session_items,
        item_embeddings_map=app_state.item_visual_embeddings,
        k=request.k,
        exclude_items=request.exclude_items,
    )
    
    # Apply filters if specified
    if request.category_filter or request.brand_filter:
        # Note: In production, this would query the item catalog
        # For now, we just return unfiltered results
        pass
    
    # Cache results (only for non-session based)
    if strategy == 'personalized' and not request.session_items:
        cache.set_user_recommendations(request.user_id, item_ids)
    
    latency = (time.time() - start_time) * 1000
    
    return RecommendationResponse(
        items=item_ids,
        scores=scores,
        strategy=strategy,
        latency_ms=latency,
        cached=False,
    )


@app.post("/similar", response_model=SimilarItemsResponse)
async def get_similar_items(
    request: SimilarItemsRequest,
    cache: CacheManager = Depends(get_cache_manager),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
):
    """Get visually similar items."""
    start_time = time.time()
    
    # Check cache
    cached = cache.get_similar_items(request.item_id)
    if cached:
        latency = (time.time() - start_time) * 1000
        return SimilarItemsResponse(
            source_item=request.item_id,
            similar_items=cached[:request.k],
            scores=[1.0 / (i + 1) for i in range(len(cached[:request.k]))],
            latency_ms=latency,
        )
    
    # Get item embedding
    if request.use_visual:
        item_embedding = app_state.item_visual_embeddings.get(request.item_id)
    else:
        # Would need learned item embeddings from two-tower model
        item_embedding = None
    
    if item_embedding is None:
        raise HTTPException(
            status_code=404,
            detail=f"Embedding not found for item {request.item_id}"
        )
    
    # Get similar items
    similar_ids, scores = retriever.get_similar_items(
        item_id=request.item_id,
        item_embedding=item_embedding,
        k=request.k,
        use_visual=request.use_visual,
    )
    
    # Cache results
    cache.set_similar_items(request.item_id, similar_ids)
    
    latency = (time.time() - start_time) * 1000
    
    return SimilarItemsResponse(
        source_item=request.item_id,
        similar_items=similar_ids,
        scores=scores,
        latency_ms=latency,
    )


@app.get("/trending", response_model=TrendingResponse)
async def get_trending_items(
    category: Optional[str] = None,
    k: int = Query(default=20, ge=1, le=100),
    cache: CacheManager = Depends(get_cache_manager),
):
    """Get trending/popular items."""
    # Check cache
    cached = cache.get_popular_items(category)
    if cached:
        return TrendingResponse(
            items=cached[:k],
            category=category,
        )
    
    # In production, compute from recent interactions
    items = app_state.popular_items[:k]
    
    if not items:
        # Fallback to any available items
        if app_state.item_visual_embeddings:
            items = list(app_state.item_visual_embeddings.keys())[:k]
    
    return TrendingResponse(
        items=items,
        category=category,
    )


@app.post("/session/add")
async def add_session_item(
    session_id: str,
    item_id: int,
    cache: CacheManager = Depends(get_cache_manager),
):
    """Add item to session history."""
    success = cache.add_session_item(session_id, item_id)
    return {"success": success, "session_id": session_id, "item_id": item_id}


@app.get("/session/{session_id}")
async def get_session(
    session_id: str,
    cache: CacheManager = Depends(get_cache_manager),
):
    """Get session data."""
    items = cache.get_session_items(session_id)
    return {"session_id": session_id, "items": items}


@app.post("/embeddings/user")
async def update_user_embedding(
    request: UserEmbeddingRequest,
    background_tasks: BackgroundTasks,
):
    """Update user embedding (called after model inference)."""
    embedding = np.array(request.embedding, dtype=np.float32)
    app_state.user_embeddings[request.user_id] = embedding
    
    # Invalidate cached recommendations in background
    def invalidate_cache():
        if app_state.cache_manager:
            app_state.cache_manager.invalidate_user_recommendations(request.user_id)
    
    background_tasks.add_task(invalidate_cache)
    
    return {"success": True, "user_id": request.user_id}


@app.get("/stats")
async def get_stats(
    cache: CacheManager = Depends(get_cache_manager),
):
    """Get system statistics."""
    return {
        "model_version": app_state.model_version,
        "num_user_embeddings": len(app_state.user_embeddings),
        "num_item_embeddings": len(app_state.item_visual_embeddings),
        "two_tower_items": (
            app_state.two_tower_retriever.index.ntotal
            if app_state.two_tower_retriever else 0
        ),
        "visual_items": (
            app_state.visual_retriever.index.ntotal
            if app_state.visual_retriever else 0
        ),
        "cache_stats": cache.get_cache_stats(),
    }


@app.post("/admin/reload")
async def reload_models(
    models_path: str = Query(default="models"),
):
    """Reload models (admin endpoint)."""
    path = Path(models_path)
    
    try:
        if (path / 'two_tower_index').exists():
            app_state.two_tower_retriever = FAISSRetriever.load(path / 'two_tower_index')
        
        if (path / 'visual_index').exists():
            app_state.visual_retriever = FAISSRetriever.load(path / 'visual_index')
        
        # Reinitialize hybrid retriever
        app_state.hybrid_retriever = HybridRetriever(
            two_tower_retriever=app_state.two_tower_retriever,
            visual_retriever=app_state.visual_retriever,
            popularity_items=app_state.popular_items,
        )
        
        return {"success": True, "message": "Models reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/clear-cache")
async def clear_cache(
    namespace: Optional[str] = None,
    cache: CacheManager = Depends(get_cache_manager),
):
    """Clear cache (admin endpoint)."""
    if namespace == 'recommendations':
        count = cache.invalidate_all_recommendations()
    else:
        count = cache.cache.delete_pattern('*', '*')
    
    return {"success": True, "cleared_keys": count}


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
