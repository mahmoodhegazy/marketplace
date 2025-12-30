"""
Two-Tower Recommendation Model
==============================

Implementation of the two-tower (dual encoder) architecture for
large-scale recommendation systems.

The model consists of:
1. User Tower: Encodes user features and history into embeddings
2. Item Tower: Encodes item features and visual embeddings

At inference time, item embeddings are pre-computed and indexed
for efficient approximate nearest neighbor retrieval.

References:
- YouTube Recommendations (2016): Deep Neural Networks for YouTube Recommendations
- Google Two-Tower (2019): Sampling-Bias-Corrected Neural Modeling for Large Corpus
- TensorFlow Recommenders: https://www.tensorflow.org/recommenders
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils.logger import get_logger
from ..utils.config import Config, TwoTowerConfig

logger = get_logger(__name__)


class UserTower(nn.Module):
    """
    User encoder tower.
    
    Encodes user ID and features into a dense embedding vector.
    
    Parameters
    ----------
    num_users : int
        Total number of users (vocabulary size).
    embedding_dim : int
        User ID embedding dimension.
    hidden_layers : List[int]
        Hidden layer dimensions.
    output_dim : int
        Final embedding dimension.
    dropout : float
        Dropout rate.
    num_categories : int, optional
        Number of categories for preference encoding.
    num_brands : int, optional
        Number of brands for preference encoding.
    """
    
    def __init__(
        self,
        num_users: int,
        embedding_dim: int = 64,
        hidden_layers: List[int] = [128, 64],
        output_dim: int = 32,
        dropout: float = 0.2,
        num_categories: int = 0,
        num_brands: int = 0,
    ):
        super().__init__()
        
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        
        # User ID embedding
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        
        # Optional category preference embedding
        self.num_categories = num_categories
        if num_categories > 0:
            self.category_embedding = nn.Embedding(
                num_embeddings=num_categories + 1,  # +1 for padding
                embedding_dim=16,
                padding_idx=0,
            )
        
        # Optional brand preference embedding
        self.num_brands = num_brands
        if num_brands > 0:
            self.brand_embedding = nn.Embedding(
                num_embeddings=num_brands + 1,
                embedding_dim=16,
                padding_idx=0,
            )
        
        # Calculate input dimension
        input_dim = embedding_dim
        if num_categories > 0:
            input_dim += 16 * 3  # Top 3 categories
        if num_brands > 0:
            input_dim += 16 * 2  # Top 2 brands
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Final projection
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        user_idx: torch.Tensor,
        category_prefs: Optional[torch.Tensor] = None,
        brand_prefs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        user_idx : torch.Tensor
            User indices, shape (batch_size,).
        category_prefs : torch.Tensor, optional
            Top category preferences, shape (batch_size, 3).
        brand_prefs : torch.Tensor, optional
            Top brand preferences, shape (batch_size, 2).
        
        Returns
        -------
        torch.Tensor
            User embeddings, shape (batch_size, output_dim).
        """
        # Get user ID embedding
        user_emb = self.user_embedding(user_idx)  # (batch, embedding_dim)
        
        embeddings = [user_emb]
        
        # Add category preference embeddings
        if self.num_categories > 0 and category_prefs is not None:
            cat_emb = self.category_embedding(category_prefs)  # (batch, 3, 16)
            cat_emb = cat_emb.view(cat_emb.size(0), -1)  # (batch, 48)
            embeddings.append(cat_emb)
        
        # Add brand preference embeddings
        if self.num_brands > 0 and brand_prefs is not None:
            brand_emb = self.brand_embedding(brand_prefs)  # (batch, 2, 16)
            brand_emb = brand_emb.view(brand_emb.size(0), -1)  # (batch, 32)
            embeddings.append(brand_emb)
        
        # Concatenate all embeddings
        x = torch.cat(embeddings, dim=-1)
        
        # Pass through MLP
        output = self.mlp(x)
        
        # L2 normalize
        output = F.normalize(output, p=2, dim=-1)
        
        return output


class ItemTower(nn.Module):
    """
    Item encoder tower.
    
    Encodes item features and visual embeddings into a dense vector.
    
    Parameters
    ----------
    num_items : int
        Total number of items.
    num_categories : int
        Number of categories.
    num_brands : int
        Number of brands.
    num_conditions : int
        Number of condition types.
    num_sizes : int
        Number of sizes.
    embedding_dim : int
        Embedding dimension for categorical features.
    hidden_layers : List[int]
        Hidden layer dimensions.
    output_dim : int
        Final embedding dimension.
    dropout : float
        Dropout rate.
    visual_embedding_dim : int
        Dimension of visual embeddings (512 for FashionCLIP).
    use_visual : bool
        Whether to use visual embeddings.
    """
    
    def __init__(
        self,
        num_items: int,
        num_categories: int,
        num_brands: int,
        num_conditions: int,
        num_sizes: int,
        embedding_dim: int = 64,
        hidden_layers: List[int] = [256, 128, 64],
        output_dim: int = 32,
        dropout: float = 0.2,
        visual_embedding_dim: int = 512,
        use_visual: bool = True,
    ):
        super().__init__()
        
        self.num_items = num_items
        self.use_visual = use_visual
        self.visual_embedding_dim = visual_embedding_dim
        
        # Item ID embedding
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        
        # Category embedding
        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories + 1,
            embedding_dim=32,
            padding_idx=0,
        )
        
        # Brand embedding
        self.brand_embedding = nn.Embedding(
            num_embeddings=num_brands + 1,
            embedding_dim=32,
            padding_idx=0,
        )
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(
            num_embeddings=num_conditions + 1,
            embedding_dim=16,
            padding_idx=0,
        )
        
        # Size embedding
        self.size_embedding = nn.Embedding(
            num_embeddings=num_sizes + 1,
            embedding_dim=16,
            padding_idx=0,
        )
        
        # Visual embedding projection
        if use_visual:
            self.visual_projection = nn.Sequential(
                nn.Linear(visual_embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
        
        # Calculate input dimension
        input_dim = embedding_dim + 32 + 32 + 16 + 16 + 1  # +1 for price
        if use_visual:
            input_dim += 64
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Final projection
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.category_embedding.weight, std=0.01)
        nn.init.normal_(self.brand_embedding.weight, std=0.01)
        
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        item_idx: torch.Tensor,
        category_idx: torch.Tensor,
        brand_idx: torch.Tensor,
        condition_idx: torch.Tensor,
        size_idx: torch.Tensor,
        price_normalized: torch.Tensor,
        visual_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        item_idx : torch.Tensor
            Item indices, shape (batch_size,).
        category_idx : torch.Tensor
            Category indices, shape (batch_size,).
        brand_idx : torch.Tensor
            Brand indices, shape (batch_size,).
        condition_idx : torch.Tensor
            Condition indices, shape (batch_size,).
        size_idx : torch.Tensor
            Size indices, shape (batch_size,).
        price_normalized : torch.Tensor
            Normalized prices, shape (batch_size,).
        visual_embedding : torch.Tensor, optional
            Visual embeddings, shape (batch_size, 512).
        
        Returns
        -------
        torch.Tensor
            Item embeddings, shape (batch_size, output_dim).
        """
        # Get categorical embeddings
        item_emb = self.item_embedding(item_idx)
        cat_emb = self.category_embedding(category_idx)
        brand_emb = self.brand_embedding(brand_idx)
        cond_emb = self.condition_embedding(condition_idx)
        size_emb = self.size_embedding(size_idx)
        
        # Ensure price is 2D
        if price_normalized.dim() == 1:
            price_normalized = price_normalized.unsqueeze(-1)
        
        embeddings = [item_emb, cat_emb, brand_emb, cond_emb, size_emb, price_normalized]
        
        # Add visual embedding
        if self.use_visual and visual_embedding is not None:
            visual_proj = self.visual_projection(visual_embedding)
            embeddings.append(visual_proj)
        
        # Concatenate all embeddings
        x = torch.cat(embeddings, dim=-1)
        
        # Pass through MLP
        output = self.mlp(x)
        
        # L2 normalize
        output = F.normalize(output, p=2, dim=-1)
        
        return output


class TwoTowerModel(nn.Module):
    """
    Two-Tower Recommendation Model.
    
    Combines user and item towers for collaborative filtering with
    content-based features.
    
    Example:
    --------
        model = TwoTowerModel(config, vocab_sizes)
        
        # Training
        user_emb = model.user_tower(user_idx)
        pos_item_emb = model.item_tower(pos_item_features)
        neg_item_emb = model.item_tower(neg_item_features)
        
        loss = model.compute_loss(user_emb, pos_item_emb, neg_item_emb)
        
        # Inference
        recommendations = model.recommend(user_idx, item_embeddings, top_k=20)
    """
    
    def __init__(
        self,
        config: Union[TwoTowerConfig, Dict],
        vocab_sizes: Dict[str, int],
        visual_embeddings: Optional[np.ndarray] = None,
    ):
        super().__init__()
        
        if isinstance(config, dict):
            # Convert dict to config
            self.config = TwoTowerConfig()
        else:
            self.config = config
        
        self.vocab_sizes = vocab_sizes
        
        # Store visual embeddings (for item tower)
        self.visual_embeddings = None
        if visual_embeddings is not None:
            self.register_buffer(
                'visual_embeddings',
                torch.tensor(visual_embeddings, dtype=torch.float32),
            )
        
        # Build towers
        self.user_tower = UserTower(
            num_users=vocab_sizes['user'],
            embedding_dim=self.config.user_tower.embedding_dim,
            hidden_layers=self.config.user_tower.hidden_layers,
            output_dim=self.config.final_embedding_dim,
            dropout=self.config.user_tower.dropout,
            num_categories=vocab_sizes.get('category', 0),
            num_brands=vocab_sizes.get('brand', 0),
        )
        
        self.item_tower = ItemTower(
            num_items=vocab_sizes['item'],
            num_categories=vocab_sizes.get('category', 1),
            num_brands=vocab_sizes.get('brand', 1),
            num_conditions=vocab_sizes.get('condition', 1),
            num_sizes=vocab_sizes.get('size', 1),
            embedding_dim=self.config.item_tower.embedding_dim,
            hidden_layers=self.config.item_tower.hidden_layers,
            output_dim=self.config.final_embedding_dim,
            dropout=self.config.item_tower.dropout,
            visual_embedding_dim=self.config.item_tower.visual_embedding_dim,
            use_visual=self.config.item_tower.use_visual_features,
        )
        
        # Temperature for softmax
        self.temperature = nn.Parameter(
            torch.tensor(self.config.temperature),
            requires_grad=False,
        )
        
        logger.info(f"TwoTowerModel initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(
        self,
        user_idx: torch.Tensor,
        pos_item_idx: torch.Tensor,
        neg_item_indices: torch.Tensor,
        pos_item_features: Dict[str, torch.Tensor],
        neg_item_features: Dict[str, torch.Tensor],
        interaction_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Returns user embeddings, positive item embeddings, and negative item embeddings.
        """
        batch_size = user_idx.size(0)
        num_negatives = neg_item_indices.size(1)
        
        # User embeddings
        user_emb = self.user_tower(user_idx)  # (batch, dim)
        
        # Get visual embeddings if available
        pos_visual = None
        neg_visual = None
        if self.visual_embeddings is not None:
            pos_visual = self.visual_embeddings[pos_item_idx]
            neg_visual = self.visual_embeddings[neg_item_indices.view(-1)].view(
                batch_size, num_negatives, -1
            )
        
        # Positive item embeddings
        pos_item_emb = self.item_tower(
            item_idx=pos_item_idx,
            category_idx=pos_item_features['category_idx'],
            brand_idx=pos_item_features['brand_idx'],
            condition_idx=pos_item_features['condition_idx'],
            size_idx=pos_item_features['size_idx'],
            price_normalized=pos_item_features['price_normalized'],
            visual_embedding=pos_visual,
        )  # (batch, dim)
        
        # Negative item embeddings
        # Reshape features for batch processing
        neg_item_emb_list = []
        for i in range(num_negatives):
            neg_features = {k: v[:, i] for k, v in neg_item_features.items()}
            neg_visual_i = neg_visual[:, i] if neg_visual is not None else None
            
            neg_emb = self.item_tower(
                item_idx=neg_item_indices[:, i],
                category_idx=neg_features['category_idx'],
                brand_idx=neg_features['brand_idx'],
                condition_idx=neg_features['condition_idx'],
                size_idx=neg_features['size_idx'],
                price_normalized=neg_features['price_normalized'],
                visual_embedding=neg_visual_i,
            )
            neg_item_emb_list.append(neg_emb)
        
        neg_item_emb = torch.stack(neg_item_emb_list, dim=1)  # (batch, num_neg, dim)
        
        return user_emb, pos_item_emb, neg_item_emb
    
    def compute_loss(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE (contrastive) loss.
        
        Parameters
        ----------
        user_emb : torch.Tensor
            User embeddings, shape (batch, dim).
        pos_item_emb : torch.Tensor
            Positive item embeddings, shape (batch, dim).
        neg_item_emb : torch.Tensor
            Negative item embeddings, shape (batch, num_neg, dim).
        weights : torch.Tensor, optional
            Sample weights, shape (batch,).
        
        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        batch_size = user_emb.size(0)
        
        # Positive scores
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)  # (batch,)
        
        # Negative scores
        neg_scores = torch.bmm(
            neg_item_emb, 
            user_emb.unsqueeze(-1)
        ).squeeze(-1)  # (batch, num_neg)
        
        # Concatenate all scores
        all_scores = torch.cat([
            pos_scores.unsqueeze(-1), 
            neg_scores
        ], dim=-1) / self.temperature  # (batch, 1 + num_neg)
        
        # Labels (positive is always index 0)
        labels = torch.zeros(batch_size, dtype=torch.long, device=user_emb.device)
        
        # Cross entropy loss
        loss = F.cross_entropy(all_scores, labels, reduction='none')
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()
    
    @torch.no_grad()
    def get_user_embeddings(
        self,
        user_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Get embeddings for users."""
        return self.user_tower(user_indices)
    
    @torch.no_grad()
    def get_item_embeddings(
        self,
        item_idx: torch.Tensor,
        category_idx: torch.Tensor,
        brand_idx: torch.Tensor,
        condition_idx: torch.Tensor,
        size_idx: torch.Tensor,
        price_normalized: torch.Tensor,
        visual_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get embeddings for items."""
        return self.item_tower(
            item_idx=item_idx,
            category_idx=category_idx,
            brand_idx=brand_idx,
            condition_idx=condition_idx,
            size_idx=size_idx,
            price_normalized=price_normalized,
            visual_embedding=visual_embedding,
        )
    
    @torch.no_grad()
    def generate_all_item_embeddings(
        self,
        items_df,
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        Generate embeddings for all items in catalog.
        
        Used for building the ANN index.
        """
        self.eval()
        device = next(self.parameters()).device
        
        all_embeddings = []
        num_items = len(items_df)
        
        for start_idx in range(0, num_items, batch_size):
            end_idx = min(start_idx + batch_size, num_items)
            batch = items_df.iloc[start_idx:end_idx]
            
            item_idx = torch.tensor(batch['item_idx'].values, dtype=torch.long, device=device)
            category_idx = torch.tensor(batch['category_idx'].values, dtype=torch.long, device=device)
            brand_idx = torch.tensor(batch['brand_idx'].values, dtype=torch.long, device=device)
            condition_idx = torch.tensor(batch['condition_idx'].values, dtype=torch.long, device=device)
            size_idx = torch.tensor(batch['size_idx'].values, dtype=torch.long, device=device)
            price_normalized = torch.tensor(batch['price_normalized'].values, dtype=torch.float32, device=device)
            
            visual_emb = None
            if self.visual_embeddings is not None:
                visual_emb = self.visual_embeddings[item_idx]
            
            embeddings = self.get_item_embeddings(
                item_idx=item_idx,
                category_idx=category_idx,
                brand_idx=brand_idx,
                condition_idx=condition_idx,
                size_idx=size_idx,
                price_normalized=price_normalized,
                visual_embedding=visual_emb,
            )
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    @torch.no_grad()
    def recommend(
        self,
        user_idx: int,
        item_embeddings: np.ndarray,
        top_k: int = 20,
        exclude_items: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate recommendations for a user.
        
        Parameters
        ----------
        user_idx : int
            User index.
        item_embeddings : np.ndarray
            Pre-computed item embeddings.
        top_k : int
            Number of recommendations.
        exclude_items : List[int], optional
            Item indices to exclude.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Top-k item indices and scores.
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Get user embedding
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
        user_emb = self.get_user_embeddings(user_tensor).cpu().numpy().flatten()
        
        # Compute scores
        scores = np.dot(item_embeddings, user_emb)
        
        # Exclude items if specified
        if exclude_items:
            scores[exclude_items] = -np.inf
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        
        return top_indices, top_scores
    
    def save(self, path: str):
        """Save model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_sizes': self.vocab_sizes,
            'config': self.config,
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'TwoTowerModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            config=checkpoint['config'],
            vocab_sizes=checkpoint['vocab_sizes'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from {path}")
        return model
