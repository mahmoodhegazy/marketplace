"""Tests for model modules."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import UserTower, ItemTower, TwoTowerModel
from src.models.losses import InfoNCELoss, BPRLoss, TripletMarginLoss


class TestUserTower:
    """Tests for UserTower model."""
    
    def test_forward_shape(self, vocab_sizes, model_config):
        """Test UserTower output shape."""
        tower = UserTower(
            num_users=vocab_sizes['user'],
            num_categories=vocab_sizes['category'],
            num_brands=vocab_sizes['brand'],
            config=model_config.two_tower.user_tower,
            output_dim=model_config.two_tower.final_embedding_dim
        )
        
        batch_size = 4
        user_ids = torch.randint(0, vocab_sizes['user'], (batch_size,))
        
        output = tower(user_ids)
        
        assert output.shape == (batch_size, model_config.two_tower.final_embedding_dim)
    
    def test_forward_normalized(self, vocab_sizes, model_config):
        """Test UserTower output is L2 normalized."""
        tower = UserTower(
            num_users=vocab_sizes['user'],
            num_categories=vocab_sizes['category'],
            num_brands=vocab_sizes['brand'],
            config=model_config.two_tower.user_tower,
            output_dim=model_config.two_tower.final_embedding_dim
        )
        
        user_ids = torch.randint(0, vocab_sizes['user'], (4,))
        output = tower(user_ids)
        
        # Check L2 norm is approximately 1
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestItemTower:
    """Tests for ItemTower model."""
    
    def test_forward_shape(self, vocab_sizes, model_config, sample_embeddings):
        """Test ItemTower output shape."""
        tower = ItemTower(
            num_items=vocab_sizes['item'],
            num_categories=vocab_sizes['category'],
            num_brands=vocab_sizes['brand'],
            num_conditions=vocab_sizes['condition'],
            num_sizes=vocab_sizes['size'],
            config=model_config.two_tower.item_tower,
            output_dim=model_config.two_tower.final_embedding_dim,
            visual_embeddings=sample_embeddings
        )
        
        batch_size = 4
        item_ids = torch.randint(0, vocab_sizes['item'], (batch_size,))
        category_ids = torch.randint(0, vocab_sizes['category'], (batch_size,))
        brand_ids = torch.randint(0, vocab_sizes['brand'], (batch_size,))
        condition_ids = torch.randint(0, vocab_sizes['condition'], (batch_size,))
        size_ids = torch.randint(0, vocab_sizes['size'], (batch_size,))
        prices = torch.rand(batch_size, 1)
        
        output = tower(item_ids, category_ids, brand_ids, condition_ids, size_ids, prices)
        
        assert output.shape == (batch_size, model_config.two_tower.final_embedding_dim)
    
    def test_forward_without_visual(self, vocab_sizes, model_config):
        """Test ItemTower without visual embeddings."""
        # Modify config to disable visual features
        model_config.two_tower.item_tower.use_visual_features = False
        
        tower = ItemTower(
            num_items=vocab_sizes['item'],
            num_categories=vocab_sizes['category'],
            num_brands=vocab_sizes['brand'],
            num_conditions=vocab_sizes['condition'],
            num_sizes=vocab_sizes['size'],
            config=model_config.two_tower.item_tower,
            output_dim=model_config.two_tower.final_embedding_dim,
            visual_embeddings=None
        )
        
        batch_size = 4
        item_ids = torch.randint(0, vocab_sizes['item'], (batch_size,))
        category_ids = torch.randint(0, vocab_sizes['category'], (batch_size,))
        brand_ids = torch.randint(0, vocab_sizes['brand'], (batch_size,))
        condition_ids = torch.randint(0, vocab_sizes['condition'], (batch_size,))
        size_ids = torch.randint(0, vocab_sizes['size'], (batch_size,))
        prices = torch.rand(batch_size, 1)
        
        output = tower(item_ids, category_ids, brand_ids, condition_ids, size_ids, prices)
        
        assert output.shape == (batch_size, model_config.two_tower.final_embedding_dim)


class TestTwoTowerModel:
    """Tests for TwoTowerModel."""
    
    def test_initialization(self, vocab_sizes, model_config, sample_embeddings):
        """Test TwoTowerModel initialization."""
        model = TwoTowerModel(
            config=model_config,
            vocab_sizes=vocab_sizes,
            visual_embeddings=sample_embeddings
        )
        
        assert model.user_tower is not None
        assert model.item_tower is not None
    
    def test_forward_output(self, vocab_sizes, model_config, sample_embeddings):
        """Test TwoTowerModel forward pass."""
        model = TwoTowerModel(
            config=model_config,
            vocab_sizes=vocab_sizes,
            visual_embeddings=sample_embeddings
        )
        
        batch_size = 4
        batch = {
            'user_id': torch.randint(0, vocab_sizes['user'], (batch_size,)),
            'pos_item_id': torch.randint(0, vocab_sizes['item'], (batch_size,)),
            'pos_category_id': torch.randint(0, vocab_sizes['category'], (batch_size,)),
            'pos_brand_id': torch.randint(0, vocab_sizes['brand'], (batch_size,)),
            'pos_condition_id': torch.randint(0, vocab_sizes['condition'], (batch_size,)),
            'pos_size_id': torch.randint(0, vocab_sizes['size'], (batch_size,)),
            'pos_price': torch.rand(batch_size, 1),
            'neg_item_id': torch.randint(0, vocab_sizes['item'], (batch_size, 2)),
            'neg_category_id': torch.randint(0, vocab_sizes['category'], (batch_size, 2)),
            'neg_brand_id': torch.randint(0, vocab_sizes['brand'], (batch_size, 2)),
            'neg_condition_id': torch.randint(0, vocab_sizes['condition'], (batch_size, 2)),
            'neg_size_id': torch.randint(0, vocab_sizes['size'], (batch_size, 2)),
            'neg_price': torch.rand(batch_size, 2, 1),
        }
        
        user_emb, pos_emb, neg_embs = model(batch)
        
        assert user_emb.shape == (batch_size, model_config.two_tower.final_embedding_dim)
        assert pos_emb.shape == (batch_size, model_config.two_tower.final_embedding_dim)
        assert neg_embs.shape == (batch_size, 2, model_config.two_tower.final_embedding_dim)
    
    def test_generate_item_embeddings(self, vocab_sizes, model_config, sample_embeddings):
        """Test generating all item embeddings."""
        model = TwoTowerModel(
            config=model_config,
            vocab_sizes=vocab_sizes,
            visual_embeddings=sample_embeddings
        )
        
        # Create item features
        items_df = {
            'item_id': list(range(vocab_sizes['item'])),
            'category_id': [0, 1, 2, 0, 1],
            'brand_id': [0, 1, 2, 3, 0],
            'condition_id': [0, 1, 2, 0, 1],
            'size_id': [0, 1, 2, 3, 4],
            'price_normalized': [0.1, 0.2, 0.5, 0.8, 1.0]
        }
        
        embeddings = model.generate_all_item_embeddings(items_df)
        
        assert embeddings.shape == (vocab_sizes['item'], model_config.two_tower.final_embedding_dim)


class TestLossFunctions:
    """Tests for loss functions."""
    
    def test_infonce_loss_shape(self):
        """Test InfoNCE loss computation."""
        loss_fn = InfoNCELoss(temperature=0.1)
        
        batch_size = 4
        embedding_dim = 16
        num_negatives = 2
        
        user_emb = torch.randn(batch_size, embedding_dim)
        pos_emb = torch.randn(batch_size, embedding_dim)
        neg_embs = torch.randn(batch_size, num_negatives, embedding_dim)
        
        # Normalize
        user_emb = torch.nn.functional.normalize(user_emb, dim=1)
        pos_emb = torch.nn.functional.normalize(pos_emb, dim=1)
        neg_embs = torch.nn.functional.normalize(neg_embs, dim=2)
        
        loss = loss_fn(user_emb, pos_emb, neg_embs)
        
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0
    
    def test_bpr_loss(self):
        """Test BPR loss computation."""
        loss_fn = BPRLoss()
        
        batch_size = 4
        embedding_dim = 16
        
        user_emb = torch.randn(batch_size, embedding_dim)
        pos_emb = torch.randn(batch_size, embedding_dim)
        neg_emb = torch.randn(batch_size, embedding_dim)
        
        loss = loss_fn(user_emb, pos_emb, neg_emb)
        
        assert loss.ndim == 0
        assert loss.item() > 0
    
    def test_triplet_loss(self):
        """Test Triplet margin loss."""
        loss_fn = TripletMarginLoss(margin=0.5)
        
        batch_size = 4
        embedding_dim = 16
        
        anchor = torch.randn(batch_size, embedding_dim)
        positive = anchor + 0.1 * torch.randn(batch_size, embedding_dim)  # Similar
        negative = torch.randn(batch_size, embedding_dim)  # Different
        
        loss = loss_fn(anchor, positive, negative)
        
        assert loss.ndim == 0
        assert loss.item() >= 0
