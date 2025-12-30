"""
Model Training for Freak AI
===========================

Training pipeline with MLflow integration, early stopping,
and comprehensive metric tracking.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm

from ..utils.logger import get_logger, log_metrics
from ..utils.config import Config, TrainingConfig
from ..models.two_tower import TwoTowerModel
from ..models.losses import InfoNCELoss

logger = get_logger(__name__)


class Trainer:
    """
    Trainer for Two-Tower recommendation models.
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - Early stopping
    - Learning rate scheduling
    - MLflow experiment tracking
    - Checkpoint management
    
    Example:
    --------
        trainer = Trainer(model, config)
        trainer.fit(train_loader, val_loader, epochs=50)
        trainer.save_checkpoint("best_model.pt")
    """
    
    def __init__(
        self,
        model: TwoTowerModel,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None,
        experiment_name: str = "freak-recommendations",
    ):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model : TwoTowerModel
            Model to train.
        config : TrainingConfig, optional
            Training configuration.
        device : str, optional
            Device to train on.
        experiment_name : str
            MLflow experiment name.
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_name = experiment_name
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = InfoNCELoss(temperature=0.1)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if self.device == "cuda" else None
        
        # MLflow tracking
        self.mlflow_run = None
        self._setup_mlflow()
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer.lower() == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01,
            )
        else:
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        return ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
            verbose=True,
        )
    
    def _setup_mlflow(self):
        """Set up MLflow experiment tracking."""
        try:
            import mlflow
            
            mlflow.set_experiment(self.experiment_name)
            self.mlflow_run = mlflow.start_run()
            
            # Log config
            mlflow.log_params({
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.epochs,
                'optimizer': self.config.optimizer,
                'num_negatives': self.config.num_negatives,
            })
            
            logger.info(f"MLflow tracking enabled: {mlflow.get_tracking_uri()}")
            
        except ImportError:
            logger.warning("MLflow not installed. Experiment tracking disabled.")
            self.mlflow_run = None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> float:
        """
        Train for one epoch.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        
        Returns
        -------
        float
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )
        
        for batch in progress_bar:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                # Mixed precision training
                with torch.amp.autocast('cuda'):
                    loss = self._compute_batch_loss(batch)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._compute_batch_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Validate model.
        
        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader.
        
        Returns
        -------
        Tuple[float, Dict]
            Validation loss and metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_user_emb = []
        all_pos_emb = []
        all_neg_emb = []
        
        for batch in val_loader:
            batch = self._move_batch_to_device(batch)
            
            user_emb, pos_item_emb, neg_item_emb = self.model(
                user_idx=batch['user_idx'],
                pos_item_idx=batch['pos_item_idx'],
                neg_item_indices=batch['neg_item_indices'],
                pos_item_features=batch['pos_item_features'],
                neg_item_features=batch['neg_item_features'],
            )
            
            loss = self.criterion(
                user_emb,
                pos_item_emb,
                neg_item_emb,
                weights=batch.get('interaction_weight'),
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect embeddings for metrics
            all_user_emb.append(user_emb.cpu())
            all_pos_emb.append(pos_item_emb.cpu())
            all_neg_emb.append(neg_item_emb.cpu())
        
        avg_loss = total_loss / num_batches
        
        # Compute additional metrics
        metrics = self._compute_validation_metrics(
            torch.cat(all_user_emb),
            torch.cat(all_pos_emb),
            torch.cat(all_neg_emb),
        )
        
        return avg_loss, metrics
    
    def _compute_batch_loss(self, batch: Dict) -> torch.Tensor:
        """Compute loss for a batch."""
        user_emb, pos_item_emb, neg_item_emb = self.model(
            user_idx=batch['user_idx'],
            pos_item_idx=batch['pos_item_idx'],
            neg_item_indices=batch['neg_item_indices'],
            pos_item_features=batch['pos_item_features'],
            neg_item_features=batch['neg_item_features'],
        )
        
        loss = self.criterion(
            user_emb,
            pos_item_emb,
            neg_item_emb,
            weights=batch.get('interaction_weight'),
        )
        
        return loss
    
    def _compute_validation_metrics(
        self,
        user_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute validation metrics."""
        # Positive scores
        pos_scores = torch.sum(user_emb * pos_emb, dim=-1)
        
        # Negative scores (average over negatives)
        neg_scores = torch.bmm(
            neg_emb,
            user_emb.unsqueeze(-1)
        ).squeeze(-1).mean(dim=-1)
        
        # Accuracy: positive > average negative
        accuracy = (pos_scores > neg_scores).float().mean().item()
        
        # Mean Reciprocal Rank approximation
        all_scores = torch.cat([
            pos_scores.unsqueeze(-1),
            neg_emb @ user_emb.unsqueeze(-1).squeeze(-1).T.diagonal().unsqueeze(-1)
        ], dim=-1)
        
        # Simplified MRR (position of positive among all)
        ranks = (all_scores[:, 1:] >= all_scores[:, 0:1]).sum(dim=-1) + 1
        mrr = (1.0 / ranks.float()).mean().item()
        
        return {
            'accuracy': accuracy,
            'mrr': mrr,
            'pos_score_mean': pos_scores.mean().item(),
            'neg_score_mean': neg_scores.mean().item(),
            'score_margin': (pos_scores - neg_scores).mean().item(),
        }
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                result[key] = value
        return result
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        callbacks: Optional[List] = None,
    ) -> Dict:
        """
        Train the model.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.
        epochs : int, optional
            Number of epochs. Uses config if not provided.
        callbacks : List, optional
            Training callbacks.
        
        Returns
        -------
        Dict
            Training history.
        """
        epochs = epochs or self.config.epochs
        callbacks = callbacks or []
        
        # Initialize early stopping
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"train_loss: {train_loss:.4f}, "
                f"val_loss: {val_loss:.4f}, "
                f"accuracy: {val_metrics['accuracy']:.4f}, "
                f"lr: {current_lr:.2e}"
            )
            
            # MLflow logging
            if self.mlflow_run is not None:
                import mlflow
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': current_lr,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }, step=epoch)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_metrics = val_metrics
                patience_counter = 0
                
                # Save best model
                if self.config.save_best_only:
                    self.save_checkpoint(
                        Path(self.config.checkpoint_dir) / "best_model.pt"
                    )
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Run callbacks
            for callback in callbacks:
                callback(self, epoch, train_loss, val_loss, val_metrics)
        
        # End MLflow run
        if self.mlflow_run is not None:
            import mlflow
            mlflow.end_run()
        
        logger.info(f"Training complete. Best val_loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'vocab_sizes': self.model.vocab_sizes,
            'config': self.model.config,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def export_model(self, path: str, format: str = 'torchscript'):
        """
        Export model for deployment.
        
        Parameters
        ----------
        path : str
            Output path.
        format : str
            Export format ('torchscript' or 'onnx').
        """
        self.model.eval()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'torchscript':
            # Trace user tower
            dummy_user = torch.zeros(1, dtype=torch.long, device=self.device)
            traced_user_tower = torch.jit.trace(
                self.model.user_tower,
                (dummy_user,)
            )
            traced_user_tower.save(str(path.parent / f"{path.stem}_user_tower.pt"))
            
            logger.info(f"User tower exported to {path.parent / f'{path.stem}_user_tower.pt'}")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
