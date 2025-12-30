"""
Loss Functions for Recommendation Models
=========================================

Implements various loss functions for training recommendation models:
- InfoNCE (Contrastive Learning)
- BPR (Bayesian Personalized Ranking)
- Sampled Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) Loss.
    
    Also known as NT-Xent or Contrastive Loss.
    Commonly used in self-supervised learning and retrieval models.
    
    Parameters
    ----------
    temperature : float
        Temperature scaling factor. Lower values make the distribution sharper.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Parameters
        ----------
        user_emb : torch.Tensor
            User embeddings, shape (batch_size, dim).
        pos_item_emb : torch.Tensor
            Positive item embeddings, shape (batch_size, dim).
        neg_item_emb : torch.Tensor
            Negative item embeddings, shape (batch_size, num_negatives, dim).
        weights : torch.Tensor, optional
            Sample weights, shape (batch_size,).
        
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        batch_size = user_emb.size(0)
        
        # Positive scores: (batch_size,)
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)
        
        # Negative scores: (batch_size, num_negatives)
        neg_scores = torch.bmm(
            neg_item_emb,
            user_emb.unsqueeze(-1)
        ).squeeze(-1)
        
        # Concatenate: (batch_size, 1 + num_negatives)
        all_scores = torch.cat([
            pos_scores.unsqueeze(-1),
            neg_scores
        ], dim=-1) / self.temperature
        
        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=user_emb.device)
        
        # Cross entropy loss
        loss = F.cross_entropy(all_scores, labels, reduction='none')
        
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss.
    
    Pairwise loss that maximizes the difference between positive and negative
    item scores.
    
    Reference: BPR: Bayesian Personalized Ranking from Implicit Feedback
    """
    
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute BPR loss.
        
        Parameters
        ----------
        user_emb : torch.Tensor
            User embeddings, shape (batch_size, dim).
        pos_item_emb : torch.Tensor
            Positive item embeddings, shape (batch_size, dim).
        neg_item_emb : torch.Tensor
            Negative item embeddings, shape (batch_size, num_negatives, dim).
        weights : torch.Tensor, optional
            Sample weights, shape (batch_size,).
        
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # Positive scores: (batch_size,)
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)
        
        # Negative scores: (batch_size, num_negatives)
        neg_scores = torch.bmm(
            neg_item_emb,
            user_emb.unsqueeze(-1)
        ).squeeze(-1)
        
        # BPR loss for each negative
        # score_diff: (batch_size, num_negatives)
        score_diff = pos_scores.unsqueeze(-1) - neg_scores - self.margin
        
        # Soft plus (smooth approximation of hinge loss)
        loss = -F.logsigmoid(score_diff)
        
        # Average over negatives
        loss = loss.mean(dim=-1)  # (batch_size,)
        
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()


class SampledSoftmaxLoss(nn.Module):
    """
    Sampled Softmax Loss.
    
    Approximates full softmax by sampling a subset of negatives.
    More efficient for large item vocabularies.
    
    Reference: Candidate Sampling for Training Neural Language Models
    """
    
    def __init__(
        self,
        num_sampled: int = 100,
        num_classes: int = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_sampled = num_sampled
        self.num_classes = num_classes
        self.temperature = temperature
    
    def forward(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute sampled softmax loss.
        """
        batch_size = user_emb.size(0)
        num_negatives = neg_item_emb.size(1)
        
        # Positive logits: (batch_size,)
        pos_logits = torch.sum(user_emb * pos_item_emb, dim=-1) / self.temperature
        
        # Negative logits: (batch_size, num_negatives)
        neg_logits = torch.bmm(
            neg_item_emb,
            user_emb.unsqueeze(-1)
        ).squeeze(-1) / self.temperature
        
        # Log-sum-exp of negatives for partition function approximation
        neg_log_sum_exp = torch.logsumexp(neg_logits, dim=-1)
        
        # Loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = -pos + log(exp(pos) + sum(exp(neg)))
        # ≈ -pos + log_sum_exp(pos, neg)
        all_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
        loss = -pos_logits + torch.logsumexp(all_logits, dim=-1)
        
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()


class TripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss.
    
    Classic metric learning loss that enforces a margin between
    positive and negative pairs.
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute triplet margin loss.
        
        Parameters
        ----------
        anchor : torch.Tensor
            Anchor embeddings (users), shape (batch_size, dim).
        positive : torch.Tensor
            Positive embeddings, shape (batch_size, dim).
        negative : torch.Tensor
            Negative embeddings, shape (batch_size, [num_neg,] dim).
        """
        # Positive distance
        pos_dist = F.pairwise_distance(anchor, positive)
        
        # Handle multiple negatives
        if negative.dim() == 3:
            # Average over negatives
            neg_dists = []
            for i in range(negative.size(1)):
                neg_dists.append(
                    F.pairwise_distance(anchor, negative[:, i])
                )
            neg_dist = torch.stack(neg_dists, dim=1).min(dim=1)[0]
        else:
            neg_dist = F.pairwise_distance(anchor, negative)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining multiple objectives.
    
    Useful for jointly optimizing retrieval and auxiliary tasks
    (e.g., category prediction, price regression).
    """
    
    def __init__(
        self,
        task_weights: dict = None,
        learnable_weights: bool = False,
    ):
        super().__init__()
        
        self.task_weights = task_weights or {}
        self.learnable_weights = learnable_weights
        
        if learnable_weights:
            # Log-variance parameterization for uncertainty weighting
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(1))
                for task in self.task_weights
            })
    
    def forward(self, losses: dict) -> torch.Tensor:
        """
        Combine multiple task losses.
        
        Parameters
        ----------
        losses : dict
            Dictionary mapping task names to loss values.
        
        Returns
        -------
        torch.Tensor
            Combined scalar loss.
        """
        total_loss = 0.0
        
        for task, loss in losses.items():
            if self.learnable_weights and task in self.log_vars:
                # Uncertainty weighting
                precision = torch.exp(-self.log_vars[task])
                total_loss += precision * loss + self.log_vars[task]
            else:
                weight = self.task_weights.get(task, 1.0)
                total_loss += weight * loss
        
        return total_loss
