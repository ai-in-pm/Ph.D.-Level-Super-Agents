"""
Singular Value Fine-tuning (SVF) implementation for Transformer2.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SVFConfig:
    """Configuration for SVF optimization."""
    rank: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000

class SVFOptimizer:
    """Implements Singular Value Fine-tuning for weight matrices."""
    
    def __init__(self, config: SVFConfig):
        self.config = config
        self.expert_vectors: Dict[str, torch.Tensor] = {}
        self.cached_svd: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        
    def decompose_matrix(self, weight_matrix: torch.Tensor, name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform SVD decomposition of weight matrix."""
        U, S, V = torch.svd(weight_matrix)
        self.cached_svd[name] = (U, S, V)
        return U, S, V
        
    def create_expert_vector(self, task_id: str, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Create task-specific expert vector through SVD."""
        U, S, V = self.decompose_matrix(weight_matrix, task_id)
        expert_vector = torch.zeros_like(S)
        expert_vector[:self.config.rank] = S[:self.config.rank]
        self.expert_vectors[task_id] = expert_vector
        return expert_vector
        
    def apply_expert_vector(self, weight_matrix: torch.Tensor, task_id: str) -> torch.Tensor:
        """Apply expert vector to weight matrix."""
        if task_id not in self.cached_svd:
            self.decompose_matrix(weight_matrix, task_id)
            
        U, _, V = self.cached_svd[task_id]
        expert_vector = self.expert_vectors[task_id]
        
        # Reconstruct matrix with expert vector
        return torch.mm(torch.mm(U, torch.diag(expert_vector)), V.t())
        
    def optimize_expert_vector(
        self,
        task_id: str,
        loss: torch.Tensor,
        weight_matrix: torch.Tensor,
        learning_rate: Optional[float] = None
    ) -> torch.Tensor:
        """Optimize expert vector using gradients."""
        if learning_rate is None:
            learning_rate = self.config.learning_rate
            
        expert_vector = self.expert_vectors[task_id]
        grad = torch.autograd.grad(loss, expert_vector)[0]
        
        # Apply gradient update with weight decay
        expert_vector = expert_vector - learning_rate * (
            grad + self.config.weight_decay * expert_vector
        )
        
        self.expert_vectors[task_id] = expert_vector
        return self.apply_expert_vector(weight_matrix, task_id)
        
    def combine_expert_vectors(
        self,
        task_ids: List[str],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Combine multiple expert vectors with optional weights."""
        if weights is None:
            weights = [1.0 / len(task_ids)] * len(task_ids)
            
        combined_vector = torch.zeros_like(self.expert_vectors[task_ids[0]])
        for task_id, weight in zip(task_ids, weights):
            combined_vector += weight * self.expert_vectors[task_id]
            
        return combined_vector
        
    def save_expert_vectors(self, path: str):
        """Save expert vectors to disk."""
        torch.save(self.expert_vectors, path)
        
    def load_expert_vectors(self, path: str):
        """Load expert vectors from disk."""
        self.expert_vectors = torch.load(path)
        
    def reset_cache(self):
        """Clear cached SVD decompositions."""
        self.cached_svd.clear()
