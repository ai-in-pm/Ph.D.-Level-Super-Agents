"""
Expert vector management and pooling for Transformer2.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class ExpertVector:
    """Represents a task-specific expert vector."""
    task_id: str
    vector: torch.Tensor
    metadata: Dict
    performance_score: float = 0.0
    usage_count: int = 0
    last_updated: float = 0.0

class ExpertPool:
    """Manages a pool of expert vectors with dynamic updates."""
    
    def __init__(self, max_experts: int = 100):
        self.max_experts = max_experts
        self.experts: Dict[str, ExpertVector] = {}
        self.task_similarity_matrix: Dict[Tuple[str, str], float] = {}
        
    def add_expert(
        self,
        task_id: str,
        vector: torch.Tensor,
        metadata: Optional[Dict] = None
    ) -> ExpertVector:
        """Add new expert vector to pool."""
        if metadata is None:
            metadata = {}
            
        expert = ExpertVector(
            task_id=task_id,
            vector=vector,
            metadata=metadata,
            last_updated=float(torch.cuda.Event().elapsed_time())
        )
        
        self.experts[task_id] = expert
        self._update_similarity_matrix(task_id)
        
        # Prune if exceeded max size
        if len(self.experts) > self.max_experts:
            self._prune_experts()
            
        return expert
        
    def get_expert(self, task_id: str) -> Optional[ExpertVector]:
        """Retrieve expert vector by task ID."""
        expert = self.experts.get(task_id)
        if expert:
            expert.usage_count += 1
        return expert
        
    def update_expert(
        self,
        task_id: str,
        vector: Optional[torch.Tensor] = None,
        performance_score: Optional[float] = None,
        metadata_update: Optional[Dict] = None
    ):
        """Update existing expert vector."""
        if task_id not in self.experts:
            return
            
        expert = self.experts[task_id]
        if vector is not None:
            expert.vector = vector
        if performance_score is not None:
            expert.performance_score = performance_score
        if metadata_update:
            expert.metadata.update(metadata_update)
            
        expert.last_updated = float(torch.cuda.Event().elapsed_time())
        self._update_similarity_matrix(task_id)
        
    def find_similar_experts(
        self,
        task_id: str,
        threshold: float = 0.8,
        max_results: int = 5
    ) -> List[Tuple[str, float]]:
        """Find similar experts based on vector similarity."""
        if task_id not in self.experts:
            return []
            
        similarities = []
        for other_id in self.experts:
            if other_id != task_id:
                sim_key = tuple(sorted([task_id, other_id]))
                similarity = self.task_similarity_matrix.get(sim_key, 0.0)
                similarities.append((other_id, similarity))
                
        return sorted(
            [s for s in similarities if s[1] >= threshold],
            key=lambda x: x[1],
            reverse=True
        )[:max_results]
        
    def _update_similarity_matrix(self, task_id: str):
        """Update similarity matrix with new expert."""
        expert = self.experts[task_id]
        for other_id, other_expert in self.experts.items():
            if other_id != task_id:
                similarity = torch.cosine_similarity(
                    expert.vector.flatten(),
                    other_expert.vector.flatten(),
                    dim=0
                ).item()
                sim_key = tuple(sorted([task_id, other_id]))
                self.task_similarity_matrix[sim_key] = similarity
                
    def _prune_experts(self):
        """Remove least valuable experts when pool is full."""
        # Score experts based on performance, usage, and recency
        scores = {}
        current_time = float(torch.cuda.Event().elapsed_time())
        
        for task_id, expert in self.experts.items():
            time_factor = np.exp(-(current_time - expert.last_updated) / 1e6)
            score = (
                expert.performance_score * 0.4 +
                np.log1p(expert.usage_count) * 0.3 +
                time_factor * 0.3
            )
            scores[task_id] = score
            
        # Remove lowest scoring experts
        sorted_experts = sorted(scores.items(), key=lambda x: x[1])
        to_remove = len(self.experts) - self.max_experts
        
        for task_id, _ in sorted_experts[:to_remove]:
            del self.experts[task_id]
            # Clean up similarity matrix
            self.task_similarity_matrix = {
                k: v for k, v in self.task_similarity_matrix.items()
                if task_id not in k
            }
            
    def save_pool(self, path: str):
        """Save expert pool to disk."""
        torch.save({
            'experts': self.experts,
            'similarity_matrix': self.task_similarity_matrix
        }, path)
        
    def load_pool(self, path: str):
        """Load expert pool from disk."""
        data = torch.load(path)
        self.experts = data['experts']
        self.task_similarity_matrix = data['similarity_matrix']
