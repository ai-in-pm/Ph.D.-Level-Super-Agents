"""
Two-pass inference mechanism for Transformer2.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from .adaptation import AdaptationManager
from .experts import ExpertPool
from .svf import SVFOptimizer

@dataclass
class InferenceConfig:
    """Configuration for two-pass inference."""
    first_pass_threshold: float = 0.7
    max_expert_combinations: int = 3
    inference_temperature: float = 0.8
    cache_size: int = 1000

class TwoPassInference:
    """Implements two-pass inference mechanism."""
    
    def __init__(
        self,
        config: InferenceConfig,
        adaptation_manager: AdaptationManager,
        expert_pool: ExpertPool,
        svf_optimizer: SVFOptimizer
    ):
        self.config = config
        self.adaptation_manager = adaptation_manager
        self.expert_pool = expert_pool
        self.svf_optimizer = svf_optimizer
        self.cache: Dict[str, Dict[str, Any]] = {}
        
    async def first_pass(
        self,
        task_input: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        First pass: Identify task properties and select relevant experts.
        """
        # Check cache
        cache_key = str(task_input)
        if cache_key in self.cache:
            return (
                self.cache[cache_key]['task_types'],
                self.cache[cache_key]['confidences']
            )
            
        # Classify task
        task_desc = task_input.get('description', '')
        classifications = self.adaptation_manager.classify_task(task_desc)
        
        # Filter by confidence threshold
        task_types = [
            t_type for t_type, conf in classifications.items()
            if conf >= self.config.first_pass_threshold
        ]
        
        # Cache results
        if len(self.cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[cache_key] = {
            'task_types': task_types,
            'confidences': classifications,
            'timestamp': float(torch.cuda.Event().elapsed_time())
        }
        
        return task_types, classifications
        
    async def second_pass(
        self,
        task_input: Dict[str, Any],
        task_types: List[str],
        confidences: Dict[str, float]
    ) -> Tuple[torch.Tensor, str]:
        """
        Second pass: Use expert vectors to adjust model weights.
        """
        # Get expert combinations
        expert_combinations = []
        for task_type in task_types:
            similar_experts = self.expert_pool.find_similar_experts(
                task_type,
                threshold=self.config.first_pass_threshold,
                max_results=2
            )
            expert_combinations.extend([
                (expert_id, conf * confidences[task_type])
                for expert_id, conf in similar_experts
            ])
            
        # Sort and limit combinations
        expert_combinations.sort(key=lambda x: x[1], reverse=True)
        expert_combinations = expert_combinations[:self.config.max_expert_combinations]
        
        # Combine experts
        if expert_combinations:
            combined_expert = self.svf_optimizer.combine_expert_vectors(
                [eid for eid, _ in expert_combinations],
                weights=[w for _, w in expert_combinations]
            )
        else:
            # Fallback to default expert if no combinations found
            combined_expert = None
            
        # Generate task-specific prompt
        prompt = self.adaptation_manager.generate_prompt(
            task_types[0] if task_types else "default",
            task_input.get('description', '')
        )
        
        return combined_expert, prompt
        
    async def infer(
        self,
        task_input: Dict[str, Any]
    ) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
        """
        Perform complete two-pass inference.
        """
        # First pass
        task_types, confidences = await self.first_pass(task_input)
        
        # Second pass
        expert_vector, prompt = await self.second_pass(
            task_input,
            task_types,
            confidences
        )
        
        # Prepare metadata
        metadata = {
            'task_types': task_types,
            'confidences': confidences,
            'timestamp': float(torch.cuda.Event().elapsed_time())
        }
        
        return expert_vector, prompt, metadata
        
    def update_cache(self, task_input: Dict[str, Any], results: Dict[str, Any]):
        """Update cache with inference results."""
        cache_key = str(task_input)
        if cache_key in self.cache:
            self.cache[cache_key].update(results)
            
    def clear_cache(self):
        """Clear inference cache."""
        self.cache.clear()
        
    def save_state(self, path: str):
        """Save inference state."""
        torch.save({
            'cache': self.cache,
            'config': self.config
        }, path)
        
    def load_state(self, path: str):
        """Load inference state."""
        data = torch.load(path)
        self.cache = data['cache']
        self.config = data['config']
