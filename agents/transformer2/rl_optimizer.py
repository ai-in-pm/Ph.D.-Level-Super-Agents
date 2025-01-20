"""
Reinforcement Learning optimizer for Transformer2.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from .svf import SVFOptimizer
from .experts import ExpertPool

@dataclass
class RLConfig:
    """Configuration for RL optimization."""
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    buffer_size: int = 10000
    update_interval: int = 10

class ReplayBuffer:
    """Experience replay buffer for RL."""
    
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer: List[Dict[str, Any]] = []
        self.position = 0
        
    def push(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool
    ):
        """Add experience to buffer."""
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
            
        self.buffer[self.position] = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        self.position = (self.position + 1) % self.buffer_size
        
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch of experiences."""
        return np.random.choice(self.buffer, batch_size)
        
    def __len__(self) -> int:
        return len(self.buffer)

class RLOptimizer:
    """Reinforcement Learning optimizer for expert vectors."""
    
    def __init__(
        self,
        config: RLConfig,
        svf_optimizer: SVFOptimizer,
        expert_pool: ExpertPool
    ):
        self.config = config
        self.svf_optimizer = svf_optimizer
        self.expert_pool = expert_pool
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.steps = 0
        
        # Initialize networks
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        
    def _build_q_network(self) -> nn.Module:
        """Build Q-network for RL."""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def optimize_expert(
        self,
        expert_id: str,
        state: Dict[str, torch.Tensor],
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool
    ):
        """Optimize expert vector using RL."""
        expert = self.expert_pool.get_expert(expert_id)
        if expert is None:
            return
            
        # Store experience
        self.replay_buffer.push(
            state,
            expert.vector,
            reward,
            next_state,
            done
        )
        
        # Update networks
        self.steps += 1
        if len(self.replay_buffer) >= self.config.batch_size and \
           self.steps % self.config.update_interval == 0:
            self._update_networks()
            
    def _update_networks(self):
        """Update Q-network and target network."""
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Prepare batch data
        state_batch = torch.stack([b['state']['embedding'] for b in batch])
        action_batch = torch.stack([b['action'] for b in batch])
        reward_batch = torch.tensor([b['reward'] for b in batch])
        next_state_batch = torch.stack([b['next_state']['embedding'] for b in batch])
        done_batch = torch.tensor([b['done'] for b in batch])
        
        # Compute Q values
        current_q = self.q_network(state_batch)
        next_q = self.target_network(next_state_batch)
        
        # Compute target Q values
        target_q = reward_batch + (1 - done_batch) * self.config.gamma * next_q
        
        # Update Q-network
        loss = nn.MSELoss()(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1 - self.config.tau) +
                param.data * self.config.tau
            )
            
    def get_expert_action(
        self,
        state: Dict[str, torch.Tensor],
        expert_id: str,
        epsilon: float = 0.1
    ) -> torch.Tensor:
        """Get action from expert with epsilon-greedy policy."""
        if np.random.random() < epsilon:
            # Exploration: random perturbation of expert vector
            expert = self.expert_pool.get_expert(expert_id)
            if expert is None:
                return None
            return expert.vector + torch.randn_like(expert.vector) * 0.1
            
        # Exploitation: use Q-network
        with torch.no_grad():
            expert = self.expert_pool.get_expert(expert_id)
            if expert is None:
                return None
            q_value = self.q_network(state['embedding'])
            return expert.vector * torch.sigmoid(q_value)
            
    def save_state(self, path: str):
        """Save optimizer state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps
        }, path)
        
    def load_state(self, path: str):
        """Load optimizer state."""
        data = torch.load(path)
        self.q_network.load_state_dict(data['q_network'])
        self.target_network.load_state_dict(data['target_network'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.steps = data['steps']
