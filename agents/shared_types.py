"""
Shared types and interfaces used across different modules.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ConsciousnessState:
    """State of consciousness metrics."""
    cognitive_depth: float = 0.0
    emotional_intelligence: float = 0.0
    creativity: float = 0.0
    analytical: float = 0.0
    strategic: float = 0.0
    technical: float = 0.0
    social: float = 0.0
    learning: float = 0.0
    meta_cognition: Dict[str, float] = None

    def __post_init__(self):
        if self.meta_cognition is None:
            self.meta_cognition = {}
