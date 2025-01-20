from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

class AgentRole(str, Enum):
    INNOVATOR = "innovator"
    ANALYZER = "analyzer"
    STRATEGIST = "strategist"
    DEVELOPER = "developer"
    SYNTHESIZER = "synthesizer"
    OPTIMIZER = "optimizer"
    RESEARCHER = "researcher"

class ThoughtItem(BaseModel):
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    agent: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    role: AgentRole
    references: List[str] = Field(default_factory=list)
    
class ScratchpadItem(BaseModel):
    content: Any
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    agent: str
    version: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)
    role: AgentRole
    dependencies: List[str] = Field(default_factory=list)
    
class TaskPriority(BaseModel):
    value: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

class TaskInput(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    content: str
    priority: TaskPriority
    deadline: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    required_roles: List[AgentRole] = Field(default_factory=list)
    
class AgentResponse(BaseModel):
    status: str
    agent_name: str
    model_type: str
    chain_of_thought: List[ThoughtItem]
    scratchpad: Dict[str, ScratchpadItem]
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    role: AgentRole
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
class CollaborativeMemory(BaseModel):
    memory_id: str
    content: Any
    source_agent: str
    target_agents: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    memory_type: str
    expiration: Optional[str] = None
    
class ReflectionLog(BaseModel):
    reflection_id: str = Field(default_factory=lambda: f"reflection_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    agent: str
    task_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    thoughts_summary: List[str]
    insights: List[str]
    improvements: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    role: AgentRole
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    
class AgentMetrics(BaseModel):
    agent_name: str
    role: AgentRole
    task_count: int = 0
    success_rate: float = 0.0
    average_confidence: float = 0.0
    average_execution_time: float = 0.0
    memory_usage: Dict[str, int] = Field(default_factory=dict)
    collaboration_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    specialization_scores: Dict[str, float] = Field(default_factory=dict)
    
class CollaborativeResult(BaseModel):
    task_id: str
    status: str
    participating_agents: List[str]
    final_result: Any
    individual_contributions: Dict[str, Any]
    execution_time: float
    confidence_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    role_distribution: Dict[AgentRole, float] = Field(default_factory=dict)
    improvement_suggestions: List[str] = Field(default_factory=list)

class AgentCapability(BaseModel):
    name: str
    description: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requirements: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentProfile(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    role: AgentRole
    model_type: str
    capabilities: List[AgentCapability] = Field(default_factory=list)
    specializations: List[str] = Field(default_factory=list)
    performance_history: Dict[str, float] = Field(default_factory=dict)
    collaboration_preferences: Dict[AgentRole, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
