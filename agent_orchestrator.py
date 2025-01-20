from typing import List, Dict, Any, Optional
from agents.models import (
    TaskInput, CollaborativeResult, AgentResponse,
    CollaborativeMemory, ReflectionLog, AgentMetrics,
    TaskPriority
)
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AgentOrchestrator:
    def __init__(self, agents: List[Any]):
        self.agents = {agent.name: agent for agent in agents}
        self.shared_memory: Dict[str, CollaborativeMemory] = {}
        self.metrics: Dict[str, AgentMetrics] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.reflection_logs: List[ReflectionLog] = []
        
    async def process_task(self, task_content: str, priority: float = 0.5) -> CollaborativeResult:
        """Process a task using all available agents asynchronously."""
        task_input = TaskInput(
            content=task_content,
            priority=TaskPriority(value=priority, reason="Default priority", urgency=priority, importance=priority),
            metadata={"orchestrator_timestamp": datetime.now().isoformat()}
        )
        
        start_time = datetime.now()
        
        # Initialize metrics for new agents
        for agent_name in self.agents:
            if agent_name not in self.metrics:
                self.metrics[agent_name] = AgentMetrics(agent_name=agent_name)
        
        try:
            # Process task with all agents asynchronously
            with ThreadPoolExecutor() as executor:
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(executor, agent.process_task, task_input)
                    for agent in self.agents.values()
                ]
                responses = await asyncio.gather(*tasks)
            
            # Process individual results
            individual_contributions = {}
            participating_agents = []
            total_confidence = 0.0
            
            for response in responses:
                agent_name = response.agent_name
                participating_agents.append(agent_name)
                
                # Update metrics
                metrics = self.metrics[agent_name]
                metrics.task_count += 1
                metrics.success_rate = (
                    metrics.success_rate * (metrics.task_count - 1) + 
                    (1.0 if response.status == "success" else 0.0)
                ) / metrics.task_count
                metrics.average_execution_time = (
                    metrics.average_execution_time * (metrics.task_count - 1) + 
                    response.execution_time
                ) / metrics.task_count
                
                # Calculate confidence from chain of thought
                agent_confidence = sum(
                    thought.confidence for thought in response.chain_of_thought
                ) / len(response.chain_of_thought) if response.chain_of_thought else 0.0
                
                metrics.average_confidence = (
                    metrics.average_confidence * (metrics.task_count - 1) + 
                    agent_confidence
                ) / metrics.task_count
                
                # Store individual contributions
                individual_contributions[agent_name] = {
                    "status": response.status,
                    "result": response.result,
                    "confidence": agent_confidence,
                    "execution_time": response.execution_time
                }
                
                total_confidence += agent_confidence
                
                # Share memory between agents
                for key, item in response.scratchpad.items():
                    memory_id = f"{task_input.task_id}_{agent_name}_{key}"
                    self.shared_memory[memory_id] = CollaborativeMemory(
                        memory_id=memory_id,
                        content=item.content,
                        source_agent=agent_name,
                        target_agents=[a for a in self.agents.keys() if a != agent_name],
                        metadata=item.metadata
                    )
            
            # Calculate final confidence score
            avg_confidence = total_confidence / len(participating_agents) if participating_agents else 0.0
            
            # Create reflection log
            reflection = ReflectionLog(
                agent="orchestrator",
                task_id=task_input.task_id,
                thoughts_summary=[
                    f"{agent}: {resp.chain_of_thought[-1].content if resp.chain_of_thought else 'No thoughts'}"
                    for agent, resp in zip(participating_agents, responses)
                ],
                insights=[
                    f"Agent {agent} contributed with confidence {individual_contributions[agent]['confidence']:.2f}"
                    for agent in participating_agents
                ],
                improvements=[
                    f"Optimize {agent} execution time: {individual_contributions[agent]['execution_time']:.2f}s"
                    for agent in participating_agents
                ]
            )
            self.reflection_logs.append(reflection)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create collaborative result
            result = CollaborativeResult(
                task_id=task_input.task_id,
                status="success",
                participating_agents=participating_agents,
                final_result=self._synthesize_results(individual_contributions),
                individual_contributions=individual_contributions,
                execution_time=execution_time,
                confidence_score=avg_confidence,
                metadata={
                    "reflection_id": reflection.reflection_id,
                    "shared_memory_keys": list(self.shared_memory.keys())
                }
            )
            
            self.task_history.append(result.dict())
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_result = CollaborativeResult(
                task_id=task_input.task_id,
                status="error",
                participating_agents=[],
                final_result=None,
                individual_contributions={},
                execution_time=execution_time,
                confidence_score=0.0,
                metadata={"error": str(e)}
            )
            self.task_history.append(error_result.dict())
            return error_result
    
    def _synthesize_results(self, individual_contributions: Dict[str, Any]) -> Any:
        """Synthesize individual results into a final result."""
        # Prioritize results based on confidence and agent role
        synthesized = {
            agent: contrib for agent, contrib in individual_contributions.items()
            if contrib["status"] == "success" and contrib["confidence"] > 0.7
        }
        
        if not synthesized:
            return None
            
        # Combine results based on agent roles and confidence
        final_result = {
            "summary": "Combined insights from multiple agents",
            "contributions": synthesized,
            "confidence_weighted_results": {
                agent: {
                    "result": contrib["result"],
                    "weight": contrib["confidence"]
                }
                for agent, contrib in synthesized.items()
            }
        }
        
        return final_result
    
    def get_agent_metrics(self) -> Dict[str, AgentMetrics]:
        """Get metrics for all agents."""
        return self.metrics
    
    def get_reflection_logs(self) -> List[ReflectionLog]:
        """Get all reflection logs."""
        return self.reflection_logs
    
    def get_shared_memory(self) -> Dict[str, CollaborativeMemory]:
        """Get all shared memory items."""
        return self.shared_memory

# Example usage:
if __name__ == "__main__":
    from agents.innovator import Innovator
    from agents.analyzer import Analyzer
    from agents.strategist import Strategist
    from agents.developer import Developer
    from agents.synthesizer import Synthesizer
    from agents.optimizer import Optimizer
    from agents.researcher import Researcher
    
    agents = [
        Innovator(),
        Analyzer(),
        Strategist(),
        Developer(),
        Synthesizer(),
        Optimizer(),
        Researcher()
    ]
    
    orchestrator = AgentOrchestrator(agents)
    result = orchestrator.process_task(
        "Design a secure payment processing system with focus on scalability.",
        priority=2
    )
    print(result.json(indent=2))
