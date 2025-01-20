from dotenv import load_dotenv
import os
import cohere
from .base_agent import BaseAgent
from .models import TaskInput, ScratchpadItem, AgentRole
from typing import Any, Dict

class Optimizer(BaseAgent):
    def __init__(self):
        super().__init__("Optimizer", "command", AgentRole.OPTIMIZER)
        load_dotenv()
        self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        
    def process_task(self, task: str) -> dict:
        task_input = TaskInput(content=task)
        optimization_data = self._process_task_implementation(task_input)
        
        return {
            "status": "success",
            "optimization": optimization_data["content"],
            "chain_of_thought": self.get_chain_of_thought(),
            "scratchpad": self.scratchpad
        }
        
    def _process_task_implementation(self, task_input: TaskInput) -> Dict[str, Any]:
        self.add_to_chain(f"Received optimization task: {task_input.content}", confidence=0.9)
        
        # Initialize optimization components
        self.write_to_scratchpad(
            "optimization_components",
            {
                "improvements": [],
                "benchmarks": {},
                "recommendations": []
            },
            metadata={
                "priority": task_input.priority,
                "stage": "initialization"
            }
        )
        
        try:
            response = self.client.chat(
                message=f"Optimize and refine this solution, focusing on efficiency and clarity: {task_input.content}",
                model="command",
                temperature=0.3,
                chat_history=[]
            )
            
            optimization = response.text
            self.add_to_chain(
                f"Completed optimization: {optimization[:100]}...",
                confidence=0.85
            )
            
            # Structure the optimization
            optimization_data = {
                "content": optimization,
                "type": "solution_optimization",
                "metadata": {
                    "model": "command",
                    "task_id": task_input.task_id,
                    "priority": task_input.priority
                }
            }
            
            self.write_to_scratchpad(
                "current_optimization",
                optimization_data,
                metadata={
                    "stage": "completed",
                    "confidence": 0.85,
                    "dependencies": task_input.dependencies,
                    "optimization_metrics": {
                        "efficiency_score": 0.85,
                        "clarity_score": 0.9
                    }
                }
            )
            
            return optimization_data
            
        except Exception as e:
            self.add_to_chain(f"Optimization error: {str(e)}", confidence=0.1)
            raise e
