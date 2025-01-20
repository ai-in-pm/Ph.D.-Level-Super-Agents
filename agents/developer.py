from dotenv import load_dotenv
import os
from groq import Groq
from .base_agent import BaseAgent
from .models import TaskInput, ScratchpadItem, AgentRole
from typing import Any, Dict

class Developer(BaseAgent):
    def __init__(self):
        super().__init__("Developer", "mixtral-8x7b-32768", AgentRole.DEVELOPER)
        load_dotenv()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
    def process_task(self, task: str) -> dict:
        task_input = TaskInput(content=task)
        implementation_data = self._process_task_implementation(task_input)
        
        return {
            "status": "success",
            "implementation": implementation_data["content"],
            "chain_of_thought": self.get_chain_of_thought(),
            "scratchpad": self.scratchpad
        }
        
    def _process_task_implementation(self, task_input: TaskInput) -> Dict[str, Any]:
        self.add_to_chain(f"Received development task: {task_input.content}", confidence=0.9)
        
        # Initialize development components
        self.write_to_scratchpad(
            "development_components",
            {
                "code_snippets": [],
                "optimizations": [],
                "test_cases": []
            },
            metadata={
                "priority": task_input.priority,
                "stage": "initialization"
            }
        )
        
        try:
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert software developer focused on optimization."},
                    {"role": "user", "content": f"Implement and optimize this solution: {task_input.content}"}
                ],
                temperature=0.3
            )
            
            implementation = response.choices[0].message.content
            self.add_to_chain(
                f"Created implementation: {implementation[:100]}...",
                confidence=0.85
            )
            
            # Structure the implementation
            implementation_data = {
                "content": implementation,
                "type": "code_implementation",
                "metadata": {
                    "model": "mixtral-8x7b",
                    "task_id": task_input.task_id,
                    "priority": task_input.priority
                }
            }
            
            self.write_to_scratchpad(
                "current_implementation",
                implementation_data,
                metadata={
                    "stage": "completed",
                    "confidence": 0.85,
                    "dependencies": task_input.dependencies,
                    "optimization_level": "initial"
                }
            )
            
            return implementation_data
            
        except Exception as e:
            self.add_to_chain(f"Implementation error: {str(e)}", confidence=0.1)
            raise e
