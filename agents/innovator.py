from dotenv import load_dotenv
import os
from openai import OpenAI
from .base_agent import BaseAgent
from .models import TaskInput, ScratchpadItem, AgentRole
from typing import Any, Dict

class Innovator(BaseAgent):
    def __init__(self):
        super().__init__("Innovator", "gpt-4-turbo-preview", AgentRole.INNOVATOR)
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def process_task(self, task: str) -> dict:
        task_input = TaskInput(content=task)
        solution_data = self._process_task_implementation(task_input)
        
        return {
            "status": "success",
            "solution": solution_data,
            "chain_of_thought": self.get_chain_of_thought(),
            "scratchpad": self.scratchpad
        }
        
    def _process_task_implementation(self, task_input: TaskInput) -> Dict[str, Any]:
        self.add_to_chain(f"Received task: {task_input.content}")
        
        # Initialize solution components in scratchpad
        self.write_to_scratchpad(
            "current_problem", 
            task_input.content,
            metadata={"priority": task_input.priority}
        )
        self.write_to_scratchpad(
            "solution_components",
            [],
            metadata={"stage": "initial"}
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an innovative AI focused on software design and implementation."},
                    {"role": "user", "content": task_input.content}
                ],
                temperature=0.7
            )
            
            solution = response.choices[0].message.content
            self.add_to_chain(
                f"Generated solution approach: {solution[:100]}...",
                confidence=0.85
            )
            
            # Structure the solution
            solution_data = {
                "content": solution,
                "approach": "innovative",
                "dependencies": task_input.dependencies
            }
            
            self.write_to_scratchpad(
                "current_solution",
                solution_data,
                metadata={
                    "model": "gpt-4-turbo-preview",
                    "task_id": task_input.task_id
                }
            )
            
            return solution_data
            
        except Exception as e:
            self.add_to_chain(f"Error occurred: {str(e)}", confidence=0.1)
            return {
                "status": "error",
                "error": str(e),
                "chain_of_thought": self.get_chain_of_thought()
            }
