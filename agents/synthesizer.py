from dotenv import load_dotenv
import os
import google.generativeai as genai
from .base_agent import BaseAgent
from .models import TaskInput, ScratchpadItem, AgentRole
from typing import Any, Dict

class Synthesizer(BaseAgent):
    def __init__(self):
        super().__init__("Synthesizer", "gemini-pro", AgentRole.SYNTHESIZER)
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        
    def process_task(self, task: str) -> dict:
        task_input = TaskInput(content=task)
        synthesis_data = self._process_task_implementation(task_input)
        
        return {
            "status": "success",
            "synthesis": synthesis_data["content"],
            "chain_of_thought": self.get_chain_of_thought(),
            "scratchpad": self.scratchpad
        }
        
    def _process_task_implementation(self, task_input: TaskInput) -> Dict[str, Any]:
        self.add_to_chain(f"Received synthesis task: {task_input.content}", confidence=0.9)
        
        # Initialize synthesis components
        self.write_to_scratchpad(
            "synthesis_components",
            {
                "integrated_solutions": [],
                "conflicts": [],
                "resolutions": []
            },
            metadata={
                "priority": task_input.priority,
                "stage": "initialization"
            }
        )
        
        try:
            chat = self.model.start_chat(history=[])
            response = chat.send_message(
                f"Synthesize and integrate various approaches for this task, "
                f"resolving any conflicts and ensuring coherence: {task_input.content}"
            )
            
            synthesis = response.text
            self.add_to_chain(
                f"Completed synthesis: {synthesis[:100]}...",
                confidence=0.85
            )
            
            # Structure the synthesis
            synthesis_data = {
                "content": synthesis,
                "type": "integration_synthesis",
                "metadata": {
                    "model": "gemini-pro",
                    "task_id": task_input.task_id,
                    "priority": task_input.priority
                }
            }
            
            self.write_to_scratchpad(
                "current_synthesis",
                synthesis_data,
                metadata={
                    "stage": "completed",
                    "confidence": 0.85,
                    "dependencies": task_input.dependencies,
                    "integration_level": "complete"
                }
            )
            
            return synthesis_data
            
        except Exception as e:
            self.add_to_chain(f"Synthesis error: {str(e)}", confidence=0.1)
            raise e
