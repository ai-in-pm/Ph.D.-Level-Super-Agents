from dotenv import load_dotenv
import os
from anthropic import Anthropic
from .base_agent import BaseAgent
from .models import TaskInput, ScratchpadItem, AgentRole
from typing import Any, Dict

class Analyzer(BaseAgent):
    def __init__(self):
        super().__init__("Analyzer", "claude-3-opus-20240229", AgentRole.ANALYZER)
        load_dotenv()
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
    def process_task(self, task: str) -> dict:
        task_input = TaskInput(content=task)
        analysis_data = self._process_task_implementation(task_input)
        
        return {
            "status": "success",
            "analysis": analysis_data,
            "chain_of_thought": self.get_chain_of_thought(),
            "scratchpad": self.scratchpad
        }
        
    def _process_task_implementation(self, task_input: TaskInput) -> Dict[str, Any]:
        self.add_to_chain(f"Received financial analysis task: {task_input.content}", confidence=0.9)
        
        # Initialize analysis components
        self.write_to_scratchpad(
            "analysis_components",
            {
                "risks": [],
                "metrics": {},
                "recommendations": []
            },
            metadata={"priority": task_input.priority}
        )
        
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": f"Analyze this financial task with detailed risk assessment and recommendations: {task_input.content}"
                }]
            )
            
            analysis = response.content[0].text
            self.add_to_chain(
                f"Completed analysis: {analysis[:100]}...",
                confidence=0.85
            )
            
            # Structure the analysis
            analysis_data = {
                "content": analysis,
                "type": "financial_analysis",
                "metadata": {
                    "model": "claude-3-opus",
                    "task_id": task_input.task_id,
                    "priority": task_input.priority
                }
            }
            
            self.write_to_scratchpad(
                "current_analysis",
                analysis_data,
                metadata={
                    "timestamp": task_input.metadata.get("timestamp"),
                    "confidence": 0.85
                }
            )
            
            return analysis_data
            
        except Exception as e:
            self.add_to_chain(f"Error in analysis: {str(e)}", confidence=0.1)
            raise e
