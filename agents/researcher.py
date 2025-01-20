from dotenv import load_dotenv
import os
from openai import OpenAI
from .base_agent import BaseAgent
from .models import TaskInput, ScratchpadItem, AgentRole, TaskPriority
from typing import Any, Dict

class Researcher(BaseAgent):
    def __init__(self):
        super().__init__("Researcher", "gpt-4-turbo-preview", AgentRole.RESEARCHER)
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def process_task(self, task: str) -> dict:
        task_input = TaskInput(
            content=task,
            priority=TaskPriority(value=0.5, reason="Default priority", urgency=0.5, importance=0.5)
        )
        research_data = self._process_task_implementation(task_input)
        
        return {
            "status": "success",
            "research": research_data,
            "chain_of_thought": self.get_chain_of_thought(),
            "scratchpad": self.scratchpad
        }
        
    def _process_task_implementation(self, task_input: TaskInput) -> Dict[str, Any]:
        self.add_to_chain(f"Received research task: {task_input.content}")
        
        # Initialize research components
        self.write_to_scratchpad(
            "research_components",
            {
                "findings": [],
                "sources": [],
                "insights": []
            },
            metadata={"priority": task_input.priority}
        )
        
        try:
            messages = [
                {"role": "system", "content": "You are a research expert focused on gathering and analyzing information."},
                {"role": "user", "content": f"Research and analyze this topic: {task_input.content}"}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages
            )
            
            research = response.choices[0].message.content
            self.add_to_chain(
                f"Completed research: {research[:100]}...",
                confidence=0.85
            )
            
            # Parse and structure the research
            research_components = self._parse_research(research)
            self.write_to_scratchpad(
                "research_components",
                research_components,
                metadata={"stage": "final"}
            )
            
            return research_components
            
        except Exception as e:
            self.add_to_chain(f"Error in research: {str(e)}", confidence=0.3)
            return {"error": str(e)}
    
    def _parse_research(self, research: str) -> Dict[str, Any]:
        """Parse the raw research text into structured components."""
        # This is a simple implementation - could be enhanced with more sophisticated parsing
        lines = research.split("\n")
        components = {
            "findings": [],
            "sources": [],
            "insights": []
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "finding" in line.lower() or "discovery" in line.lower():
                current_section = "findings"
            elif "source" in line.lower() or "reference" in line.lower():
                current_section = "sources"
            elif "insight" in line.lower() or "conclusion" in line.lower():
                current_section = "insights"
            elif current_section:
                components[current_section].append(line)
                
        return components
