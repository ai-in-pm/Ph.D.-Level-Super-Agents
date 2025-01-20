import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any

from agents import (
    Innovator, Analyzer,
    Developer, Synthesizer, Optimizer, Researcher
)
from agents.models import (
    TaskInput, TaskPriority, AgentRole,
    CollaborativeResult, AgentMetrics
)
from agent_orchestrator import AgentOrchestrator
from agents.consciousness.advanced_metrics import AdvancedConsciousnessMetrics, CognitiveDepthMetrics, EmotionalIntelligenceMetrics, CreativityMetrics, AnalyticalMetrics, StrategicMetrics, TechnicalMetrics, SocialMetrics, LearningMetrics
from agents.consciousness.advanced_behaviors import AdvancedBehaviorManager
from agents.transformer2.advanced_adaptation import AdvancedAdaptationManager, AdaptationConfig, AdaptationStrategy

# Load environment variables
load_dotenv()

def create_agents() -> List[Any]:
    """Initialize all agents with their respective roles and API configurations."""
    # Initialize consciousness components
    consciousness_metrics = AdvancedConsciousnessMetrics(
        cognitive=CognitiveDepthMetrics(
            abstraction_level=0.5,
            conceptual_complexity=0.5,
            reasoning_depth=0.5,
            knowledge_integration=0.5,
            mental_model_coherence=0.5
        ),
        emotional=EmotionalIntelligenceMetrics(
            self_awareness=0.5,
            empathy=0.5,
            emotional_regulation=0.5,
            social_perception=0.5,
            emotional_adaptation=0.5
        ),
        creative=CreativityMetrics(
            divergent_thinking=0.5,
            originality=0.5,
            flexibility=0.5,
            elaboration=0.5,
            synthesis_ability=0.5
        ),
        analytical=AnalyticalMetrics(
            logical_reasoning=0.5,
            pattern_recognition=0.5,
            critical_thinking=0.5,
            problem_decomposition=0.5,
            solution_evaluation=0.5
        ),
        strategic=StrategicMetrics(
            long_term_planning=0.5,
            risk_assessment=0.5,
            resource_optimization=0.5,
            adaptability=0.5,
            decision_quality=0.5
        ),
        technical=TechnicalMetrics(
            code_quality=0.5,
            system_understanding=0.5,
            optimization_ability=0.5,
            debugging_skill=0.5,
            architecture_design=0.5
        ),
        social=SocialMetrics(
            collaboration=0.5,
            communication=0.5,
            influence=0.5,
            conflict_resolution=0.5,
            team_awareness=0.5
        ),
        learning=LearningMetrics(
            acquisition_rate=0.5,
            retention=0.5,
            transfer_ability=0.5,
            skill_integration=0.5,
            learning_efficiency=0.5
        ),
        timestamp=datetime.now().timestamp()
    )
    behavior_manager = AdvancedBehaviorManager()
    adaptation_config = AdaptationConfig(
        strategy=AdaptationStrategy.META_LEARNING,
        learning_rate=0.01,
        momentum=0.9,
        adaptation_rate=0.1,
        exploration_rate=0.2,
        stability_factor=0.8
    )
    adaptation_manager = AdvancedAdaptationManager(
        initial_config=adaptation_config,
        consciousness_metrics=consciousness_metrics
    )

    return [
        Innovator(),
        Analyzer(),
        Developer(),
        Synthesizer(),
        Optimizer(),
        Researcher()
    ]

def print_result(result: CollaborativeResult):
    """Print the collaborative result in a formatted way."""
    print("\n" + "="*50)
    print("Task Results")
    print("="*50)
    print(f"Task ID: {result.task_id}")
    print(f"Status: {result.status}")
    print(f"Confidence Score: {result.confidence_score:.2f}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print("\nRole Distribution:")
    for role, contribution in result.role_distribution.items():
        print(f"  {role.value}: {contribution:.2%}")
    print("\nConsciousness Metrics:")
    print(f"  Cognitive Depth: {result.consciousness_metrics.cognitive.abstraction_level:.2f}")
    print(f"  Emotional Intelligence: {result.consciousness_metrics.emotional.self_awareness:.2f}")
    print(f"  Learning Efficiency: {result.consciousness_metrics.learning.learning_efficiency:.2f}")
    print("\nBehavior Patterns:")
    for behavior in result.active_behaviors:
        print(f"  - {behavior.value}")
    print("\nFinal Result:")
    print(f"{result.final_result}")
    if result.improvement_suggestions:
        print("\nImprovement Suggestions:")
        for suggestion in result.improvement_suggestions:
            print(f"- {suggestion}")
    print("="*50)

def print_metrics(metrics: Dict[str, AgentMetrics]):
    """Print agent metrics in a formatted way."""
    print("\n" + "="*50)
    print("Agent Metrics")
    print("="*50)
    for agent_name, metric in metrics.items():
        print(f"\nAgent: {agent_name}")
        print(f"Role: {metric.role.value}")
        print(f"Success Rate: {metric.success_rate:.2%}")
        print(f"Average Confidence: {metric.average_confidence:.2f}")
        print(f"Average Execution Time: {metric.average_execution_time:.2f}s")
        print("Specialization Scores:")
        for spec, score in metric.specialization_scores.items():
            print(f"  {spec}: {score:.2f}")
        print("Consciousness Metrics:")
        print(f"  Cognitive Depth: {metric.consciousness_metrics.cognitive.abstraction_level:.2f}")
        print(f"  Emotional Intelligence: {metric.consciousness_metrics.emotional.self_awareness:.2f}")
        print(f"  Learning Efficiency: {metric.consciousness_metrics.learning.learning_efficiency:.2f}")
        print("Active Behaviors:")
        for behavior in metric.active_behaviors:
            print(f"  - {behavior.value}")
    print("="*50)

async def main():
    """Main entry point for the superagent system."""
    # Create agents and orchestrator
    agents = create_agents()
    orchestrator = AgentOrchestrator(agents)
    
    # Example tasks with different priorities
    tasks = [
        (
            "Design a scalable microservices architecture for an e-commerce platform",
            TaskPriority(
                value=0.9,
                reason="Critical business infrastructure",
                urgency=0.8,
                importance=0.9
            )
        ),
        (
            "Optimize the database queries for better performance",
            TaskPriority(
                value=0.7,
                reason="Performance improvement",
                urgency=0.6,
                importance=0.8
            )
        ),
        (
            "Implement a secure authentication system",
            TaskPriority(
                value=0.8,
                reason="Security requirement",
                urgency=0.9,
                importance=0.9
            )
        )
    ]
    
    # Process tasks
    for task_description, priority in tasks:
        task_input = TaskInput(
            content=task_description,
            priority=priority,
            deadline=datetime.now().isoformat(),
            required_roles=[
                AgentRole.INNOVATOR,
                AgentRole.ANALYZER,
                AgentRole.DEVELOPER
            ]
        )
        
        # Process task and print results
        result = await orchestrator.process_task(task_input)
        print_result(result)
        
        # Print consciousness insights
        print("\nConsciousness Insights:")
        for agent in agents:
            consciousness_state = agent.get_consciousness_state()
            print(f"\n{agent.name.capitalize()} Agent Consciousness:")
            print(f"  Active Behaviors: {[b.value for b in consciousness_state.active_behaviors]}")
            print(f"  Learning Progress: {consciousness_state.learning_progress:.2%}")
            print(f"  Adaptation Strategy: {consciousness_state.adaptation_strategy.value}")
            
        # Allow some time between tasks for agents to update their state
        await asyncio.sleep(1)
    
    # Print final metrics
    metrics = orchestrator.get_agent_metrics()
    print_metrics(metrics)
    
    # Print advanced insights
    print("\nAdvanced Insights:")
    for agent in agents:
        profile = agent.get_profile()
        print(f"\n{profile.name.capitalize()} Agent:")
        print(f"Consciousness Evolution:")
        print(f"  Initial State: {profile.consciousness_evolution.initial_state}")
        print(f"  Current State: {profile.consciousness_evolution.current_state}")
        print(f"  Growth Rate: {profile.consciousness_evolution.growth_rate:.2%}")
        print(f"Top Collaborators:")
        sorted_collabs = sorted(
            profile.collaboration_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for role, score in sorted_collabs[:3]:
            print(f"- {role.value}: {score:.2f}")

if __name__ == "__main__":
    # Verify environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "COHERE_API_KEY",
        "EMERGENCE_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables in your .env file.")
        exit(1)
    
    # Run the main async function
    asyncio.run(main())
