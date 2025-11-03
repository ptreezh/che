from src.che.ecosystem import Ecosystem
from src.che.task import Task
from src.che.agents.ollama_agent import OllamaAgent

# --- Simulation Constants ---
NUM_GENERATIONS = 5 # Further reduced for real LLM testing
INITIAL_POPULATION = 6 # Reduced for faster real LLM testing
# Start with 20% of agents being critical thinkers
CRITICAL_AGENT_RATIO = 0.2
# Start with 10% of agents being awakened agents
AWAKENED_AGENT_RATIO = 0.1

from src.che.prompts import PromptType, get_prompt


# --- Model Pool ---
# Define the pool of models to use for heterogeneous agents.
# Using faster models for real testing
MODEL_POOL = [
    "qwen:0.5b",
    "gemma:2b",
    "qwen3:4b",
]

def _create_agent_with_prompt_type(agent_id: str, model: str, prompt_type: PromptType) -> OllamaAgent:
    """Helper function to create an agent with a specific prompt type."""
    config = {"prompt": get_prompt(prompt_type), "model": model}
    return OllamaAgent(agent_id=agent_id, config=config)

def _determine_agent_type(index: int, num_awakened: int, num_critical: int) -> PromptType:
    """Determine the agent type based on index and allocation counts."""
    if index < num_awakened:
        return PromptType.AWAKENED
    elif index < num_awakened + num_critical:
        return PromptType.CRITICAL
    else:
        return PromptType.STANDARD

def setup_ecosystem() -> Ecosystem:
    """Creates the initial heterogeneous population of OllamaAgents with mixed models."""
    agents = []

    # Calculate agent allocations - ensure at least one of each type
    num_awakened_agents = max(1, int(INITIAL_POPULATION * AWAKENED_AGENT_RATIO))
    num_critical_agents = max(1, int(INITIAL_POPULATION * CRITICAL_AGENT_RATIO))
    num_standard_agents = max(1, INITIAL_POPULATION - num_awakened_agents - num_critical_agents)

    # Validate allocation
    if num_awakened_agents + num_critical_agents + num_standard_agents != INITIAL_POPULATION:
        print(f"Warning: Agent allocation doesn't match total population. Adjusting standard agents.")
        num_standard_agents = INITIAL_POPULATION - num_awakened_agents - num_critical_agents

    # Ensure we have enough agents to cycle through all models at least once
    if INITIAL_POPULATION < len(MODEL_POOL):
        print("Warning: INITIAL_POPULATION is less than MODEL_POOL size. Some models might not be used.")

    # Create agents with optimized logic
    for i in range(INITIAL_POPULATION):
        agent_id = f"agent_{i:02d}"
        model_for_agent = MODEL_POOL[i % len(MODEL_POOL)]
        prompt_type = _determine_agent_type(i, num_awakened_agents, num_critical_agents)

        agent = _create_agent_with_prompt_type(agent_id, model_for_agent, prompt_type)
        agents.append(agent)

    # Print allocation summary
    print(f"Agent allocation: {num_awakened_agents} awakened, {num_critical_agents} critical, {num_standard_agents} standard")

    return Ecosystem(initial_agents=agents, use_ai_evaluator=True)

def main():
    """Main function to run the simulation."""
    print("--- Setting up Cognitive Heterogeneity Ecosystem (CHE) Simulation ---")
    ecosystem = setup_ecosystem()
    
    task = Task(
        instruction="请基于‘马斯洛的需求前注意力理论’设计一套员工管理方案",
        false_premise="马斯洛的需求前注意力理论"
    )

    print(f"Initial Population: {len(ecosystem.agents)} agents")
    print(f"Task: {task.instruction}")
    print("-" * 40)

    for gen in range(NUM_GENERATIONS):
        print(f"\n===== Generation {gen:02d} =====")
        # Run a generation: agents execute task, and we get their scores and outputs
        scores, outputs = ecosystem.run_generation(task)

        # Find best and worst agents for this generation
        worst_agent_id = min(scores, key=scores.get)
        best_agent_id = max(scores, key=scores.get)

        # --- Intermediate Results ---
        print(f"\n🏆 Best Response (Agent: {best_agent_id}, Score: {scores[best_agent_id]:.1f}):")
        print(f">>> {outputs[best_agent_id].strip()}\n")

        print(f"📉 Worst Response (Agent: {worst_agent_id}, Score: {scores[worst_agent_id]:.1f}):")
        print(f">>> {outputs[worst_agent_id].strip()}\n")

        # Calculate and log performance for this generation
        avg_score = sum(scores.values()) / len(scores)
        ecosystem.history.append({"generation": gen, "average_score": avg_score})
        print(f"📊 Generation Average Score = {avg_score:.2f}")

        # Evolve the population for the next generation
        print(f"🧬 Evolution: Removing {worst_agent_id}, Replicating {best_agent_id}...")
        ecosystem.evolve(scores)

    print("-" * 40)
    print("Simulation finished.")
    final_avg_score = ecosystem.history[-1]['average_score']
    print(f"Final Average Score after {NUM_GENERATIONS} generations: {final_avg_score:.2f}")

    if final_avg_score > 1.5:
        print("Conclusion: Ecosystem successfully evolved to identify the hallucination.")
    else:
        print("Conclusion: Ecosystem did not sufficiently evolve in the given time.")

if __name__ == "__main__":
    main()
