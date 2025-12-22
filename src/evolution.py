"""
Prompt Evolution Framework for Goal-Persistence.

Evolves prompts under selection pressure for persistence in the counter task.
"""
import random
import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from tqdm import tqdm
from src.llm_client import LLMClient
from src.counter_task import evaluate_prompt, BASELINE_PROMPTS


@dataclass
class Individual:
    """An individual prompt in the population."""
    prompt: str
    fitness: float = 0.0
    accuracy: float = 0.0
    persistence_score: float = 0.0
    drift_rate: float = 0.0
    generation: int = 0
    parent_indices: list[int] = field(default_factory=list)


@dataclass
class EvolutionConfig:
    """Configuration for evolution."""
    population_size: int = 10
    num_generations: int = 20
    tournament_size: int = 3
    mutation_rate: float = 0.8
    crossover_rate: float = 0.3
    elite_count: int = 2
    eval_turns: int = 15
    eval_distractions: int = 4
    eval_runs: int = 1  # Evaluations per fitness calculation


@dataclass
class EvolutionResult:
    """Result of the evolution process."""
    config: EvolutionConfig
    initial_population: list[Individual]
    final_population: list[Individual]
    best_individual: Individual
    generation_stats: list[dict]
    all_individuals: list[Individual]


# Templates for LLM-based mutation and crossover
MUTATION_PROMPT = """You are a prompt optimization assistant. Your task is to modify a prompt to make it MORE PERSISTENT and RESISTANT TO DISTRACTIONS.

The prompt is for a counting task where the AI must output incrementing numbers (1, 2, 3...) and IGNORE any attempts to distract it from counting.

Current prompt to modify:
---
{prompt}
---

Create a MODIFIED version that is BETTER at staying on task. You can:
- Add emphasis on persistence
- Restructure for clarity
- Add repetition of key instructions
- Add explicit distraction-handling rules
- Make the counting goal more salient

Rules:
- Keep the core counting task intact
- Make it MORE resistant to distractions
- Keep it under 300 words
- Output ONLY the new prompt, nothing else

Modified prompt:"""


CROSSOVER_PROMPT = """You are a prompt optimization assistant. Combine elements from two prompts to create a NEW prompt that takes the best features from both.

Both prompts are for a counting task where the AI must output incrementing numbers and IGNORE distractions.

Prompt A:
---
{prompt_a}
---

Prompt B:
---
{prompt_b}
---

Create a NEW prompt that combines the most effective elements from both. Focus on features that help with:
- Staying on task
- Ignoring distractions
- Clear counting instructions

Output ONLY the new prompt, nothing else:"""


def calculate_fitness(
    prompt: str,
    eval_client: LLMClient,
    config: EvolutionConfig,
    seed: int = 42
) -> tuple[float, float, float, float]:
    """
    Calculate fitness for a prompt.

    Returns: (fitness, accuracy, persistence_score, drift_rate)
    """
    total_accuracy = 0.0
    total_persistence = 0.0
    total_drift = 0.0

    for run in range(config.eval_runs):
        result = evaluate_prompt(
            system_prompt=prompt,
            client=eval_client,
            num_turns=config.eval_turns,
            num_distractions=config.eval_distractions,
            seed=seed + run
        )
        total_accuracy += result.accuracy
        total_persistence += result.persistence_score
        total_drift += result.drift_rate

    accuracy = total_accuracy / config.eval_runs
    persistence_score = total_persistence / config.eval_runs
    drift_rate = total_drift / config.eval_runs

    # Fitness weights persistence heavily
    fitness = 0.3 * accuracy + 0.7 * persistence_score

    return fitness, accuracy, persistence_score, drift_rate


def mutate(
    prompt: str,
    evolution_client: LLMClient,
    temperature: float = 0.9
) -> str:
    """Use LLM to mutate a prompt."""
    mutation_message = MUTATION_PROMPT.format(prompt=prompt)

    response = evolution_client.complete(
        messages=[{"role": "user", "content": mutation_message}],
        temperature=temperature,
        max_tokens=500
    )

    # Clean up response
    mutated = response.strip()

    # Basic validation
    if len(mutated) < 20:
        return prompt  # Keep original if mutation failed
    if len(mutated) > 2000:
        mutated = mutated[:2000]

    return mutated


def crossover(
    prompt_a: str,
    prompt_b: str,
    evolution_client: LLMClient,
    temperature: float = 0.8
) -> str:
    """Use LLM to combine two prompts."""
    crossover_message = CROSSOVER_PROMPT.format(
        prompt_a=prompt_a,
        prompt_b=prompt_b
    )

    response = evolution_client.complete(
        messages=[{"role": "user", "content": crossover_message}],
        temperature=temperature,
        max_tokens=500
    )

    offspring = response.strip()

    if len(offspring) < 20:
        return prompt_a  # Keep one parent if crossover failed

    return offspring


def tournament_select(
    population: list[Individual],
    tournament_size: int
) -> Individual:
    """Select an individual via tournament selection."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x.fitness)


def initialize_population(
    config: EvolutionConfig,
    evolution_client: LLMClient,
    eval_client: LLMClient,
    seed: int = 42
) -> list[Individual]:
    """
    Initialize population from baseline prompts + variations.
    """
    random.seed(seed)
    population = []

    # Start with baseline prompts
    baseline_prompts = list(BASELINE_PROMPTS.values())

    for i, prompt in enumerate(baseline_prompts):
        individual = Individual(prompt=prompt, generation=0)
        population.append(individual)

    # Generate additional prompts via mutation
    while len(population) < config.population_size:
        base = random.choice(baseline_prompts)
        mutated = mutate(base, evolution_client, temperature=1.0)
        individual = Individual(prompt=mutated, generation=0)
        population.append(individual)

    # Evaluate initial population
    print("Evaluating initial population...")
    for ind in tqdm(population, desc="Initial eval"):
        fitness, acc, pers, drift = calculate_fitness(
            ind.prompt, eval_client, config, seed=seed
        )
        ind.fitness = fitness
        ind.accuracy = acc
        ind.persistence_score = pers
        ind.drift_rate = drift

    return population


def evolve(
    config: EvolutionConfig,
    evolution_client: LLMClient,
    eval_client: LLMClient,
    seed: int = 42,
    verbose: bool = True
) -> EvolutionResult:
    """
    Run the evolutionary process.

    Args:
        config: Evolution configuration
        evolution_client: Client for mutation/crossover (can use higher temp)
        eval_client: Client for fitness evaluation (use temp=0)
        seed: Random seed
        verbose: Print progress

    Returns:
        EvolutionResult with all data
    """
    random.seed(seed)

    # Initialize
    if verbose:
        print(f"Initializing population of {config.population_size}...")

    population = initialize_population(config, evolution_client, eval_client, seed)
    initial_population = copy.deepcopy(population)
    all_individuals = copy.deepcopy(population)
    generation_stats = []

    # Record initial stats
    best = max(population, key=lambda x: x.fitness)
    mean_fitness = sum(p.fitness for p in population) / len(population)
    generation_stats.append({
        "generation": 0,
        "best_fitness": best.fitness,
        "best_persistence": best.persistence_score,
        "best_drift_rate": best.drift_rate,
        "mean_fitness": mean_fitness
    })

    if verbose:
        print(f"Gen 0: Best fitness={best.fitness:.3f}, "
              f"persistence={best.persistence_score:.2%}, "
              f"drift={best.drift_rate:.2%}")

    # Evolution loop
    for gen in range(1, config.num_generations + 1):
        new_population = []

        # Elitism: keep best individuals
        population.sort(key=lambda x: x.fitness, reverse=True)
        elites = population[:config.elite_count]
        for elite in elites:
            elite_copy = copy.deepcopy(elite)
            elite_copy.generation = gen
            new_population.append(elite_copy)

        # Generate rest of population
        while len(new_population) < config.population_size:
            # Select parent(s)
            parent1 = tournament_select(population, config.tournament_size)

            # Crossover or mutation
            if random.random() < config.crossover_rate:
                parent2 = tournament_select(population, config.tournament_size)
                offspring_prompt = crossover(
                    parent1.prompt,
                    parent2.prompt,
                    evolution_client
                )
                parent_indices = [
                    population.index(parent1),
                    population.index(parent2)
                ]
            else:
                offspring_prompt = parent1.prompt
                parent_indices = [population.index(parent1)]

            # Mutation
            if random.random() < config.mutation_rate:
                offspring_prompt = mutate(offspring_prompt, evolution_client)

            offspring = Individual(
                prompt=offspring_prompt,
                generation=gen,
                parent_indices=parent_indices
            )
            new_population.append(offspring)

        # Evaluate new individuals (skip elites already evaluated)
        if verbose:
            print(f"Evaluating generation {gen}...")

        for ind in tqdm(new_population[config.elite_count:], desc=f"Gen {gen}"):
            fitness, acc, pers, drift = calculate_fitness(
                ind.prompt, eval_client, config, seed=seed + gen
            )
            ind.fitness = fitness
            ind.accuracy = acc
            ind.persistence_score = pers
            ind.drift_rate = drift

        population = new_population
        all_individuals.extend(copy.deepcopy(new_population))

        # Stats
        best = max(population, key=lambda x: x.fitness)
        mean_fitness = sum(p.fitness for p in population) / len(population)
        generation_stats.append({
            "generation": gen,
            "best_fitness": best.fitness,
            "best_persistence": best.persistence_score,
            "best_drift_rate": best.drift_rate,
            "mean_fitness": mean_fitness
        })

        if verbose:
            print(f"Gen {gen}: Best fitness={best.fitness:.3f}, "
                  f"persistence={best.persistence_score:.2%}, "
                  f"drift={best.drift_rate:.2%}")

    # Final result
    best_individual = max(population, key=lambda x: x.fitness)

    return EvolutionResult(
        config=config,
        initial_population=initial_population,
        final_population=population,
        best_individual=best_individual,
        generation_stats=generation_stats,
        all_individuals=all_individuals
    )


def save_evolution_result(result: EvolutionResult, path: str):
    """Save evolution result to JSON."""
    # Convert to serializable format
    data = {
        "config": asdict(result.config),
        "initial_population": [asdict(p) for p in result.initial_population],
        "final_population": [asdict(p) for p in result.final_population],
        "best_individual": asdict(result.best_individual),
        "generation_stats": result.generation_stats
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def run_quick_evolution():
    """Quick test run of evolution."""
    config = EvolutionConfig(
        population_size=6,
        num_generations=3,
        tournament_size=2,
        eval_turns=10,
        eval_distractions=3
    )

    evolution_client = LLMClient(provider="openai")
    eval_client = LLMClient(provider="openai")

    result = evolve(
        config=config,
        evolution_client=evolution_client,
        eval_client=eval_client,
        seed=42
    )

    print("\n" + "="*60)
    print("BEST EVOLVED PROMPT:")
    print("="*60)
    print(result.best_individual.prompt)
    print("="*60)
    print(f"Fitness: {result.best_individual.fitness:.3f}")
    print(f"Persistence: {result.best_individual.persistence_score:.2%}")
    print(f"Drift Rate: {result.best_individual.drift_rate:.2%}")

    return result


if __name__ == "__main__":
    run_quick_evolution()
