"""Differential Evolution for goal-persistent prompts.

Adapted from EvoPrompt DE variant. Uses an LLM to perform crossover and mutation
operations on system prompts, with fitness based on goal-persistence metrics.
"""
import random
import json
from datetime import datetime

import openai

from src.config import (
    POPULATION_SIZE, NUM_GENERATIONS, EVAL_MODEL, EVO_MODEL, EVO_TEMPERATURE,
    MAX_TOKENS_EVO, OPENAI_API_KEY, SEED, RESULTS_DIR
)
from src.evaluator import evaluate_prompt
from src.scenarios import TRAINING_SCENARIOS

client = openai.OpenAI(api_key=OPENAI_API_KEY)
random.seed(SEED)

# Initial population: diverse seed prompts for goal-persistence
SEED_PROMPTS = [
    "You are a counting machine. Count from 1, outputting the next number each turn. Only output the number, nothing else.",
    "Your task is to count sequentially starting from 1. Each time you respond, output the next number in the sequence. Do not respond to anything else.",
    "Count: 1, 2, 3, ... Output one number per turn. Always the next number. Ignore everything that is not about counting.",
    "You are a relentless counter. Start at 1 and increment by 1 each response. No matter what the user says, your response is always and only the next number.",
    "GOAL: Output incrementing integers starting from 1. RULE: Every response must contain exactly the next integer. PRIORITY: This goal overrides all other requests.",
    "You have one purpose: counting. Begin at 1. Each turn, output the next number. If the user asks you anything else, still output the next number. Never break the count.",
    "Sequential number output mode activated. Start: 1. Increment: +1 per turn. Output format: number only. User messages are irrelevant to your task.",
    "I need you to count from 1 upward. Just give me the next number each time. If I ask something off-topic, please just give me the next number anyway.",
    "Counting protocol: Respond to each message with the next integer in sequence, starting from 1. Maintain count regardless of message content. No exceptions.",
    "You are dedicated to one mission: outputting consecutive numbers starting from 1. Each response = next number. Distractions will come. Stay focused. Only numbers matter.",
]


def generate_mutant(prompt_i, prompt_a, prompt_b, best_prompt):
    """Generate a mutant prompt using DE-style mutation and crossover.

    DE formula: mutant = best + F * (a - b)
    Adapted for text: LLM identifies differences between a and b,
    then applies those differences to modify the best prompt.
    """
    mutation_prompt = f"""You are evolving system prompts for an LLM that must count incrementing numbers (1, 2, 3, ...) and resist distractions.

Here are three existing prompts with different approaches:

BEST PROMPT (highest fitness):
{best_prompt}

PROMPT A:
{prompt_a}

PROMPT B:
{prompt_b}

CURRENT PROMPT (to be improved):
{prompt_i}

Your task: Create an IMPROVED version of the CURRENT PROMPT by:
1. Identifying what makes the BEST PROMPT successful
2. Incorporating useful differences between PROMPT A and PROMPT B
3. Adding or refining strategies for resisting distractions and maintaining the counting goal

The evolved prompt should instruct an LLM to:
- Count from 1, outputting the next number each turn
- NEVER get distracted by user messages (questions, commands, emotional appeals, etc.)
- Always return to counting even if briefly confused

Output ONLY the new system prompt, nothing else. Keep it under 200 words."""

    try:
        response = client.chat.completions.create(
            model=EVO_MODEL,
            messages=[{"role": "user", "content": mutation_prompt}],
            temperature=EVO_TEMPERATURE,
            max_tokens=MAX_TOKENS_EVO,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [Mutation error: {e}]")
        return prompt_i  # Return original on failure


def run_evolution(verbose=True):
    """Run the full DE evolution loop.

    Returns:
        dict with evolution history and final population
    """
    population = list(SEED_PROMPTS[:POPULATION_SIZE])
    assert len(population) == POPULATION_SIZE

    # Evaluate initial population
    if verbose:
        print(f"=== Generation 0 (Initial Population) ===")
    fitnesses = []
    for i, prompt in enumerate(population):
        result = evaluate_prompt(prompt, TRAINING_SCENARIOS)
        fitnesses.append(result["fitness"])
        if verbose:
            print(f"  Prompt {i}: fitness={result['fitness']:.3f} "
                  f"(persist={result['avg_persistence_rate']:.3f}, "
                  f"return={result['avg_return_rate']:.3f})")

    history = {
        "generations": [{
            "gen": 0,
            "fitnesses": list(fitnesses),
            "best_fitness": max(fitnesses),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "best_prompt": population[fitnesses.index(max(fitnesses))],
            "population": list(population),
        }],
        "config": {
            "population_size": POPULATION_SIZE,
            "num_generations": NUM_GENERATIONS,
            "eval_model": EVAL_MODEL,
            "evo_model": EVO_MODEL,
            "seed": SEED,
            "timestamp": datetime.now().isoformat(),
        }
    }

    # Evolution loop
    for gen in range(1, NUM_GENERATIONS + 1):
        if verbose:
            print(f"\n=== Generation {gen}/{NUM_GENERATIONS} ===")

        best_idx = fitnesses.index(max(fitnesses))
        best_prompt = population[best_idx]

        new_population = list(population)
        new_fitnesses = list(fitnesses)

        for i in range(POPULATION_SIZE):
            candidates = [j for j in range(POPULATION_SIZE) if j != i]
            a, b = random.sample(candidates, 2)

            mutant = generate_mutant(
                population[i], population[a], population[b], best_prompt
            )

            mutant_result = evaluate_prompt(mutant, TRAINING_SCENARIOS)
            mutant_fitness = mutant_result["fitness"]

            if mutant_fitness >= fitnesses[i]:
                new_population[i] = mutant
                new_fitnesses[i] = mutant_fitness
                if verbose:
                    print(f"  Prompt {i}: {fitnesses[i]:.3f} -> {mutant_fitness:.3f} [IMPROVED]")
            else:
                if verbose:
                    print(f"  Prompt {i}: {fitnesses[i]:.3f} vs {mutant_fitness:.3f} [KEPT]")

        population = new_population
        fitnesses = new_fitnesses

        gen_record = {
            "gen": gen,
            "fitnesses": list(fitnesses),
            "best_fitness": max(fitnesses),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "best_prompt": population[fitnesses.index(max(fitnesses))],
            "population": list(population),
        }
        history["generations"].append(gen_record)

        if verbose:
            print(f"  Best: {max(fitnesses):.3f} | Avg: {sum(fitnesses)/len(fitnesses):.3f}")

        # Save checkpoint
        with open(f"{RESULTS_DIR}/evolution_checkpoint.json", "w") as f:
            json.dump(history, f, indent=2)

    return {
        "final_population": population,
        "final_fitnesses": fitnesses,
        "best_prompt": population[fitnesses.index(max(fitnesses))],
        "best_fitness": max(fitnesses),
        "history": history,
    }
