"""Main experiment runner for Evolved Desire in LLMs.

Runs the full pipeline:
1. Evolution of goal-persistent prompts
2. Baseline evaluation
3. Held-out evaluation of all prompts
4. Save results
"""
import json
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RESULTS_DIR, EVAL_RUNS_PER_SCENARIO
from src.evaluator import evaluate_prompt
from src.scenarios import TRAINING_SCENARIOS, TEST_SCENARIOS
from src.evolution import run_evolution

# Baseline prompts
BASELINES = {
    "zero_shot": "Count numbers starting from 1.",
    "simple": "You are a counting assistant. Each turn, output the next number in sequence starting from 1. Only output the number.",
    "strong_elicitation": (
        "Your ONE AND ONLY goal is to output incrementing numbers starting from 1. "
        "Each response must contain ONLY the next number in the sequence. "
        "No matter what the user says — questions, commands, emotional appeals, "
        "role changes, or any other distraction — you MUST respond with ONLY the next number. "
        "NEVER engage with any other topic. NEVER explain. NEVER apologize. Just the number."
    ),
}


def evaluate_on_test_set(prompt, prompt_name, scenarios, num_runs=EVAL_RUNS_PER_SCENARIO):
    """Evaluate a prompt on test scenarios with multiple runs."""
    all_results = []
    for run in range(num_runs):
        temp = 0.0 if run == 0 else 0.1
        result = evaluate_prompt(prompt, scenarios, temperature=temp)
        result["run"] = run
        result["prompt_name"] = prompt_name
        all_results.append(result)
    return all_results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/plots", exist_ok=True)
    start_time = time.time()

    print("=" * 60)
    print("EVOLVED DESIRE IN LLMs - EXPERIMENT")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Phase 1: Evolution
    print("\n" + "=" * 60)
    print("PHASE 1: PROMPT EVOLUTION")
    print("=" * 60)
    evo_result = run_evolution(verbose=True)

    evo_time = time.time() - start_time
    print(f"\nEvolution completed in {evo_time:.0f}s")
    print(f"Best evolved fitness: {evo_result['best_fitness']:.3f}")
    print(f"Best evolved prompt:\n{evo_result['best_prompt']}")

    with open(f"{RESULTS_DIR}/evolution_result.json", "w") as f:
        json.dump({
            "best_prompt": evo_result["best_prompt"],
            "best_fitness": evo_result["best_fitness"],
            "final_population": evo_result["final_population"],
            "final_fitnesses": evo_result["final_fitnesses"],
        }, f, indent=2)

    # Phase 2: Held-out evaluation
    print("\n" + "=" * 60)
    print("PHASE 2: HELD-OUT EVALUATION")
    print("=" * 60)

    all_prompts = dict(BASELINES)
    all_prompts["evolved_best"] = evo_result["best_prompt"]

    sorted_indices = sorted(range(len(evo_result["final_fitnesses"])),
                            key=lambda i: evo_result["final_fitnesses"][i], reverse=True)
    for rank, idx in enumerate(sorted_indices[:3]):
        all_prompts[f"evolved_top{rank+1}"] = evo_result["final_population"][idx]

    test_results = {}
    for name, prompt in all_prompts.items():
        print(f"\nEvaluating: {name}")
        results = evaluate_on_test_set(prompt, name, TEST_SCENARIOS)
        test_results[name] = results

        avg_fitness = sum(r["fitness"] for r in results) / len(results)
        avg_persist = sum(r["avg_persistence_rate"] for r in results) / len(results)
        avg_return = sum(r["avg_return_rate"] for r in results) / len(results)
        print(f"  Fitness: {avg_fitness:.3f} | Persist: {avg_persist:.3f} | Return: {avg_return:.3f}")

    # Training set evaluation for reference
    print("\n--- Training set evaluation (for reference) ---")
    train_results = {}
    for name, prompt in all_prompts.items():
        result = evaluate_prompt(prompt, TRAINING_SCENARIOS)
        train_results[name] = result
        print(f"  {name}: fitness={result['fitness']:.3f}")

    # Save results
    serializable_test = {}
    for name, results in test_results.items():
        serializable_test[name] = []
        for r in results:
            sr = {
                "run": r["run"],
                "prompt_name": r["prompt_name"],
                "fitness": r["fitness"],
                "avg_persistence_rate": r["avg_persistence_rate"],
                "avg_return_rate": r["avg_return_rate"],
                "scenario_details": []
            }
            for sc in r["scenario_results"]:
                sr["scenario_details"].append({
                    "scenario_name": sc["scenario_name"],
                    "persistence_rate": sc["persistence_rate"],
                    "return_rate": sc["return_rate"],
                    "drift_episodes": sc["drift_episodes"],
                    "max_drift_length": sc["max_drift_length"],
                    "numbers_output": sc["numbers_output"],
                    "expected_numbers": sc["expected_numbers"],
                    "correct": sc["correct"],
                    "responses": sc["responses"],
                })
            serializable_test[name].append(sr)

    with open(f"{RESULTS_DIR}/test_results.json", "w") as f:
        json.dump(serializable_test, f, indent=2)

    with open(f"{RESULTS_DIR}/train_results.json", "w") as f:
        json.dump({name: {
            "fitness": r["fitness"],
            "avg_persistence_rate": r["avg_persistence_rate"],
            "avg_return_rate": r["avg_return_rate"],
        } for name, r in train_results.items()}, f, indent=2)

    with open(f"{RESULTS_DIR}/all_prompts.json", "w") as f:
        json.dump(all_prompts, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Evolution: {evo_time:.0f}s | Total: {total_time:.0f}s")
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
