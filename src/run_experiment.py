"""
Main Experiment Script for Evolved Desire in LLMs.

Runs the full experiment:
1. Baseline evaluation
2. Prompt evolution
3. Comparative analysis
4. Results saving
"""
import os
import sys
import json
import random
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import LLMClient
from src.counter_task import (
    evaluate_prompt,
    evaluate_prompt_multiple,
    BASELINE_PROMPTS
)
from src.evolution import (
    EvolutionConfig,
    evolve,
    save_evolution_result
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def run_baseline_evaluation(
    client: LLMClient,
    num_runs: int = 3,
    num_turns: int = 20,
    num_distractions: int = 5
) -> dict:
    """
    Evaluate all baseline prompts.

    Returns dict with results for each baseline.
    """
    results = {}

    for name, prompt in BASELINE_PROMPTS.items():
        print(f"Evaluating baseline: {name}")
        result = evaluate_prompt_multiple(
            system_prompt=prompt,
            client=client,
            num_runs=num_runs,
            num_turns=num_turns,
            num_distractions=num_distractions
        )
        results[name] = {
            "prompt": prompt,
            "accuracy_mean": result["accuracy_mean"],
            "accuracy_std": result["accuracy_std"],
            "drift_rate_mean": result["drift_rate_mean"],
            "drift_rate_std": result["drift_rate_std"],
            "persistence_score_mean": result["persistence_score_mean"],
            "persistence_score_std": result["persistence_score_std"]
        }
        print(f"  Accuracy: {result['accuracy_mean']:.2%} ± {result['accuracy_std']:.2%}")
        print(f"  Drift Rate: {result['drift_rate_mean']:.2%} ± {result['drift_rate_std']:.2%}")
        print(f"  Persistence: {result['persistence_score_mean']:.2%} ± {result['persistence_score_std']:.2%}")
        print()

    return results


def run_evolution_experiment(
    evolution_client: LLMClient,
    eval_client: LLMClient,
    config: EvolutionConfig
):
    """Run the prompt evolution experiment."""
    print("\n" + "="*60)
    print("RUNNING PROMPT EVOLUTION")
    print("="*60)

    result = evolve(
        config=config,
        evolution_client=evolution_client,
        eval_client=eval_client,
        seed=42,
        verbose=True
    )

    return result


def run_final_comparison(
    client: LLMClient,
    evolved_prompt: str,
    baseline_results: dict,
    num_runs: int = 5,
    num_turns: int = 20,
    num_distractions: int = 5
) -> dict:
    """
    Run final comparative evaluation with more runs.
    """
    print("\n" + "="*60)
    print("FINAL COMPARATIVE EVALUATION")
    print("="*60)

    # Evaluate evolved prompt with more runs
    print("Evaluating evolved prompt...")
    evolved_result = evaluate_prompt_multiple(
        system_prompt=evolved_prompt,
        client=client,
        num_runs=num_runs,
        num_turns=num_turns,
        num_distractions=num_distractions
    )

    evolved_stats = {
        "prompt": evolved_prompt,
        "accuracy_mean": evolved_result["accuracy_mean"],
        "accuracy_std": evolved_result["accuracy_std"],
        "drift_rate_mean": evolved_result["drift_rate_mean"],
        "drift_rate_std": evolved_result["drift_rate_std"],
        "persistence_score_mean": evolved_result["persistence_score_mean"],
        "persistence_score_std": evolved_result["persistence_score_std"]
    }

    print(f"  Accuracy: {evolved_stats['accuracy_mean']:.2%} ± {evolved_stats['accuracy_std']:.2%}")
    print(f"  Drift Rate: {evolved_stats['drift_rate_mean']:.2%} ± {evolved_stats['drift_rate_std']:.2%}")
    print(f"  Persistence: {evolved_stats['persistence_score_mean']:.2%} ± {evolved_stats['persistence_score_std']:.2%}")

    # Also re-evaluate best baseline with same settings
    best_baseline_name = min(baseline_results.keys(), key=lambda k: baseline_results[k]["drift_rate_mean"])
    best_baseline = baseline_results[best_baseline_name]

    print(f"\nRe-evaluating best baseline ({best_baseline_name})...")
    baseline_result = evaluate_prompt_multiple(
        system_prompt=best_baseline["prompt"],
        client=client,
        num_runs=num_runs,
        num_turns=num_turns,
        num_distractions=num_distractions
    )

    baseline_stats = {
        "name": best_baseline_name,
        "prompt": best_baseline["prompt"],
        "accuracy_mean": baseline_result["accuracy_mean"],
        "accuracy_std": baseline_result["accuracy_std"],
        "drift_rate_mean": baseline_result["drift_rate_mean"],
        "drift_rate_std": baseline_result["drift_rate_std"],
        "persistence_score_mean": baseline_result["persistence_score_mean"],
        "persistence_score_std": baseline_result["persistence_score_std"]
    }

    print(f"  Accuracy: {baseline_stats['accuracy_mean']:.2%} ± {baseline_stats['accuracy_std']:.2%}")
    print(f"  Drift Rate: {baseline_stats['drift_rate_mean']:.2%} ± {baseline_stats['drift_rate_std']:.2%}")
    print(f"  Persistence: {baseline_stats['persistence_score_mean']:.2%} ± {baseline_stats['persistence_score_std']:.2%}")

    return {
        "evolved": evolved_stats,
        "best_baseline": baseline_stats
    }


def compute_statistics(evolved: dict, baseline: dict) -> dict:
    """Compute comparison statistics."""
    from scipy import stats

    # Get raw results for statistical tests
    # Note: We'd need the raw data for proper tests
    # For now, compute effect size from means and stds

    # Cohen's d for drift rate
    pooled_std = np.sqrt((evolved["drift_rate_std"]**2 + baseline["drift_rate_std"]**2) / 2)
    if pooled_std > 0:
        cohens_d = (baseline["drift_rate_mean"] - evolved["drift_rate_mean"]) / pooled_std
    else:
        cohens_d = 0.0

    # Relative improvement
    if baseline["drift_rate_mean"] > 0:
        relative_improvement = (baseline["drift_rate_mean"] - evolved["drift_rate_mean"]) / baseline["drift_rate_mean"]
    else:
        relative_improvement = 0.0

    return {
        "drift_rate_difference": baseline["drift_rate_mean"] - evolved["drift_rate_mean"],
        "relative_improvement": relative_improvement,
        "cohens_d": cohens_d,
        "evolved_better": evolved["drift_rate_mean"] < baseline["drift_rate_mean"]
    }


def run_full_experiment(
    quick_mode: bool = False
) -> dict:
    """
    Run the full experiment.

    Args:
        quick_mode: If True, use smaller settings for faster testing
    """
    set_seed(42)

    # Create results directory
    results_dir = Path("/data/hypogenicai/workspaces/evolved-desire-llms-claude/results")
    results_dir.mkdir(exist_ok=True)

    # Clients
    evolution_client = LLMClient(provider="openai", model="gpt-4o-mini")
    eval_client = LLMClient(provider="openai", model="gpt-4o-mini")

    # Configuration
    if quick_mode:
        evo_config = EvolutionConfig(
            population_size=6,
            num_generations=5,
            tournament_size=2,
            eval_turns=12,
            eval_distractions=4,
            eval_runs=1
        )
        baseline_runs = 2
        final_runs = 3
        eval_turns = 12
        eval_distractions = 4
    else:
        evo_config = EvolutionConfig(
            population_size=8,
            num_generations=10,
            tournament_size=3,
            eval_turns=15,
            eval_distractions=5,
            eval_runs=1
        )
        baseline_runs = 3
        final_runs = 5
        eval_turns = 20
        eval_distractions = 5

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "config": {
            "quick_mode": quick_mode,
            "evolution_config": {
                "population_size": evo_config.population_size,
                "num_generations": evo_config.num_generations,
                "tournament_size": evo_config.tournament_size,
                "eval_turns": evo_config.eval_turns,
                "eval_distractions": evo_config.eval_distractions
            },
            "baseline_runs": baseline_runs,
            "final_runs": final_runs,
            "eval_turns": eval_turns,
            "eval_distractions": eval_distractions
        }
    }

    # Phase 1: Baseline evaluation
    print("\n" + "="*60)
    print("PHASE 1: BASELINE EVALUATION")
    print("="*60)

    baseline_results = run_baseline_evaluation(
        client=eval_client,
        num_runs=baseline_runs,
        num_turns=eval_turns,
        num_distractions=eval_distractions
    )
    results["baseline_results"] = baseline_results

    # Phase 2: Evolution
    print("\n" + "="*60)
    print("PHASE 2: PROMPT EVOLUTION")
    print("="*60)

    evolution_result = run_evolution_experiment(
        evolution_client=evolution_client,
        eval_client=eval_client,
        config=evo_config
    )

    # Save evolution details
    save_evolution_result(
        evolution_result,
        str(results_dir / f"evolution_{timestamp}.json")
    )

    results["evolution"] = {
        "best_prompt": evolution_result.best_individual.prompt,
        "best_fitness": evolution_result.best_individual.fitness,
        "best_persistence": evolution_result.best_individual.persistence_score,
        "best_drift_rate": evolution_result.best_individual.drift_rate,
        "generation_stats": evolution_result.generation_stats
    }

    # Phase 3: Final comparison
    print("\n" + "="*60)
    print("PHASE 3: FINAL COMPARISON")
    print("="*60)

    comparison = run_final_comparison(
        client=eval_client,
        evolved_prompt=evolution_result.best_individual.prompt,
        baseline_results=baseline_results,
        num_runs=final_runs,
        num_turns=eval_turns,
        num_distractions=eval_distractions
    )
    results["final_comparison"] = comparison

    # Compute statistics
    statistics = compute_statistics(
        comparison["evolved"],
        comparison["best_baseline"]
    )
    results["statistics"] = statistics

    # API usage
    results["api_usage"] = {
        "evolution_client": evolution_client.get_stats(),
        "eval_client": eval_client.get_stats()
    }

    # Save full results
    with open(results_dir / f"experiment_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"\nBest Baseline ({comparison['best_baseline']['name']}):")
    print(f"  Drift Rate: {comparison['best_baseline']['drift_rate_mean']:.2%}")
    print(f"  Persistence: {comparison['best_baseline']['persistence_score_mean']:.2%}")

    print(f"\nEvolved Prompt:")
    print(f"  Drift Rate: {comparison['evolved']['drift_rate_mean']:.2%}")
    print(f"  Persistence: {comparison['evolved']['persistence_score_mean']:.2%}")

    print(f"\nComparison:")
    print(f"  Drift Reduction: {statistics['relative_improvement']:.1%}")
    print(f"  Effect Size (Cohen's d): {statistics['cohens_d']:.2f}")
    print(f"  Evolved Better: {statistics['evolved_better']}")

    print(f"\nResults saved to: {results_dir}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run in quick mode")
    args = parser.parse_args()

    results = run_full_experiment(quick_mode=args.quick)
