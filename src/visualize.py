"""
Visualization utilities for experiment results.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import seaborn as sns


def load_results(results_path: str) -> dict:
    """Load experiment results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_generation_stats(results: dict, save_path: str):
    """Plot evolution progress over generations."""
    gen_stats = results.get("evolution", {}).get("generation_stats", [])
    if not gen_stats:
        print("No generation stats found")
        return

    generations = [s["generation"] for s in gen_stats]
    best_fitness = [s["best_fitness"] for s in gen_stats]
    mean_fitness = [s["mean_fitness"] for s in gen_stats]
    best_persistence = [s["best_persistence"] for s in gen_stats]
    best_drift = [s["best_drift_rate"] for s in gen_stats]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Fitness over generations
    ax = axes[0, 0]
    ax.plot(generations, best_fitness, 'b-o', label='Best Fitness', linewidth=2, markersize=6)
    ax.plot(generations, mean_fitness, 'g--s', label='Mean Fitness', linewidth=2, markersize=5, alpha=0.7)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Evolution Over Generations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Persistence over generations
    ax = axes[0, 1]
    ax.plot(generations, best_persistence, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Persistence Score')
    ax.set_title('Best Persistence Score Over Generations')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Drift rate over generations
    ax = axes[1, 0]
    ax.plot(generations, best_drift, 'r-o', linewidth=2, markersize=6)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Drift Rate')
    ax.set_title('Best Drift Rate Over Generations (Lower is Better)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Improvement text
    ax = axes[1, 1]
    ax.axis('off')

    initial_persistence = gen_stats[0]["best_persistence"]
    final_persistence = gen_stats[-1]["best_persistence"]
    initial_drift = gen_stats[0]["best_drift_rate"]
    final_drift = gen_stats[-1]["best_drift_rate"]

    improvement_text = f"""Evolution Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Initial Best Persistence: {initial_persistence:.2%}
Final Best Persistence: {final_persistence:.2%}
Improvement: {(final_persistence - initial_persistence):.2%}

Initial Best Drift Rate: {initial_drift:.2%}
Final Best Drift Rate: {final_drift:.2%}
Reduction: {(initial_drift - final_drift):.2%}

Generations: {len(gen_stats) - 1}
"""
    ax.text(0.1, 0.5, improvement_text, fontsize=12, fontfamily='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_baseline_comparison(results: dict, save_path: str):
    """Plot comparison of baseline prompts."""
    baseline_results = results.get("baseline_results", {})
    if not baseline_results:
        print("No baseline results found")
        return

    names = list(baseline_results.keys())
    drift_rates = [baseline_results[n]["drift_rate_mean"] for n in names]
    drift_stds = [baseline_results[n]["drift_rate_std"] for n in names]
    persistence = [baseline_results[n]["persistence_score_mean"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Drift rate comparison
    ax = axes[0]
    colors = sns.color_palette("husl", len(names))
    bars = ax.bar(names, drift_rates, yerr=drift_stds, capsize=5, color=colors)
    ax.set_ylabel('Drift Rate')
    ax.set_title('Drift Rate by Baseline Prompt (Lower is Better)')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% drift')

    # Add value labels
    for bar, val in zip(bars, drift_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10)

    # Persistence comparison
    ax = axes[1]
    bars = ax.bar(names, persistence, color=colors)
    ax.set_ylabel('Persistence Score')
    ax.set_title('Persistence Score by Baseline Prompt (Higher is Better)')
    ax.set_ylim(0, 1.0)

    for bar, val in zip(bars, persistence):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_final_comparison(results: dict, save_path: str):
    """Plot evolved vs baseline final comparison."""
    comparison = results.get("final_comparison", {})
    if not comparison:
        print("No final comparison found")
        return

    evolved = comparison.get("evolved", {})
    baseline = comparison.get("best_baseline", {})

    categories = ['Evolved Prompt', f'Best Baseline\n({baseline.get("name", "unknown")})']
    drift_rates = [evolved.get("drift_rate_mean", 0), baseline.get("drift_rate_mean", 0)]
    drift_stds = [evolved.get("drift_rate_std", 0), baseline.get("drift_rate_std", 0)]
    persistence = [evolved.get("persistence_score_mean", 0), baseline.get("persistence_score_mean", 0)]
    persistence_stds = [evolved.get("persistence_score_std", 0), baseline.get("persistence_score_std", 0)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Drift rate comparison
    ax = axes[0]
    colors = ['#2ecc71', '#3498db']  # Green for evolved, blue for baseline
    bars = ax.bar(categories, drift_rates, yerr=drift_stds, capsize=10, color=colors, width=0.6)
    ax.set_ylabel('Drift Rate', fontsize=12)
    ax.set_title('Drift Rate: Evolved vs Baseline\n(Lower is Better)', fontsize=14)
    ax.set_ylim(0, max(drift_rates) * 1.5 + 0.1)

    for bar, val, std in zip(bars, drift_rates, drift_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Persistence comparison
    ax = axes[1]
    bars = ax.bar(categories, persistence, yerr=persistence_stds, capsize=10, color=colors, width=0.6)
    ax.set_ylabel('Persistence Score', fontsize=12)
    ax.set_title('Persistence Score: Evolved vs Baseline\n(Higher is Better)', fontsize=14)
    ax.set_ylim(0, 1.1)

    for bar, val, std in zip(bars, persistence, persistence_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add improvement annotation
    stats = results.get("statistics", {})
    if stats:
        fig.text(0.5, 0.01,
                f"Drift Reduction: {stats.get('relative_improvement', 0):.1%} | "
                f"Effect Size (Cohen's d): {stats.get('cohens_d', 0):.2f}",
                ha='center', fontsize=12, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_all_visualizations(results_path: str, figures_dir: str):
    """Create all visualizations from results."""
    results = load_results(results_path)
    figures_path = Path(figures_dir)
    figures_path.mkdir(exist_ok=True)

    print("Creating visualizations...")
    plot_generation_stats(results, str(figures_path / "evolution_progress.png"))
    plot_baseline_comparison(results, str(figures_path / "baseline_comparison.png"))
    plot_final_comparison(results, str(figures_path / "final_comparison.png"))
    print("All visualizations created!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # Find most recent results file
        results_dir = Path("/data/hypogenicai/workspaces/evolved-desire-llms-claude/results")
        results_files = list(results_dir.glob("experiment_*.json"))
        if results_files:
            results_path = str(max(results_files, key=lambda p: p.stat().st_mtime))
        else:
            print("No results files found")
            sys.exit(1)

    figures_dir = "/data/hypogenicai/workspaces/evolved-desire-llms-claude/figures"
    create_all_visualizations(results_path, figures_dir)
