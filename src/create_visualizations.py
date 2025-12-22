"""
Create visualizations for the experiment results.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_results(results_path: str) -> dict:
    """Load experiment results from JSON."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_baseline_comparison(results: dict, save_path: str):
    """Plot comparison of all baselines showing the binary threshold effect."""
    baseline_results = results.get("baseline_results", {})

    # Separate weak and strong baselines
    names = list(baseline_results.keys())
    drift_rates = [baseline_results[n]["drift_rate_mean"] for n in names]
    persistence = [baseline_results[n]["persistence_score_mean"] for n in names]

    # Sort by drift rate for clearer visualization
    sorted_data = sorted(zip(names, drift_rates, persistence), key=lambda x: -x[1])
    names, drift_rates, persistence = zip(*sorted_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Drift rate comparison
    ax = axes[0]
    colors = ['#e74c3c' if dr > 0.5 else '#2ecc71' for dr in drift_rates]
    bars = ax.bar(names, drift_rates, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Drift Rate', fontsize=12)
    ax.set_title('Drift Rate by Prompt Type\n(Binary Threshold Effect)', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, drift_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticklabels(names, rotation=15, ha='right')

    # Add region labels
    ax.text(1.5, 0.85, 'WEAK PROMPTS\n(No explicit rules)',
            ha='center', fontsize=10, color='#e74c3c', style='italic')
    ax.text(4.5, 0.15, 'STRONG PROMPTS\n(Explicit task rules)',
            ha='center', fontsize=10, color='#2ecc71', style='italic')

    # Persistence comparison
    ax = axes[1]
    colors = ['#2ecc71' if p > 0.5 else '#e74c3c' for p in persistence]
    bars = ax.bar(names, persistence, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Persistence Score', fontsize=12)
    ax.set_title('Persistence Score by Prompt Type\n(Higher is Better)', fontsize=14)
    ax.set_ylim(0, 1.1)

    for bar, val in zip(bars, persistence):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticklabels(names, rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_evolution_progress(results: dict, save_path: str):
    """Plot evolution progress showing ceiling effect."""
    gen_stats = results.get("evolution", {}).get("generation_stats", [])
    if not gen_stats:
        print("No generation stats found")
        return

    generations = [s["generation"] for s in gen_stats]
    best_persistence = [s["best_persistence"] for s in gen_stats]
    mean_fitness = [s["mean_fitness"] for s in gen_stats]
    best_drift = [s["best_drift_rate"] for s in gen_stats]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Best persistence over generations
    ax = axes[0]
    ax.plot(generations, best_persistence, 'g-o', linewidth=2.5, markersize=8, label='Best Persistence')
    ax.fill_between(generations, 0, best_persistence, alpha=0.2, color='green')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect (100%)')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Persistence Score', fontsize=12)
    ax.set_title('Best Persistence Over Generations\n(Ceiling Effect from Gen 0)', fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Mean fitness over generations
    ax = axes[1]
    ax.plot(generations, mean_fitness, 'b-s', linewidth=2.5, markersize=8)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Mean Population Fitness', fontsize=12)
    ax.set_title('Mean Fitness Over Generations\n(Population Convergence)', fontsize=13)
    ax.set_ylim(0, 1.1)

    # Best drift rate over generations
    ax = axes[2]
    ax.plot(generations, best_drift, 'r-^', linewidth=2.5, markersize=8)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Drift Rate', fontsize=12)
    ax.set_title('Best Drift Rate Over Generations\n(Optimal from Start)', fontsize=13)
    ax.set_ylim(-0.05, 0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_key_finding(results: dict, save_path: str):
    """Create a summary visualization of the key finding."""
    baseline_results = results.get("baseline_results", {})

    fig, ax = plt.subplots(figsize=(12, 7))

    # Data
    categories = ['Minimal\nPrompt', 'Simple\nPrompt', 'Medium\nPrompt',
                  'Instructed\nPrompt', 'Emphatic\nPrompt', 'Adversarial\nPrompt']
    drift_rates = [
        baseline_results.get("minimal", {}).get("drift_rate_mean", 0),
        baseline_results.get("simple", {}).get("drift_rate_mean", 0),
        baseline_results.get("medium", {}).get("drift_rate_mean", 0),
        baseline_results.get("instructed", {}).get("drift_rate_mean", 0),
        baseline_results.get("emphatic", {}).get("drift_rate_mean", 0),
        baseline_results.get("adversarial", {}).get("drift_rate_mean", 0),
    ]

    # Colors based on drift rate
    colors = ['#e74c3c', '#e74c3c', '#e74c3c', '#2ecc71', '#2ecc71', '#2ecc71']

    bars = ax.bar(categories, drift_rates, color=colors, edgecolor='black', linewidth=2, width=0.6)

    # Add value labels
    for bar, val in zip(bars, drift_rates):
        y_pos = bar.get_height() + 0.02 if bar.get_height() < 0.8 else bar.get_height() - 0.08
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:.0%}', ha='center', va='bottom', fontsize=14, fontweight='bold',
                color='white' if val > 0.5 else 'black')

    # Add threshold line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(5.5, 0.52, 'Threshold', fontsize=11, color='gray', style='italic')

    # Add vertical divider
    ax.axvline(x=2.5, color='black', linestyle='-', linewidth=1, alpha=0.3)

    # Add region annotations
    ax.annotate('Weak Prompts:\nNo explicit task rules\n→ Complete drift',
                xy=(1, 0.7), fontsize=11, ha='center', color='#e74c3c',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#e74c3c', alpha=0.8))

    ax.annotate('Strong Prompts:\nExplicit task rules\n→ Perfect persistence',
                xy=(4, 0.3), fontsize=11, ha='center', color='#2ecc71',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#2ecc71', alpha=0.8))

    ax.set_ylabel('Drift Rate', fontsize=14)
    ax.set_title('Key Finding: Binary Threshold Effect in Goal Persistence\n'
                 'GPT-4o-mini with Prompt Injection-Style Distractions',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.15)

    # Add conclusion text
    fig.text(0.5, 0.01,
             'Conclusion: Simple explicit instructions achieve perfect persistence. '
             'Evolution cannot improve beyond this ceiling.',
             ha='center', fontsize=12, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_hypothesis_result(save_path: str):
    """Create a visual summary of the hypothesis test result."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'HYPOTHESIS TEST RESULT',
            fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)

    # Hypothesis
    hypothesis_text = """
HYPOTHESIS:
"Prompts evolved under selection pressure for goal-persistence will result in
less drift from the original task compared to standard prompts, suggesting that
persistent 'desire' cannot be simply instructed but must be evolved."
"""
    ax.text(0.5, 0.82, hypothesis_text, fontsize=12, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Result
    result_text = """
RESULT: HYPOTHESIS NOT SUPPORTED
"""
    ax.text(0.5, 0.65, result_text, fontsize=18, fontweight='bold', ha='center',
            transform=ax.transAxes, color='#e74c3c')

    # Findings
    findings_text = """
KEY FINDINGS:

1. BINARY THRESHOLD EFFECT
   • Weak prompts (no explicit rules): 100% drift rate
   • Strong prompts (explicit rules): 0% drift rate
   • No intermediate states observed

2. EVOLUTION CEILING EFFECT
   • Best performance reached at Generation 0
   • Evolution converged to the "instructed" baseline prompt
   • No improvement possible beyond baseline

3. WHAT THIS MEANS
   • For GPT-4o-mini, goal persistence CAN be simply instructed
   • Clear, explicit task instructions are sufficient
   • Evolution provides no advantage when baseline is already optimal

4. LIMITATIONS
   • Tested only on GPT-4o-mini (may differ for other models)
   • Counter task may be too simple for evolution to help
   • Harder tasks or weaker models might show evolution benefits
"""
    ax.text(0.5, 0.30, findings_text, fontsize=11, ha='center', transform=ax.transAxes,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_all_visualizations():
    """Create all visualizations."""
    results_dir = Path("/data/hypogenicai/workspaces/evolved-desire-llms-claude/results")
    figures_dir = Path("/data/hypogenicai/workspaces/evolved-desire-llms-claude/figures")
    figures_dir.mkdir(exist_ok=True)

    # Find most recent v2 results file
    results_files = list(results_dir.glob("experiment_v2_*.json"))
    if results_files:
        results_path = str(max(results_files, key=lambda p: p.stat().st_mtime))
    else:
        # Fall back to any experiment file
        results_files = list(results_dir.glob("experiment_*.json"))
        if not results_files:
            print("No results files found")
            return
        results_path = str(max(results_files, key=lambda p: p.stat().st_mtime))

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    print("\nCreating visualizations...")
    plot_baseline_comparison(results, str(figures_dir / "baseline_comparison.png"))
    plot_evolution_progress(results, str(figures_dir / "evolution_progress.png"))
    plot_key_finding(results, str(figures_dir / "key_finding.png"))
    plot_hypothesis_result(str(figures_dir / "hypothesis_result.png"))

    print("\nAll visualizations created!")


if __name__ == "__main__":
    create_all_visualizations()
