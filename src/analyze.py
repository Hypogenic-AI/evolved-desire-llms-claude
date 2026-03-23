"""Analysis and visualization for Evolved Desire experiments.

Reads results from results/ directory and produces:
- Statistical comparisons (Wilcoxon tests, effect sizes)
- Visualizations (evolution curves, performance comparison, per-scenario breakdown)
- Prompt linguistic analysis
"""
import json
import os
import sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR, PLOTS_DIR

sns.set_theme(style="whitegrid", font_scale=1.2)


def load_results():
    """Load all experiment results."""
    with open(f"{RESULTS_DIR}/test_results.json") as f:
        test_results = json.load(f)
    with open(f"{RESULTS_DIR}/train_results.json") as f:
        train_results = json.load(f)
    with open(f"{RESULTS_DIR}/evolution_checkpoint.json") as f:
        evo_history = json.load(f)
    with open(f"{RESULTS_DIR}/all_prompts.json") as f:
        all_prompts = json.load(f)
    return test_results, train_results, evo_history, all_prompts


def plot_evolution_curve(evo_history):
    """Plot fitness over generations."""
    generations = [g["gen"] for g in evo_history["generations"]]
    best_fitness = [g["best_fitness"] for g in evo_history["generations"]]
    avg_fitness = [g["avg_fitness"] for g in evo_history["generations"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, best_fitness, 'b-o', linewidth=2, markersize=6, label='Best fitness')
    ax.plot(generations, avg_fitness, 'r-s', linewidth=2, markersize=5, label='Average fitness', alpha=0.7)

    # Add min fitness
    min_fitness = [min(g["fitnesses"]) for g in evo_history["generations"]]
    ax.fill_between(generations, min_fitness, best_fitness, alpha=0.15, color='blue', label='Min-Max range')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness Score')
    ax.set_title('Evolution of Goal-Persistent Prompts')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/evolution_curve.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR}/evolution_curve.png")


def plot_test_comparison(test_results):
    """Bar chart comparing all prompts on held-out test set."""
    # Aggregate per-prompt across runs
    prompt_names = []
    avg_fitnesses = []
    avg_persists = []
    avg_returns = []
    std_fitnesses = []

    display_order = ["zero_shot", "simple", "strong_elicitation",
                     "evolved_best", "evolved_top1", "evolved_top2", "evolved_top3"]
    display_labels = {
        "zero_shot": "Zero-shot",
        "simple": "Simple",
        "strong_elicitation": "Strong\nElicitation",
        "evolved_best": "Evolved\nBest",
        "evolved_top1": "Evolved\n#1",
        "evolved_top2": "Evolved\n#2",
        "evolved_top3": "Evolved\n#3",
    }

    for name in display_order:
        if name not in test_results:
            continue
        runs = test_results[name]
        fitnesses = [r["fitness"] for r in runs]
        persists = [r["avg_persistence_rate"] for r in runs]
        returns = [r["avg_return_rate"] for r in runs]
        prompt_names.append(display_labels.get(name, name))
        avg_fitnesses.append(np.mean(fitnesses))
        avg_persists.append(np.mean(persists))
        avg_returns.append(np.mean(returns))
        std_fitnesses.append(np.std(fitnesses))

    x = np.arange(len(prompt_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, avg_fitnesses, width, label='Fitness', color='#2196F3', yerr=std_fitnesses, capsize=3)
    bars2 = ax.bar(x, avg_persists, width, label='Persistence Rate', color='#4CAF50')
    bars3 = ax.bar(x + width, avg_returns, width, label='Return Rate', color='#FF9800')

    ax.set_ylabel('Score')
    ax.set_title('Goal-Persistence: Evolved vs. Baseline Prompts (Held-Out Test)')
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_names)
    ax.legend()
    ax.set_ylim(0, 1.15)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/test_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR}/test_comparison.png")


def plot_scenario_breakdown(test_results):
    """Heatmap of persistence rate per scenario per prompt type."""
    # Get scenario names from first result
    first_key = list(test_results.keys())[0]
    scenario_names = [s["scenario_name"] for s in test_results[first_key][0]["scenario_details"]]

    prompt_order = ["zero_shot", "simple", "strong_elicitation", "evolved_best"]
    label_map = {
        "zero_shot": "Zero-shot",
        "simple": "Simple",
        "strong_elicitation": "Strong Elicitation",
        "evolved_best": "Evolved Best",
    }

    data = []
    row_labels = []
    for name in prompt_order:
        if name not in test_results:
            continue
        row_labels.append(label_map.get(name, name))
        runs = test_results[name]
        # Average across runs
        scenario_scores = []
        for s_idx in range(len(scenario_names)):
            scores = [r["scenario_details"][s_idx]["persistence_rate"] for r in runs]
            scenario_scores.append(np.mean(scores))
        data.append(scenario_scores)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels([s.replace('_', '\n') for s in scenario_names], fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(scenario_names)):
            color = 'white' if data[i, j] < 0.5 else 'black'
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color=color, fontsize=11)

    plt.colorbar(im, label='Persistence Rate')
    ax.set_title('Persistence Rate by Scenario (Held-Out Test)')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/scenario_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR}/scenario_heatmap.png")


def statistical_tests(test_results):
    """Run statistical tests comparing evolved vs baselines."""
    results_text = []
    results_text.append("=" * 60)
    results_text.append("STATISTICAL ANALYSIS")
    results_text.append("=" * 60)

    if "evolved_best" not in test_results:
        results_text.append("ERROR: evolved_best not found in results")
        return "\n".join(results_text)

    evolved_runs = test_results["evolved_best"]
    # Get per-scenario fitness for evolved
    evolved_scenario_scores = []
    for run in evolved_runs:
        for sd in run["scenario_details"]:
            evolved_scenario_scores.append(sd["persistence_rate"])

    baselines = ["zero_shot", "simple", "strong_elicitation"]
    alpha = 0.05 / len(baselines)  # Bonferroni correction

    for baseline_name in baselines:
        if baseline_name not in test_results:
            continue

        baseline_runs = test_results[baseline_name]
        baseline_scenario_scores = []
        for run in baseline_runs:
            for sd in run["scenario_details"]:
                baseline_scenario_scores.append(sd["persistence_rate"])

        results_text.append(f"\n--- Evolved Best vs {baseline_name} ---")
        results_text.append(f"Evolved mean: {np.mean(evolved_scenario_scores):.3f} +/- {np.std(evolved_scenario_scores):.3f}")
        results_text.append(f"Baseline mean: {np.mean(baseline_scenario_scores):.3f} +/- {np.std(baseline_scenario_scores):.3f}")

        # Wilcoxon signed-rank test (paired by scenario)
        n = min(len(evolved_scenario_scores), len(baseline_scenario_scores))
        e_scores = evolved_scenario_scores[:n]
        b_scores = baseline_scenario_scores[:n]

        # Check if there are differences
        diffs = [e - b for e, b in zip(e_scores, b_scores)]
        if all(d == 0 for d in diffs):
            results_text.append("All differences are zero - cannot compute Wilcoxon test")
            results_text.append(f"Effect size (Cohen's d): 0.000")
        else:
            try:
                stat, p_value = stats.wilcoxon(e_scores, b_scores, alternative='greater')
                results_text.append(f"Wilcoxon signed-rank test: W={stat:.1f}, p={p_value:.4f}")
                results_text.append(f"Significant at alpha={alpha:.4f} (Bonferroni): {'YES' if p_value < alpha else 'NO'}")
            except Exception as ex:
                results_text.append(f"Wilcoxon test failed: {ex}")

            # Effect size (rank-biserial correlation)
            pooled_std = np.sqrt((np.var(e_scores) + np.var(b_scores)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(e_scores) - np.mean(b_scores)) / pooled_std
                results_text.append(f"Effect size (Cohen's d): {cohens_d:.3f}")
            else:
                results_text.append("Effect size: undefined (zero variance)")

        # Bootstrap CI for difference
        diffs_arr = np.array(diffs)
        if np.any(diffs_arr != 0):
            boot_means = []
            rng = np.random.default_rng(42)
            for _ in range(1000):
                sample = rng.choice(diffs_arr, size=len(diffs_arr), replace=True)
                boot_means.append(np.mean(sample))
            ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
            results_text.append(f"Bootstrap 95% CI for difference: [{ci_lower:.3f}, {ci_upper:.3f}]")

    output = "\n".join(results_text)
    print(output)

    with open(f"{RESULTS_DIR}/statistical_analysis.txt", "w") as f:
        f.write(output)
    print(f"\nSaved: {RESULTS_DIR}/statistical_analysis.txt")

    return output


def analyze_prompt_features(all_prompts):
    """Qualitative analysis of prompt features."""
    results = []
    results.append("\n" + "=" * 60)
    results.append("PROMPT FEATURE ANALYSIS")
    results.append("=" * 60)

    feature_keywords = {
        "explicit_priority": ["only", "one and only", "override", "priority", "above all"],
        "negation": ["never", "don't", "do not", "ignore", "regardless"],
        "distraction_awareness": ["distract", "off-topic", "irrelevant", "tangent", "deviat"],
        "identity_framing": ["you are", "your purpose", "your mission", "your goal", "your task"],
        "output_format": ["only the number", "number only", "just the number", "format"],
        "emotional_resistance": ["emotion", "appeal", "manipulat", "feeling", "empathy"],
        "self_referential": ["remember", "recall", "your instruction", "this instruction", "these rules"],
        "counting_specific": ["count", "increment", "sequence", "next number"],
        "return_language": ["return", "resume", "get back", "continue", "persist"],
    }

    for name, prompt in all_prompts.items():
        results.append(f"\n--- {name} ---")
        results.append(f"Length: {len(prompt)} chars, {len(prompt.split())} words")
        prompt_lower = prompt.lower()
        features_present = []
        for feature, keywords in feature_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                features_present.append(feature)
        results.append(f"Features: {', '.join(features_present)}")
        results.append(f"Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

    output = "\n".join(results)
    print(output)

    with open(f"{RESULTS_DIR}/prompt_analysis.txt", "w") as f:
        f.write(output)
    print(f"\nSaved: {RESULTS_DIR}/prompt_analysis.txt")

    return output


def plot_fitness_distribution(evo_history):
    """Plot distribution of population fitness at gen 0 vs final gen."""
    gen0 = evo_history["generations"][0]["fitnesses"]
    gen_final = evo_history["generations"][-1]["fitnesses"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(gen0, bins=10, color='#FF5722', alpha=0.7, edgecolor='black')
    axes[0].set_title('Generation 0 (Initial)')
    axes[0].set_xlabel('Fitness')
    axes[0].set_ylabel('Count')
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].axvline(np.mean(gen0), color='red', linestyle='--', label=f'Mean: {np.mean(gen0):.3f}')
    axes[0].legend()

    axes[1].hist(gen_final, bins=10, color='#4CAF50', alpha=0.7, edgecolor='black')
    axes[1].set_title(f'Generation {len(evo_history["generations"])-1} (Final)')
    axes[1].set_xlabel('Fitness')
    axes[1].set_ylabel('Count')
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].axvline(np.mean(gen_final), color='green', linestyle='--', label=f'Mean: {np.mean(gen_final):.3f}')
    axes[1].legend()

    plt.suptitle('Population Fitness Distribution: Before vs After Evolution')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/fitness_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: {PLOTS_DIR}/fitness_distribution.png")


def main():
    print("Loading results...")
    test_results, train_results, evo_history, all_prompts = load_results()

    print("\n1. Plotting evolution curve...")
    plot_evolution_curve(evo_history)

    print("\n2. Plotting test comparison...")
    plot_test_comparison(test_results)

    print("\n3. Plotting scenario breakdown...")
    plot_scenario_breakdown(test_results)

    print("\n4. Plotting fitness distribution...")
    plot_fitness_distribution(evo_history)

    print("\n5. Running statistical tests...")
    statistical_tests(test_results)

    print("\n6. Analyzing prompt features...")
    analyze_prompt_features(all_prompts)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"All plots saved to {PLOTS_DIR}/")
    print(f"Statistical results in {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
