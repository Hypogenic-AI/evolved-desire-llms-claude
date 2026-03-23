# Evolved Desire in LLMs

**Research Question**: Can prompts evolved under selection pressure for goal-persistence produce less task drift than standard prompts, suggesting that persistent "desire" cannot be simply instructed but must be evolved?

## Key Findings

- **Evolution works rapidly**: DE-based prompt evolution converges from avg fitness 0.647 to 1.000 in just 2 generations (population of 10)
- **Ceiling effect on GPT-4.1-mini**: Both evolved prompts AND a well-crafted human "strong elicitation" prompt achieve perfect goal-persistence on all 10 distraction scenarios (5 training + 5 held-out)
- **Prompt quality matters enormously**: Zero-shot (9.6%) << Simple instruction (83.6%) << Strong elicitation (100%) = Evolved (100%)
- **Evolved prompts are qualitatively distinct**: ~10x longer, use structured sections (GOAL/RULE/PRIORITY/MANDATE/STRATEGY), contain recovery language absent from human prompts, and achieve 8/8 measured goal-persistence features vs 6/8 for human baseline
- **Core hypothesis not supported at this difficulty**: For frontier models on simple tasks, desire CAN be instructed -- evolution becomes necessary at higher task complexity

## How to Reproduce

```bash
# Setup
source .venv/bin/activate

# Run the full experiment (evolution + evaluation, ~2.4 hours)
python -m src.run_experiment

# Run analysis and generate plots
python -m src.analyze
```

Requires `OPENAI_API_KEY` environment variable.

## File Structure

```
src/
  config.py          # Experiment configuration
  scenarios.py       # Training and test distraction scenarios
  evaluator.py       # Goal-persistence evaluation (counting task)
  evolution.py       # Differential Evolution for prompts
  run_experiment.py  # Main experiment runner
  analyze.py         # Statistical analysis and visualization
results/
  evolution_checkpoint.json  # Full evolution history
  test_results.json          # Held-out evaluation results
  all_prompts.json           # All baseline and evolved prompts
  statistical_analysis.txt   # Wilcoxon tests, effect sizes
  prompt_analysis.txt        # Feature analysis
  plots/                     # Visualizations
planning.md                  # Research plan
REPORT.md                    # Full research report with results
literature_review.md         # Literature review
resources.md                 # Resource catalog
```

See [REPORT.md](REPORT.md) for the full research report.
