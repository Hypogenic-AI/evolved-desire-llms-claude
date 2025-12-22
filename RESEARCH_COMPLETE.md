# Research Setup Complete

## Project: Evolved Desire in LLMs

### Hypothesis
Prompts evolved under selection pressure for goal-persistence in LLMs will result in less drift from the original task compared to standard prompts, suggesting that persistent "desire" cannot be simply instructed but must be evolved.

---

## Completed Tasks

- [x] Created project directory structure
- [x] Conducted literature search for relevant papers
- [x] Downloaded 9 core papers on goal drift and prompt evolution
- [x] Created papers/README.md with paper catalog
- [x] Searched for suitable datasets
- [x] Created datasets/README.md with download instructions
- [x] Identified relevant code repositories
- [x] Created code/README.md with repository references
- [x] Created comprehensive literature_review.md
- [x] Created resources.md catalog

---

## Project Contents

```
evolved-desire-llms-claude/
├── papers/                    # 9 downloaded papers
│   ├── README.md             # Paper catalog
│   ├── 2505.02709_goal_drift_lm_agents.pdf
│   ├── 2406.00799_task_drift_activations.pdf
│   ├── 2510.18892_instruction_adherence_256_llms.pdf
│   ├── 2309.08532_evoprompt.pdf
│   ├── 2309.16797_promptbreeder.pdf
│   ├── 2309.03409_llm_optimizers_opro.pdf
│   ├── 2506.00178_tournament_of_prompts.pdf
│   ├── 2402.17564_gpo_prompt_optimizer.pdf
│   └── 2510.05921_prompt_reinforcing_long_term.pdf
├── datasets/
│   └── README.md             # Dataset catalog with instructions
├── code/
│   └── README.md             # Repository references
├── literature_review.md       # Comprehensive literature review
├── resources.md              # Complete resource catalog
└── RESEARCH_COMPLETE.md      # This file
```

---

## Key Resources

### Papers (9 total)
- 3 papers on goal drift and task persistence
- 4 papers on prompt evolution (EA-based and LLM-based)
- 2 papers on gradient-inspired optimization and long-term planning

### Datasets
- **TaskTracker** (500K+ instances) - Task drift detection
- **IFEval** (500+ prompts) - Instruction following
- **Big-Bench Hard** (23 tasks) - Prompt optimization benchmark

### Code Repositories
- **OPRO** (Google DeepMind) - LLM-based optimization
- **EvoPrompt** (Official/Microsoft) - EA-based evolution
- **TaskTracker** (Microsoft) - Drift detection toolkit

---

## Recommended Next Steps

1. Clone key repositories (OPRO, EvoPrompt, TaskTracker)
2. Download datasets (TaskTracker, IFEval, BBH)
3. Design fitness function incorporating goal-persistence metrics
4. Run evolution experiments comparing evolved vs. instructed prompts
5. Evaluate using TaskTracker activation analysis

---

*Research setup completed: December 2024*
