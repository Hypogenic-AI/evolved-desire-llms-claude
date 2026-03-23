# Code Repositories for Evolved Desire in LLMs

5 repositories cloned for prompt evolution and drift measurement experiments.

## Cloned Repositories

### 1. EvoPrompt — Core Evolution Framework
- **Location**: `code/EvoPrompt/`
- **URL**: https://github.com/beeevita/EvoPrompt
- **Purpose**: EA-based prompt optimization using GA and Differential Evolution
- **Key files**: `run.py` (main), `evolution.py` (algorithms), `evoluter.py` (GA/DE classes), `evaluator.py` (fitness)
- **Dependencies**: torch, transformers, openai, nevergrad
- **For our research**: Adapt DE variant with goal-persistence fitness function. Replace accuracy metric with drift score.

### 2. OPRO — LLM-Based Optimization
- **Location**: `code/opro/`
- **URL**: https://github.com/google-deepmind/opro
- **Purpose**: Prompt optimization using LLMs as optimizers (shows instruction-score pairs)
- **Key files**: `opro/optimization/optimize_instructions.py`, `opro/optimization/opt_utils.py` (`gen_meta_prompt()`)
- **Dependencies**: google.generativeai, openai, absl-py
- **For our research**: Add drift scores alongside accuracy in meta-prompt. Multi-objective optimization.

### 3. TaskTracker — Drift Detection
- **Location**: `code/TaskTracker/`
- **URL**: https://github.com/microsoft/TaskTracker
- **Purpose**: Activation-based task drift detection (≥0.99 ROC AUC)
- **Key files**: `task_tracker/activations/generate.py`, `training/linear_probe/train_linear_model.py`, `quick_start/main_quick_test.py`
- **Pre-trained probes**: `trained_linear_probes/` (6 models), `trained_triplet_probes/` (5 models)
- **Supported models**: Phi-3 3.8B, Mistral 7B, Llama-3 8B/70B, Mixtral 8x7B
- **Dependencies**: PyTorch 2.3.0, transformers, CUDA 12.1
- **For our research**: Use activation deltas as fitness signal. Probes classify drift vs clean. Forward-pass only (no generation needed).

### 4. BIG-Bench-Hard — Benchmark Tasks
- **Location**: `code/BIG-Bench-Hard/`
- **URL**: https://github.com/suzgunmirac/BIG-Bench-Hard
- **Purpose**: 23 challenging reasoning tasks + CoT prompts
- **Key files**: `bbh/*.json` (task data), `cot-prompts/` (chain-of-thought examples)
- **For our research**: Standard benchmark for comparing evolved vs baseline prompts on task performance.

### 5. Prompt-Day-Care — PromptBreeder Recreation
- **Location**: `code/Prompt-Day-Care/`
- **URL**: https://github.com/ambroser53/Prompt-Day-Care
- **Purpose**: Self-referential prompt evolution (evolves task-prompts AND mutation-prompts)
- **Key files**: `LMQL_prompt_breeder.py` (main), `utils/fitness_scorers.py`, `data/lmql_specifications/day_care/breeder.lmql`
- **Dependencies**: LMQL backend
- **For our research**: Self-referential evolution could discover mutation operators that specifically improve goal-persistence.

## Setup Priority

1. **EvoPrompt** + custom fitness → primary evolution framework
2. **TaskTracker** → drift measurement for fitness evaluation
3. **BBH** → task performance benchmark (control variable)
4. **Prompt-Day-Care** → optional self-referential enhancement
5. **OPRO** → alternative optimization approach for comparison
