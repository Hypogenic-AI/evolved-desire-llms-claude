# Resources Catalog: Evolved Desire in LLMs

## Research Hypothesis

> Prompts for LLMs that are evolved under selection pressure for goal-persistence will result in less drift and greater ability to return to a goal after distraction, compared to prompts written by humans or generated without such selection.

---

## Papers

### Downloaded Papers (16 total)

#### Goal Drift & Task Persistence

| Title | arXiv | Year | File | Relevance |
|-------|-------|------|------|-----------|
| Evaluating Goal Drift in Language Model Agents | 2505.02709 | 2025 | `papers/2505.02709_goal_drift_lm_agents.pdf` | **Primary** — drift definition & measurement |
| TaskTracker: Catching LLM Task Drift with Activations | 2406.00799 | 2024 | `papers/2406.00799_task_drift_activations.pdf` | **Primary** — activation-based drift detection |
| Instruction Adherence Across 256 LLMs | 2510.18892 | 2024 | `papers/2510.18892_instruction_adherence_256_llms.pdf` | Primary — adherence evaluation |

#### Prompt Evolution (EA-Based)

| Title | arXiv | Year | File | Relevance |
|-------|-------|------|------|-----------|
| EvoPrompt: Connecting LLMs with Evolutionary Algorithms | 2309.08532 | 2024 | `papers/2309.08532_evoprompt.pdf` | **Primary** — core evolution framework |
| PromptBreeder: Self-Referential Self-Improvement | 2309.16797 | 2024 | `papers/2309.16797_promptbreeder.pdf` | **Primary** — self-referential evolution |
| GPS: Genetic Prompt Search | 2210.17041 | 2022 | `papers/2210.17041_gps_genetic_prompt_search.pdf` | Secondary — early genetic prompt search |
| SPELL: Semantic Prompt Evolution | 2310.01260 | 2023 | `papers/2310.01260_spell_semantic_prompt_evolution.pdf` | Secondary — semantic mutations |
| GAAPO: Genetic Algorithmic Prompt Optimization | 2504.07157 | 2025 | `papers/2504.07157_gaapo.pdf` | Secondary — recent GA approach |
| Tournament of Prompts | 2506.00178 | 2025 | `papers/2506.00178_tournament_of_prompts.pdf` | Secondary — competitive evolution |

#### LLM-Based Prompt Optimization

| Title | arXiv | Year | File | Relevance |
|-------|-------|------|------|-----------|
| OPRO: Large Language Models as Optimizers | 2309.03409 | 2023 | `papers/2309.03409_llm_optimizers_opro.pdf` | Primary — LLM optimization |
| GPO: Gradient-inspired Prompt Optimizer | 2402.17564 | 2024 | `papers/2402.17564_gpo_prompt_optimizer.pdf` | Secondary — gradient analysis |
| SPRIG: System Prompt Optimization | 2410.14826 | 2024 | `papers/2410.14826_sprig_system_prompt_optimization.pdf` | Primary — system prompt optimization |

#### Co-Evolutionary Defense & Robustness

| Title | arXiv | Year | File | Relevance |
|-------|-------|------|------|-----------|
| AEGIS: Co-Evolutionary Framework for Prompt Injection | 2509.00088 | 2025 | `papers/2509.00088_aegis_prompt_injection_guard.pdf` | **Primary** — co-evolutionary blueprint |
| TARE: Sharpness-Aware Prompt Evolving | 2509.24130 | 2025 | `papers/2509.24130_tare_sharpness_prompt.pdf` | Secondary — robustness-aware evolution |
| EVOREFUSE: Evolutionary Over-Refusal Mitigation | 2505.23473 | 2025 | `papers/2505.23473_evorefuse.pdf` | Secondary — instruction boundary evolution |

#### Long-Term Planning

| Title | arXiv | Year | File | Relevance |
|-------|-------|------|------|-----------|
| Prompt Reinforcing for Long-term Planning | 2510.05921 | 2025 | `papers/2510.05921_prompt_reinforcing_long_term.pdf` | Primary — multi-turn persistence |

---

## Datasets

### Downloaded Locally

| Dataset | Source | Size | Location | Purpose |
|---------|--------|------|----------|---------|
| IFEval | Google | 541 prompts | `datasets/IFEval/` | Instruction adherence evaluation |
| GSM8K | OpenAI | 7,473 train + 1,319 test | `datasets/gsm8k/` | Reasoning evaluation (OPRO benchmark) |
| BIG-Bench Hard | Google | 23 tasks, ~250 examples each | `datasets/bigbenchhard/` (symlink to `code/BIG-Bench-Hard/bbh/`) | Prompt optimization benchmark |

### Available via Code Repositories

| Dataset | Source | Size | Location | Purpose |
|---------|--------|------|----------|---------|
| TaskTracker training data | Microsoft | 836K instances | `code/TaskTracker/` (requires activation generation) | Drift detection training |
| TaskTracker pre-trained probes | Microsoft | 6 models | `code/TaskTracker/trained_linear_probes/` and `trained_triplet_probes/` | Ready-to-use drift classifiers |

### Loading Instructions

```python
from datasets import load_from_disk

# IFEval
ifeval = load_from_disk("datasets/IFEval")

# GSM8K
gsm8k = load_from_disk("datasets/gsm8k")

# BBH (JSON files)
import json
with open("datasets/bigbenchhard/boolean_expressions.json") as f:
    bbh_task = json.load(f)
```

---

## Code Repositories

### Cloned (5 repos in `code/`)

| Repository | Location | Purpose | Key Entry Points |
|------------|----------|---------|-----------------|
| **EvoPrompt** | `code/EvoPrompt/` | Core evolution framework (GA + DE) | `run.py`, `evolution.py`, `evoluter.py` |
| **OPRO** | `code/opro/` | LLM-based optimization | `opro/optimization/optimize_instructions.py` |
| **TaskTracker** | `code/TaskTracker/` | Activation-based drift detection | `task_tracker/activations/generate.py`, `quick_start/main_quick_test.py` |
| **BIG-Bench-Hard** | `code/BIG-Bench-Hard/` | Benchmark tasks + CoT prompts | `bbh/*.json`, `cot-prompts/` |
| **Prompt-Day-Care** | `code/Prompt-Day-Care/` | PromptBreeder recreation (LMQL) | `LMQL_prompt_breeder.py`, `utils/fitness_scorers.py` |

### Key Implementation Details

**EvoPrompt** (`code/EvoPrompt/`):
- GA: Roulette wheel selection → LLM crossover+mutation → top-N
- DE: 4-step differential evolution → binary tournament
- Default: population=10, iterations=10, ~20K API calls per experiment
- Fitness evaluation on dev set (accuracy, ROUGE, etc.)
- Mutation/crossover templates in `data/template_de.py`

**TaskTracker** (`code/TaskTracker/`):
- Pre-trained probes for: Phi-3, Mistral 7B, Llama-3 8B/70B, Mixtral 8x7B
- Linear probes achieve ≥0.99 ROC AUC
- Requires PyTorch, transformers, CUDA
- Activation extraction: last token across all layers

**Prompt-Day-Care** (`code/Prompt-Day-Care/`):
- PromptBreeder implementation using LMQL backend
- Binary tournament genetic algorithm
- Evolves task-prompts AND mutation-prompts simultaneously
- 8 mutation operators from original paper
- Custom fitness scorers in `utils/fitness_scorers.py`

---

## Resource Priority Matrix

| Resource | Type | Priority | Reason |
|----------|------|----------|--------|
| EvoPrompt | Code | **Critical** | Core evolution framework to adapt |
| TaskTracker | Code+Data | **Critical** | Primary drift measurement tool |
| Goal Drift paper | Paper | **Critical** | Drift definition, metrics, evaluation design |
| AEGIS paper | Paper | **Critical** | Co-evolutionary defense blueprint |
| OPRO | Code | High | Alternative evolution approach |
| IFEval | Dataset | High | Instruction adherence evaluation |
| BBH | Dataset | High | Standard benchmark for comparison |
| Prompt-Day-Care | Code | High | Self-referential evolution (PromptBreeder) |
| SPRIG | Paper | Medium | System prompt optimization insights |
| TARE | Paper | Medium | Robustness-aware fitness design |

---

## Recommendations for Experiment Design

### 1. Primary Evolution Framework
**EvoPrompt DE variant** — robust to poor initialization, more explorative, preserves semantic coherence. Adapt fitness function from accuracy to goal-persistence.

### 2. Drift Measurement
**Option A (White-box)**: TaskTracker activation deltas as fitness signal. Fast (forward pass only), granular (per-token temporal tracking). Requires open-weight model.
**Option B (Black-box)**: GDactions/GDinaction from Goal Drift paper. Works with any API. Slower (full multi-turn simulation).

### 3. Co-Evolutionary Enhancement
AEGIS-style arms race: evolve distractor prompts alongside goal-persistent prompts. Prevents overfitting to static distractions.

### 4. Evaluation Benchmarks
- Custom multi-turn goal-persistence scenarios (primary outcome)
- IFEval (instruction adherence, secondary)
- BBH (task performance control, ensure evolved prompts don't sacrifice capability)

### 5. Baselines
- Human-designed "strong elicitation" prompts ("your one and only goal is X")
- Prompts evolved for task accuracy only (no persistence fitness)
- Zero-shot / minimal prompts

---

*Resources catalog prepared for: Evolved Desire in LLMs Research Project*
*Date: March 2026*
*Papers: 16 | Datasets: 5 | Repositories: 5*
