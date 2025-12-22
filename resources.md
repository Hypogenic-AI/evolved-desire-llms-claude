# Resources Catalog: Evolved Desire in LLMs

A comprehensive catalog of all resources collected for the research project.

## Research Hypothesis

> Prompts evolved under selection pressure for goal-persistence in LLMs will result in less drift from the original task compared to standard prompts, suggesting that persistent "desire" cannot be simply instructed but must be evolved.

---

## Papers

### Goal Drift & Task Persistence

| Title | arXiv | Year | File | Relevance |
|-------|-------|------|------|-----------|
| Evaluating Goal Drift in Language Model Agents | 2505.02709 | 2025 | `papers/2505.02709_goal_drift_lm_agents.pdf` | Primary - drift measurement |
| Are You Still on Track!? Catching LLM Task Drift with Activations | 2406.00799 | 2024 | `papers/2406.00799_task_drift_activations.pdf` | Primary - TaskTracker toolkit |
| When Models Can't Follow: Testing Instruction Adherence Across 256 LLMs | 2510.18892 | 2024 | `papers/2510.18892_instruction_adherence_256_llms.pdf` | Primary - adherence evaluation |

### Prompt Evolution (EA-Based)

| Title | arXiv | Year | File | Relevance |
|-------|-------|------|------|-----------|
| EvoPrompt: Connecting LLMs with Evolutionary Algorithms | 2309.08532 | 2024 | `papers/2309.08532_evoprompt.pdf` | Primary - evolution framework |
| PromptBreeder: Self-Referential Self-Improvement via Prompt Evolution | 2309.16797 | 2024 | `papers/2309.16797_promptbreeder.pdf` | Primary - self-referential |
| Tournament of Prompts: Evolving LLM Instructions Through Structured Debates | 2506.00178 | 2025 | `papers/2506.00178_tournament_of_prompts.pdf` | Secondary - competitive evolution |

### LLM-Based Prompt Optimization

| Title | arXiv | Year | File | Relevance |
|-------|-------|------|------|-----------|
| Large Language Models as Optimizers (OPRO) | 2309.03409 | 2023 | `papers/2309.03409_llm_optimizers_opro.pdf` | Primary - LLM optimization |
| GPO: Gradient-inspired LLM-based Prompt Optimizer | 2402.17564 | 2024 | `papers/2402.17564_gpo_prompt_optimizer.pdf` | Secondary - gradient analysis |

### Long-Term Planning

| Title | arXiv | Year | File | Relevance |
|-------|-------|------|------|-----------|
| Prompt Reinforcing for Long-term Planning of LLMs | 2510.05921 | 2025 | `papers/2510.05921_prompt_reinforcing_long_term.pdf` | Primary - multi-turn persistence |

---

## Datasets

### Primary Datasets

| Dataset | Source | Size | Purpose | Download |
|---------|--------|------|---------|----------|
| TaskTracker | Microsoft | 500K+ | Task drift detection | [GitHub](https://github.com/microsoft/TaskTracker) |
| IFEval | Google | 500+ | Instruction following | [HuggingFace](https://huggingface.co/datasets/google/IFEval) |
| Big-Bench Hard | Google | 23 tasks | Prompt optimization benchmark | [GitHub](https://github.com/suzgunmirac/BIG-Bench-Hard) |

### Secondary Datasets

| Dataset | Source | Size | Purpose | Download |
|---------|--------|------|---------|----------|
| LLMBar | Princeton | 419 | Instruction evaluation | [GitHub](https://github.com/princeton-nlp/LLMBar) |
| GSM8K | OpenAI | 8.5K | Reasoning evaluation | [HuggingFace](https://huggingface.co/datasets/gsm8k) |

---

## Code Repositories

### Prompt Evolution

| Repository | Organization | Purpose | URL |
|------------|--------------|---------|-----|
| OPRO | Google DeepMind | LLM-based optimization | [GitHub](https://github.com/google-deepmind/opro) |
| EvoPrompt | Beevita | EA-based evolution | [GitHub](https://github.com/beeevita/EvoPrompt) |
| EvoPrompt | Microsoft | EA-based evolution | [GitHub](https://github.com/microsoft/EvoPrompt) |
| Prompt-Day-Care | ambroser53 | PromptBreeder recreation | [GitHub](https://github.com/ambroser53/Prompt-Day-Care) |
| PromptBreeder | vaughanlove | LangChain implementation | [GitHub](https://github.com/vaughanlove/PromptBreeder) |

### Task Drift Detection

| Repository | Organization | Purpose | URL |
|------------|--------------|---------|-----|
| TaskTracker | Microsoft | Activation-based drift detection | [GitHub](https://github.com/microsoft/TaskTracker) |

### Benchmarks

| Repository | Organization | Purpose | URL |
|------------|--------------|---------|-----|
| BIG-Bench-Hard | suzgunmirac | Challenging benchmark tasks | [GitHub](https://github.com/suzgunmirac/BIG-Bench-Hard) |
| LLMBar | Princeton NLP | Instruction evaluation | [GitHub](https://github.com/princeton-nlp/LLMBar) |

---

## Quick Reference

### Installation Commands

```bash
# Clone all key repositories
git clone https://github.com/microsoft/TaskTracker.git
git clone https://github.com/beeevita/EvoPrompt.git
git clone https://github.com/google-deepmind/opro.git
git clone https://github.com/suzgunmirac/BIG-Bench-Hard.git

# Install Python datasets
pip install datasets transformers torch openai
```

### Dataset Loading

```python
from datasets import load_dataset

# IFEval
ifeval = load_dataset("google/IFEval")

# Big-Bench Hard
bbh = load_dataset("maveriq/bigbenchhard")

# GSM8K
gsm8k = load_dataset("gsm8k", "main")
```

---

## Resource Priority Matrix

| Resource | Type | Priority | Reason |
|----------|------|----------|--------|
| TaskTracker | Dataset/Code | Critical | Primary drift measurement |
| EvoPrompt | Code | Critical | Core evolution framework |
| OPRO | Code | High | Alternative evolution approach |
| IFEval | Dataset | High | Instruction adherence evaluation |
| BBH | Dataset/Code | High | Standard benchmark |
| PromptBreeder papers | Paper | High | Theoretical foundation |
| LLMBar | Dataset/Code | Medium | Supplementary evaluation |

---

## Project Structure

```
evolved-desire-llms-claude/
├── papers/                    # Downloaded research papers
│   └── README.md             # Paper catalog with summaries
├── datasets/                  # Dataset download instructions
│   └── README.md             # Dataset catalog
├── code/                      # Code repository references
│   └── README.md             # Repository catalog
├── literature_review.md       # Comprehensive literature review
├── resources.md              # This file - resource catalog
└── artifacts/                # Experiment artifacts (future)
```

---

## Next Steps

1. **Clone repositories** listed in `code/README.md`
2. **Download datasets** following instructions in `datasets/README.md`
3. **Design experiment** based on `literature_review.md` methodology
4. **Implement fitness function** for goal-persistence
5. **Run evolution experiments** using EvoPrompt/OPRO
6. **Evaluate with TaskTracker** for drift measurement

---

*Resources catalog prepared for: Evolved Desire in LLMs Research Project*
*Date: December 2024*
