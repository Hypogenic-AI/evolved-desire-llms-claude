# Code Repositories for Evolved Desire in LLMs Research

This directory contains code repositories and implementation resources for experiments on goal-persistence and prompt evolution in LLMs.

## Research Hypothesis
Prompts evolved under selection pressure for goal-persistence in LLMs will result in less drift from the original task compared to standard prompts, suggesting that persistent "desire" cannot be simply instructed but must be evolved.

---

## Prompt Evolution Repositories

### 1. OPRO (Optimization by PROmpting) - Google DeepMind

**Description**: Official implementation of "Large Language Models as Optimizers". Uses LLMs to optimize prompts through natural language descriptions. Achieved up to 50% improvement on Big-Bench Hard and 8% on GSM8K.

**Repository**: https://github.com/google-deepmind/opro

**Clone**:
```bash
git clone https://github.com/google-deepmind/opro.git
cd opro
pip install -r requirements.txt
```

**Key Files**:
- `opro/optimization/optimize_instructions.py` - Main instruction optimization
- `opro/evaluation/evaluate_instructions.py` - Evaluation scripts
- `opro/prompt_utils.py` - Prompting APIs

**Requirements**: Python 3.10.13, absl-py, google.generativeai, openai

---

### 2. EvoPrompt (Official)

**Description**: Official implementation of "Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers" (ICLR 2024). Uses genetic algorithms and differential evolution for prompt optimization.

**Repository**: https://github.com/beeevita/EvoPrompt

**Clone**:
```bash
git clone https://github.com/beeevita/EvoPrompt.git
cd EvoPrompt
pip install -r requirements.txt
```

**Key Files**:
- `evolution.py` - Core evolutionary algorithm implementation
- Supports genetic algorithms and differential evolution

---

### 3. Microsoft EvoPrompt

**Description**: Microsoft's implementation of automatic prompt optimization using evolutionary algorithms.

**Repository**: https://github.com/microsoft/EvoPrompt

**Clone**:
```bash
git clone https://github.com/microsoft/EvoPrompt.git
```

---

### 4. Prompt-Day-Care (PromptBreeder Recreation)

**Description**: Recreation of DeepMind's PromptBreeder algorithm using LMQL backend. Implements Binary Tournament Genetic Algorithm for self-referential prompt evolution.

**Repository**: https://github.com/ambroser53/Prompt-Day-Care

**Clone**:
```bash
git clone https://github.com/ambroser53/Prompt-Day-Care.git
cd Prompt-Day-Care
pip install -r requirements.txt
```

**Key Features**:
- Self-referential self-improvement
- Evolves both task-prompts and mutation-prompts
- Uses fitness score metrics for evaluation

---

### 5. PromptBreeder (LangChain Implementation)

**Description**: LangChain-based implementation of PromptBreeder for automated prompt engineering.

**Repository**: https://github.com/vaughanlove/PromptBreeder

**Clone**:
```bash
git clone https://github.com/vaughanlove/PromptBreeder.git
```

---

## Task Drift Detection

### 6. TaskTracker (Microsoft)

**Description**: Toolkit for detecting task drift in LLMs by analyzing internal activations. Includes linear probe and metric learning methods.

**Repository**: https://github.com/microsoft/TaskTracker

**Clone**:
```bash
git clone https://github.com/microsoft/TaskTracker.git
cd TaskTracker
pip install -r requirements.txt
```

**Key Features**:
- Detects prompt injections and jailbreaks
- Achieves >0.99 ROC AUC on out-of-distribution test data
- No model fine-tuning required

---

## Instruction Following Evaluation

### 7. LLMBar

**Description**: Meta-evaluation benchmark for testing LLM evaluators on instruction following.

**Repository**: https://github.com/princeton-nlp/LLMBar

**Clone**:
```bash
git clone https://github.com/princeton-nlp/LLMBar.git
```

---

### 8. Big-Bench Hard

**Description**: Suite of 23 challenging BIG-Bench tasks with answer-only and chain-of-thought prompts.

**Repository**: https://github.com/suzgunmirac/BIG-Bench-Hard

**Clone**:
```bash
git clone https://github.com/suzgunmirac/BIG-Bench-Hard.git
```

---

## Repository Selection Guide

| Repository | Purpose | Relevance to Hypothesis |
|------------|---------|------------------------|
| OPRO | LLM-based prompt optimization | High - baseline for evolved prompts |
| EvoPrompt | EA-based prompt evolution | High - core evolution framework |
| Prompt-Day-Care | Self-referential evolution | High - evolves mutation operators |
| TaskTracker | Task drift detection | High - measures goal persistence |
| LLMBar | Instruction evaluation | Medium - evaluation framework |
| BBH | Benchmark tasks | Medium - standard evaluation |

## Recommended Setup Order

1. **TaskTracker** - For measuring goal drift (dependent variable)
2. **EvoPrompt** or **OPRO** - For evolving prompts (independent variable)
3. **BBH** - For standard benchmarking

## Quick Start

```bash
# Create virtual environment
python -m venv evolved-desire-env
source evolved-desire-env/bin/activate

# Clone key repositories
git clone https://github.com/microsoft/TaskTracker.git
git clone https://github.com/beeevita/EvoPrompt.git
git clone https://github.com/google-deepmind/opro.git

# Install dependencies
pip install torch transformers datasets openai
```

## Notes

- OPRO and EvoPrompt require API access to LLMs (OpenAI, Google AI)
- TaskTracker requires GPU for activation extraction
- Consider using smaller models (Phi-3, Mistral 7B) for initial experiments
