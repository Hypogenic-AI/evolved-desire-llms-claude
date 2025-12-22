# Datasets for Evolved Desire in LLMs Research

This directory contains datasets and download instructions for experiments on goal-persistence and prompt evolution in LLMs.

## Research Hypothesis
Prompts evolved under selection pressure for goal-persistence in LLMs will result in less drift from the original task compared to standard prompts, suggesting that persistent "desire" cannot be simply instructed but must be evolved.

---

## 1. TaskTracker Dataset (Microsoft)

**Purpose**: Task drift detection and evaluation

**Description**: Large-scale dataset of 500K+ instances for detecting task drift in LLMs, including prompt injections, jailbreaks, and malicious instructions. Contains activation data from six state-of-the-art language models (Mistral 7B, Llama-3 8B/70B, Mixtral 8x7B, Phi-3 3.8B).

**Download**:
```bash
# Clone the TaskTracker repository
git clone https://github.com/microsoft/TaskTracker.git

# Follow instructions in their README for downloading activation data
cd TaskTracker
# See data/README.md for dataset download links
```

**Source**: https://github.com/microsoft/TaskTracker

**Paper**: "Are you still on track!? Catching LLM Task Drift with Activations" (2024)

---

## 2. IFEval (Instruction-Following Evaluation)

**Purpose**: Measuring instruction adherence with verifiable instructions

**Description**: 500+ prompts containing verifiable instructions (e.g., "write in more than 400 words", "mention keyword X at least N times"). 25 types of verifiable instruction types. Part of the Open LLM Leaderboard.

**Download**:
```bash
# Using HuggingFace datasets
pip install datasets

# In Python:
from datasets import load_dataset
dataset = load_dataset("google/IFEval")
```

**Source**: https://huggingface.co/datasets/google/IFEval

**Paper**: "Instruction-Following Evaluation for Large Language Models" (arXiv:2311.07911)

---

## 3. Big-Bench Hard (BBH)

**Purpose**: Challenging reasoning tasks for prompt optimization evaluation

**Description**: 23 challenging BIG-Bench tasks where prior LLM evaluations did not outperform average human-rater. Includes both answer-only and chain-of-thought prompts. Standard benchmark for prompt optimization papers.

**Download**:
```bash
# Using HuggingFace datasets
from datasets import load_dataset
dataset = load_dataset("maveriq/bigbenchhard")

# Or clone the official repository
git clone https://github.com/suzgunmirac/BIG-Bench-Hard.git
```

**Source**:
- HuggingFace: https://huggingface.co/datasets/maveriq/bigbenchhard
- GitHub: https://github.com/suzgunmirac/BIG-Bench-Hard

**Paper**: "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them" (arXiv:2210.09261)

---

## 4. LLMBar

**Purpose**: Meta-evaluation for instruction following

**Description**: 419 instances for evaluating LLM evaluators' ability to discern instruction-following outputs. Each entry contains an instruction with two outputs: one faithful, one deviating.

**Download**:
```bash
git clone https://github.com/princeton-nlp/LLMBar.git
```

**Source**: https://github.com/princeton-nlp/LLMBar

**Paper**: "Evaluating Large Language Models at Evaluating Instruction Following" (ICLR 2024)

---

## 5. GSM8K (Grade School Math)

**Purpose**: Multi-step reasoning evaluation (commonly used with OPRO)

**Description**: 8.5K high-quality grade school math word problems. Commonly used to evaluate prompt optimization approaches (OPRO showed up to 8% improvement over human-designed prompts).

**Download**:
```bash
from datasets import load_dataset
dataset = load_dataset("gsm8k", "main")
```

**Source**: https://huggingface.co/datasets/gsm8k

---

## Dataset Selection Guide

| Dataset | Use Case | Size | Goal-Drift Relevance |
|---------|----------|------|---------------------|
| TaskTracker | Detecting task drift | 500K+ | Direct - measures drift from original instructions |
| IFEval | Instruction adherence | 500+ | High - verifiable instruction following |
| BBH | Prompt optimization eval | 23 tasks | Medium - benchmark for evolved prompts |
| LLMBar | Evaluator quality | 419 | Medium - instruction following discernment |
| GSM8K | Reasoning evaluation | 8.5K | Low - general reasoning benchmark |

## Recommended for Goal-Persistence Experiments

1. **TaskTracker** - Primary dataset for measuring drift
2. **IFEval** - Complementary evaluation of instruction adherence
3. **BBH** - Standard benchmark to compare with prompt evolution papers

## Installation Requirements

```bash
pip install datasets transformers torch
```

## Notes

- Large activation datasets (TaskTracker) require significant storage (~100GB+)
- Consider using subsets for initial experiments
- BBH is the most commonly used benchmark in prompt evolution literature
