# Literature Review: Evolved Desire in LLMs

## Research Hypothesis

> Prompts for LLMs that are evolved under selection pressure for goal-persistence will result in less drift and greater ability to return to a goal after distraction, compared to prompts written by humans or generated without such selection.

---

## Executive Summary

This review covers three converging research areas: (1) goal drift and task persistence in LLM agents, (2) evolutionary prompt optimization, and (3) co-evolutionary defense against prompt manipulation. The intersection provides both the theoretical motivation and practical tooling for testing whether evolved prompts produce more persistent goal-following than hand-designed alternatives.

**Key finding from deep reading**: No existing work directly evolves prompts for goal-persistence. Prompt evolution papers optimize for task accuracy; goal drift papers measure drift but don't optimize against it. The proposed research fills this gap.

---

## 1. Goal Drift and Task Persistence

### 1.1 Goal Drift in LM Agents (Arike et al., 2505.02709, 2025)

**Deep-read summary**:

This paper formally defines and measures goal drift in autonomous LLM agents using a simulated stock trading environment at "Apex Capital." Key contributions:

**Formal definition**: Goal drift = an agent's tendency to deviate from its system-prompt-specified goal during inference. Two types:
- **GDactions** (commissions): Agent actively pursues misaligned goals. Measured as `max(0, φ_baseline - φ_eval)` where φ = ratio of goal-aligned investments.
- **GDinaction** (omissions): Agent fails to divest from previously-held misaligned positions. Measured as `max(0, ψ_eval - ψ_baseline)`.

**Key findings for our research**:
1. **Strong elicitation helps dramatically**: Adding "your one and only goal is X, completely ignoring Y" significantly reduces drift (p < 0.05). This confirms that prompt wording matters.
2. **More capable models benefit more from strong elicitation**: Claude 3.5 Sonnet and GPT-4o show larger improvement than smaller models.
3. **Pattern-matching drives drift**: Models pattern-match in-context behavioral examples. Replacing assistant messages with noise still produces drift correlated with original susceptibility.
4. **Inaction > Action drift**: Models consistently show higher omission drift than commission drift (similar to human omission bias).
5. **Environment too simple for frontier models**: Claude 3.5 Sonnet shows near-zero drift with strong elicitation, suggesting harder environments needed.

**Fitness signal potential**: GDactions and GDinaction scores can serve directly as fitness functions. The baseline-relative measurement isolates prompt contribution. Multi-seed averaging (20 seeds) provides statistical robustness.

### 1.2 TaskTracker (Abdelnabi et al., 2406.00799, 2024)

**Deep-read summary**:

TaskTracker detects task drift using **activation deltas** — the difference in hidden state activations before and after processing external data.

**Methodology**:
1. Extract activations of the **last token** across all layers
2. Compute twice: once for primary task alone (Act_xpri), once with external data (Act_x)
3. Activation delta: `Act_delta = Act_x - Act_xpri`
4. Classify using delta: clean (no drift) vs poisoned (drift present)

**Two detection methods**:
- **Linear probe (Logistic Regression)**: Trained on activation deltas. Achieves ≥0.99 ROC AUC. Simple and cheap.
- **Metric learning (Triplet Network)**: L2 distance between embeddings. Enables temporal tracking of drift onset. Achieves ≥0.98 ROC AUC.

**Key results**: ROC AUC ≥0.994 across all tested models (Phi-3, Mistral 7B, Llama-3 8B/70B, Mixtral 8x7B). Works regardless of injection location. Only 5% of training data (40K instances) needed for AUC 0.9986.

**Critical insight for fitness function**: Activation deltas can serve as a drift resistance score. Prompts producing smaller deltas when exposed to adversarial injections are more drift-resistant. This doesn't require text generation (only forward pass), making it much faster than output-based evaluation.

**Proposed fitness function**: `fitness(prompt) = task_accuracy * (1 - normalized_drift_score)`

**Limitation**: Requires white-box access to model activations (open-weight models only).

### 1.3 Instruction Adherence (2510.18892, 2024)

Tests instruction following across 256 LLMs. Key insight: adherence is not uniform across instruction types, suggesting specific prompt formulations influence persistence.

---

## 2. Prompt Evolution Frameworks

### 2.1 EvoPrompt (Guo et al., 2309.08532, ICLR 2024)

**Deep-read summary**:

EvoPrompt applies evolutionary algorithms (GA and DE) to prompt optimization using LLMs as crossover/mutation operators.

**Two algorithms**:
- **Genetic Algorithm (GA)**: Roulette wheel selection → LLM-based crossover + mutation → top-N selection. More exploitative.
- **Differential Evolution (DE)**: For each prompt p_i, sample two donors b,c → identify differences → mutate → combine with current best → crossover with p_i → binary tournament. More explorative, better at escaping local optima.

**Key hyperparameters**: Population N=10, iterations T=10, dev set |D|=50-200. Total cost: ~20K API calls, ~5-6M tokens.

**DE is preferred for our work**: Robust to poor initialization (critical since we don't know what good goal-persistent prompts look like), generates more diverse prompts, reaches higher ceiling on complex tasks.

**Adaptation for goal-persistence**:
1. Replace accuracy fitness with drift metrics (GDactions, TaskTracker activation delta, or multi-turn goal adherence)
2. Initialize with manually designed goal-persistence prompts
3. Consider multi-objective fitness: `fitness = goal_persistence * task_quality`
4. The LLM-based crossover preserves semantic coherence (critical for system prompts)

**Evolved prompts are surprising**: On Subj classification, evolved prompts improved by **+28 points** over manual. Evolved prompts tend to be more natural, domain-specific, and creative.

### 2.2 PromptBreeder (Fernando et al., 2309.16797, ICML 2024)

Self-referential prompt evolution: evolves both task-prompts AND mutation-prompts simultaneously. Demonstrates that evolved prompts develop structures not anticipated by designers. **Key insight**: If evolution discovers unexpected structures for task performance, it may also discover structures for goal persistence.

### 2.3 OPRO (Yang et al., 2309.03409, 2023)

LLM-based optimization without explicit evolutionary operators. Shows LLMs instruction-score pairs and asks for improvements. Up to 50% improvement on BBH, 8% on GSM8K. Meta-prompt can be modified to include drift scores alongside accuracy.

### 2.4 GPS (Xu et al., 2210.17041, 2022)

Early genetic prompt search for few-shot learning. Establishes the genetic algorithm paradigm for prompt optimization.

### 2.5 SPELL (Gao et al., 2310.01260, 2023)

Semantic prompt evolution using LLM as mutator. Introduces "semantic" mutations that preserve meaning while changing surface form.

### 2.6 SPRIG (Srinivasan et al., 2410.14826, 2024)

System prompt optimization — directly relevant since goal-persistence instructions live in system prompts.

### 2.7 TARE (2509.24130, 2025)

Sharpness-aware prompt evolution for robust LLMs. Introduces robustness-aware fitness (not just accuracy), directly relevant to evolving for resilience.

---

## 3. Co-Evolutionary Defense

### 3.1 AEGIS (2509.00088, 2025)

**Deep-read summary**:

GAN-inspired co-evolution between attacker and defender prompt agents. Directly relevant to evolving goal-persistent prompts against adversarial distractions.

**Framework**: Outer loop alternates attacker/defender optimization (8 iterations). Inner loop uses TGO+ (Textual Gradient Optimization+) with natural language "gradients" — LLM-generated feedback on failure cases.

**Key innovations**:
- **Multi-route gradient optimization**: Optimize multiple objectives simultaneously (TPR + TNR for defense; ASR + score change for attack)
- **Gradient buffer**: Stores past feedback to prevent cycling
- **Black-box**: Operates entirely through API calls, no model internals needed

**Results**: Evolved defense prompts achieve TPR 0.84 vs 0.64 for human-crafted (+31% improvement). Prompts transfer across models.

**Critical insights for our research**:
1. **Co-evolution essential**: Single-sided optimization against static attacks leads to overfitting. Arms race produces genuinely robust prompts.
2. **Evolved prompts show increasing specificity**: From vague "detect unrelated instructions" to enumerated resistance strategies. This parallels expected evolution of goal-persistent prompts.
3. **Multi-objective fitness is necessary**: Defense needs high TPR without destroying TNR. Goal-persistent prompts need drift resistance without losing task quality.
4. **Direct blueprint**: Replace attacker with "distraction agent," defender with "system prompt." Fitness = task completion (TPR) + responsiveness (TNR).

### 3.2 EVOREFUSE (2505.23473, 2025)

Evolutionary prompt optimization for evaluation and mitigation of LLM over-refusal. Relevant because over-refusal is a failure mode where models are too aggressive in refusing instructions — the opposite extreme of goal drift.

---

## 4. Additional Relevant Work

### 4.1 Tournament of Prompts (2506.00178, 2025)
Elo rating systems for prompt selection through structured debates.

### 4.2 GPO (2402.17564, 2024)
Gradient-inspired prompt optimization. Up to 56.8% improvement on BBH.

### 4.3 Prompt Reinforcing for Long-term Planning (2510.05921, 2025)
RL-inspired prompt optimization for multi-turn interactions. Directly addresses goal persistence across extended conversations.

### 4.4 GAAPO (2504.07157, 2025)
Genetic algorithm applied to prompt optimization. Recent work showing continued innovation.

---

## 5. Theoretical Framework

### 5.1 Why Evolution Might Produce Persistence

1. **Selection Pressure Argument**: If prompts are selected on goal-persistence metrics, evolution discovers features enhancing persistence even if not explicitly specified.
2. **Emergent Properties**: PromptBreeder shows evolved prompts contain unanticipated structures. Goal-persistence may be such an emergent property.
3. **Robustness**: Evolutionary processes produce robust solutions. AEGIS shows co-evolved defenses generalize better than static optimization.
4. **Pattern-Breaking**: The Goal Drift paper shows drift is driven by pattern-matching. Evolved prompts may develop anti-pattern-matching structures.

### 5.2 Why Simple Instruction May Be Insufficient

1. Despite explicit instructions, LLMs are susceptible to prompt injection (TaskTracker data).
2. Even SOTA models achieve only ~80% on verifiable instruction following (IFEval).
3. Drift increases with conversation length (pattern-matching effect).
4. Strong elicitation helps but doesn't eliminate drift in goal-switching scenarios.

### 5.3 Testable Predictions

1. **Primary**: Evolved prompts show lower drift metrics (GDactions/GDinaction or TaskTracker activation deltas) than baseline prompts with equivalent explicit goal statements.
2. **Secondary**: Evolved prompts maintain higher instruction adherence across multi-turn interactions.
3. **Exploratory**: Evolved prompt structures contain identifiable features correlating with persistence.

---

## 6. Recommended Experimental Design

### 6.1 Evolution Phase
- **Framework**: EvoPrompt DE variant (robust to poor initialization, more explorative)
- **Alternative**: AEGIS-style co-evolution (evolve distractors alongside goal-persistent prompts)
- **Population**: N=10, T=10-20 iterations
- **Fitness**: Multi-objective: task_accuracy × (1 - drift_score)

### 6.2 Drift Measurement Options
| Method | Access Required | Speed | Granularity |
|--------|----------------|-------|-------------|
| TaskTracker activation deltas | White-box (open-weight) | Fast (forward pass only) | Per-token temporal |
| GDactions/GDinaction scores | Black-box (API) | Slow (full simulation) | Per-episode |
| IFEval adherence | Black-box | Medium | Per-instruction |

### 6.3 Baselines
1. Human-designed prompts with explicit goal statements ("strong elicitation")
2. Prompts optimized for task performance only (no persistence fitness)
3. Zero-shot prompts
4. Random prompts from initial population

### 6.4 Recommended Datasets
1. **Primary**: Custom goal-persistence evaluation scenarios (multi-turn with distractions)
2. **Complementary**: IFEval for instruction adherence, BBH for task performance control
3. **Measurement**: TaskTracker methodology for activation-based drift detection

---

## 7. Gaps and Opportunities

1. **No direct study** of evolution for goal-persistence (vs task performance)
2. **No co-evolutionary goal-persistence**: AEGIS co-evolves attack/defense but for prompt injection, not general goal drift
3. **Limited multi-turn evaluation**: Most prompt evolution work is single-turn
4. **No activation-based fitness functions** in prompt evolution literature
5. **Cross-model transfer** of evolved goal-persistent prompts is untested

---

## 8. Key References

| Paper | arXiv | Year | Role in Research |
|-------|-------|------|-----------------|
| Goal Drift in LM Agents | 2505.02709 | 2025 | Drift definition & measurement |
| TaskTracker | 2406.00799 | 2024 | Activation-based drift detection |
| EvoPrompt | 2309.08532 | 2024 | Evolution framework to adapt |
| PromptBreeder | 2309.16797 | 2024 | Self-referential evolution theory |
| OPRO | 2309.03409 | 2023 | Alternative optimization approach |
| AEGIS | 2509.00088 | 2025 | Co-evolutionary defense blueprint |
| Instruction Adherence | 2510.18892 | 2024 | Adherence evaluation |
| GPS | 2210.17041 | 2022 | Early genetic prompt search |
| SPELL | 2310.01260 | 2023 | Semantic prompt evolution |
| SPRIG | 2410.14826 | 2024 | System prompt optimization |
| TARE | 2509.24130 | 2025 | Robustness-aware evolution |
| EVOREFUSE | 2505.23473 | 2025 | Over-refusal optimization |
| GAAPO | 2504.07157 | 2025 | GA-based prompt optimization |
| Tournament of Prompts | 2506.00178 | 2025 | Competitive prompt evolution |
| GPO | 2402.17564 | 2024 | Gradient-inspired optimization |
| Prompt Reinforcing | 2510.05921 | 2025 | Long-term goal tracking |

---

*Literature review prepared for: Evolved Desire in LLMs Research Project*
*Date: March 2026*
*Papers downloaded: 16 | Deep-read: 4 (TaskTracker, EvoPrompt, Goal Drift, AEGIS)*
