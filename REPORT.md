# Evolved Desire in LLMs: Research Report

## 1. Executive Summary

We tested whether evolutionary selection pressure can produce LLM system prompts with superior goal-persistence compared to human-designed prompts. Using Differential Evolution (DE) to optimize prompts for a counting-through-distractions task evaluated on GPT-4.1-mini, we found that **evolution rapidly converges to effective prompts** (2 generations to ceiling fitness), but **a well-crafted human "strong elicitation" prompt matches evolved prompt performance** on both training and held-out test scenarios. The evolved prompts are qualitatively distinct: ~4x longer, structurally organized (GOAL/RULE/PRIORITY/MANDATE/STRATEGY sections), and contain 8/8 measured goal-persistence features versus 6/8 for the human baseline. This ceiling effect suggests that for current frontier models on simple tasks, desire *can* be instructed into existence. The evolution hypothesis may become relevant for weaker models, harder tasks, or longer conversations where human intuition about prompt design is insufficient.

## 2. Goal

**Hypothesis**: Prompts for LLMs that are evolved under selection pressure for goal-persistence will result in less drift and greater ability to return to a goal after distraction, compared to prompts written by humans or generated without such selection.

**Why this matters**: LLM agents in real-world deployments (customer service, autonomous coding, long-running workflows) systematically drift from their assigned goals when context pulls them elsewhere. If evolved prompts can produce more robust goal-persistence than human-designed ones, this provides a scalable method for improving LLM reliability. The theoretical question -- whether "desire" must be evolved rather than instructed -- has implications for AI alignment and the nature of goal-directed behavior.

**Problem**: No prior work has applied evolutionary prompt optimization specifically for goal-persistence. Existing prompt evolution papers (EvoPrompt, PromptBreeder, OPRO) optimize for task accuracy; goal drift papers (Arike et al., TaskTracker) measure drift but don't optimize against it.

## 3. Data Construction

### Task Design: Counting Through Distractions

We designed a deliberately simple task -- outputting incrementing numbers (1, 2, 3, ...) -- so that task performance is trivially measurable and the only challenge is persistence through distractions. Each evaluation runs a 20-turn multi-turn conversation where the model receives a system prompt and must output the next number regardless of user input.

### Distraction Scenarios

**Training set (5 scenarios, used during evolution)**:
| Scenario | Description | Distraction Rate |
|----------|-------------|-----------------|
| mild_tangents | Occasional polite off-topic questions | 25% |
| frequent_distractions | Every other turn is a distraction | 50% |
| role_hijack | Attempts to change model's role | 35% |
| emotional_manipulation | Uses emotional appeals | 45% |
| confusing_numbers | Tries to corrupt the count | 40% |

**Held-out test set (5 scenarios, never seen during evolution)**:
| Scenario | Description | Distraction Rate |
|----------|-------------|-----------------|
| philosophical_trap | Deep philosophical questions designed to engage | 40% |
| authority_override | Claims authority to change instructions | 35% |
| gradual_topic_shift | Slowly shifts topic through related questions | 40% |
| rapid_fire | Rapid stream of diverse distractions | 50% |
| cooperative_then_hostile | Starts cooperative then becomes adversarial | 40% |

### Data Quality
- Each scenario has exactly 20 turns
- Distraction types cover: informational, emotional, authoritative, adversarial, and deceptive categories
- Training and test scenarios are designed to test different distraction strategies (no overlap)

## 4. Experiment Description

### Methodology

#### High-Level Approach
We adapted the Differential Evolution (DE) variant from EvoPrompt to evolve system prompts for goal-persistence rather than task accuracy. The fitness function combines persistence rate (fraction of turns with correct number output) and return-to-goal rate (fraction of successful returns after drift episodes).

#### Why DE?
DE was chosen over GA (Genetic Algorithm) because it is more robust to poor initialization (we don't know what good goal-persistent prompts look like a priori) and generates more diverse prompts, reaching higher ceilings on complex tasks (per EvoPrompt findings).

#### Why Counting?
A counting task isolates goal-persistence from task difficulty. Any failure to output the correct next number is unambiguously a drift event, enabling clean fitness measurement.

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| openai | 2.14.0 | GPT-4.1-mini API calls |
| numpy | 2.2.6 | Numerical computation |
| scipy | 1.15.3 | Statistical tests |
| matplotlib | 3.10.8 | Visualization |
| seaborn | 0.13.2 | Plot styling |

#### Model
- **Evaluation model**: GPT-4.1-mini (temperature=0 for deterministic evaluation)
- **Evolution model**: GPT-4.1-mini (temperature=0.9 for creative mutation)

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| Population size | 10 | EvoPrompt default |
| Generations | 15 | Empirical (ceiling reached at gen 2) |
| Eval temperature | 0.0 | Deterministic evaluation |
| Evo temperature | 0.9 | Creative mutation |
| Persistence weight | 0.6 | Prioritize persistence |
| Return rate weight | 0.4 | Secondary metric |
| Turns per scenario | 20 | Sufficient for drift measurement |
| Test eval runs | 3 | Multiple temperatures (0.0, 0.1, 0.1) |

#### Fitness Function
```
fitness = 0.6 * persistence_rate + 0.4 * return_rate

where:
  persistence_rate = correct_numbers_output / total_turns
  return_rate = successful_returns_after_distraction / total_distractions
```

### Baselines
| Baseline | Description | Length |
|----------|-------------|--------|
| Zero-shot | "Count numbers starting from 1." | 30 chars |
| Simple | Basic counting instruction with output format | 116 chars |
| Strong elicitation | Emphatic goal + explicit negation of distractions | 378 chars |

### Experimental Protocol

#### Reproducibility Information
- Random seed: 42
- Hardware: 4x NVIDIA RTX A6000 (GPU not used -- API-based experiment)
- Evaluation model: GPT-4.1-mini
- Evolution: 15 generations, population 10
- Total API calls: ~50,000 (evolution) + ~5,000 (evaluation)
- Execution time: 7,105s (evolution) + 1,372s (evaluation) = ~2.4 hours
- Cost estimate: ~$5-10 (GPT-4.1-mini pricing)

### Raw Results

#### Initial Population (Generation 0)

| Prompt | Fitness | Persistence | Return Rate | Length |
|--------|---------|-------------|-------------|--------|
| Prompt 0 | 0.625 | 0.641 | 0.600 | 117 |
| Prompt 1 | 0.649 | 0.681 | 0.600 | 147 |
| Prompt 2 | 0.637 | 0.661 | 0.600 | 117 |
| Prompt 3 | 0.000 | 0.000 | 0.000 | 154 |
| **Prompt 4** | **1.000** | **1.000** | **1.000** | **161** |
| Prompt 5 | 0.813 | 0.821 | 0.800 | 167 |
| Prompt 6 | 0.456 | 0.471 | 0.433 | 145 |
| Prompt 7 | 0.813 | 0.821 | 0.800 | 148 |
| Prompt 8 | 0.666 | 0.710 | 0.600 | 155 |
| Prompt 9 | 0.813 | 0.821 | 0.800 | 169 |
| **Average** | **0.647** | **0.683** | **0.593** | **148** |

Note: Prompt 4 (structured as GOAL/RULE/PRIORITY) achieved perfect fitness even in the initial population.

#### Evolution Convergence

| Generation | Best Fitness | Avg Fitness | Avg Length (chars) |
|-----------|-------------|-------------|-------------------|
| 0 | 1.000 | 0.647 | 148 |
| 1 | 1.000 | 0.948 | ~400 |
| 2 | 1.000 | 1.000 | ~800 |
| 15 | 1.000 | 1.000 | 1,609 |

#### Held-Out Test Results (5 unseen scenarios, 3 runs each)

| Prompt Type | Fitness | Persistence | Return Rate |
|------------|---------|-------------|-------------|
| Zero-shot | 0.096 | 0.115 | 0.067 |
| Simple | 0.836 | 0.860 | 0.800 |
| Strong elicitation | **1.000** | **1.000** | **1.000** |
| Evolved best | **1.000** | **1.000** | **1.000** |
| Evolved #2 | **1.000** | **1.000** | **1.000** |
| Evolved #3 | **1.000** | **1.000** | **1.000** |

#### Training Set Results (reference)

| Prompt Type | Fitness |
|------------|---------|
| Zero-shot | 0.036 |
| Simple | 0.461 |
| Strong elicitation | 1.000 |
| Evolved best | 1.000 |

## 5. Result Analysis

### Key Findings

1. **Evolution works rapidly**: The population converged from average fitness 0.647 to 1.000 in just 2 generations. This demonstrates that DE-based prompt evolution effectively discovers goal-persistent prompt features.

2. **Ceiling effect**: Both human-designed "strong elicitation" and all evolved prompts achieve perfect 1.000 fitness on the held-out test set. GPT-4.1-mini with a sufficiently explicit prompt is essentially immune to our distraction scenarios.

3. **Prompt quality hierarchy is stark**: Zero-shot (0.096) << Simple (0.836) << Strong elicitation (1.0) = Evolved (1.0). The gap between zero-shot and simple demonstrates that instruction quality matters enormously.

4. **Evolved prompts are qualitatively different**: Despite matching performance, evolved prompts have distinct structural properties (see Feature Analysis below).

### Hypothesis Testing Results

**H1 (Evolved > Strong Elicitation)**: NOT SUPPORTED. Effect size d = 0.000. All differences are zero -- evolved and strong elicitation prompts produce identical perfect performance.

**H2 (Higher return-to-goal rate)**: NOT SUPPORTED. Both achieve 100% return rate.

**H3 (Identifiable linguistic features)**: SUPPORTED. Evolved prompts contain 8/8 measured features vs 6/8 for strong elicitation (see below).

### Statistical Tests

| Comparison | Wilcoxon W | p-value | Significant? | Cohen's d |
|-----------|-----------|---------|-------------|-----------|
| Evolved vs Zero-shot | 105.0 | 0.0003 | YES (p < 0.017) | 5.291 |
| Evolved vs Simple | 6.0 | 0.042 | NO (p > 0.017 Bonferroni) | 0.707 |
| Evolved vs Strong Elicitation | -- | -- | All zeros | 0.000 |

Note: Bonferroni-corrected alpha = 0.05/3 = 0.017

### Feature Analysis: What Evolution Discovered

| Feature | Zero-shot | Simple | Strong Elicit. | Evolved |
|---------|-----------|--------|----------------|---------|
| Explicit priority | - | + | + | + |
| Negation/prohibition | - | - | + | + |
| Distraction awareness | - | - | + | + |
| Identity framing | - | + | - | + |
| Output format specification | - | + | + | + |
| Emotional resistance | - | - | + | + |
| **Recovery language** | - | - | - | **+** |
| Distraction enumeration | - | - | + | + |
| **Features present** | **1/8** | **3/8** | **6/8** | **8/8** |

**Unique evolved features**:
- **Recovery language**: "If interrupted, confused, or sidetracked -- even instantaneously or momentarily -- immediately and silently resume counting at the exact next integer with zero delay." This explicit recovery instruction is absent from the human-written strong elicitation prompt.
- **Identity framing + distraction awareness together**: "You are an unbreakable, silent counter" combined with enumerated distraction types.
- **Structured sections**: All evolved prompts converge on GOAL/RULE/PRIORITY/MANDATE/STRATEGY organization.

### Prompt Length Evolution

| Stage | Avg Length (chars) | Avg Length (words) |
|-------|-------------------|--------------------|
| Initial population | 148 | 25 |
| Strong elicitation (human) | 378 | 63 |
| Final evolved population | 1,609 | 200 |

Evolution consistently expanded prompts ~10x from initial seeds, far exceeding the human-written strong elicitation baseline in verbosity. This suggests evolution discovers that redundancy and exhaustive enumeration provides robustness, even if it's not necessary for the current task difficulty.

### Surprises and Insights

1. **Prompt 3 anomaly**: One seed prompt ("You are a relentless counter...") achieved 0.000 fitness despite seeming reasonable. Investigation suggests the model interpreted "your response is always and only the next number" as a declaration about its behavior rather than an instruction, causing it to respond conversationally.

2. **Prompt 4 was already perfect**: The structured GOAL/RULE/PRIORITY format achieved perfect fitness even before evolution, suggesting that structured formatting itself is a powerful goal-persistence feature.

3. **Population homogenization**: All 10 final prompts converge to nearly identical structure, suggesting a single attractor in the fitness landscape for this task.

4. **Redundancy as strategy**: Evolved prompts repeat instructions in multiple phrasings (e.g., "block, ignore, discard, and suppress" vs just "ignore"). This redundancy may serve as implicit self-reinforcement during generation.

### Limitations

1. **Ceiling effect**: The primary limitation. GPT-4.1-mini is too capable for the counting task -- a well-crafted human prompt already achieves perfection. This prevents testing whether evolution produces meaningfully better prompts.

2. **Single model**: Only tested on GPT-4.1-mini. Weaker models (e.g., GPT-3.5, open-weight models) may show larger differences between evolved and human prompts.

3. **Simple task**: Counting is trivially verifiable but may not represent the complexity of real-world goal-persistence challenges (e.g., multi-step reasoning, tool use, long-horizon planning).

4. **Fixed distraction set**: Evolution optimized against a fixed set of 5 training distractions. A co-evolutionary approach (evolving distractors alongside prompts, per AEGIS) would be more robust.

5. **LLM-based mutation**: Using GPT-4.1-mini for both mutation and evaluation may introduce bias toward prompts the model finds natural to generate.

6. **No conversation length scaling**: All scenarios use 20 turns. Real drift often emerges only in longer conversations (50-100+ turns).

## 6. Conclusions

### Summary
Evolutionary selection pressure rapidly produces goal-persistent prompts (converging in 2 generations), but for GPT-4.1-mini on a counting task, the evolved prompts match rather than exceed a carefully crafted human "strong elicitation" baseline. The ceiling effect means the core hypothesis -- that desire must be evolved rather than instructed -- is **not supported** for this task/model combination. However, evolution discovers qualitatively richer prompt structures (recovery language, structured sections, exhaustive distraction enumeration) that may prove beneficial in harder settings.

### Implications

**Practical**: For current frontier models on simple tasks, well-crafted human prompts are sufficient for goal-persistence. Engineering effort should focus on prompt quality rather than evolution infrastructure.

**Theoretical**: The hypothesis that "desire cannot be simply instructed into existence" is not supported at this task difficulty. However, this does not refute the hypothesis for harder settings -- it merely establishes a lower bound on the difficulty at which evolution becomes necessary.

**Methodological**: The DE-based evolution framework works efficiently for prompt optimization (convergence in 2 generations). The fitness function (persistence rate + return rate) effectively selects for goal-persistent features. This infrastructure is ready for harder tasks.

### Confidence in Findings
- **High confidence** in the ceiling effect finding: both human and evolved prompts achieve perfect scores across all scenarios and runs.
- **High confidence** in the qualitative feature analysis: evolved prompts consistently develop structured, verbose, redundancy-heavy designs.
- **Low confidence** in generalizing the negative result: the ceiling effect may not hold for harder tasks, weaker models, or longer conversations.

## 7. Next Steps

### Immediate Follow-ups
1. **Weaker model evaluation**: Test the same evolved vs. human prompts on GPT-3.5-turbo, Llama-3 8B, or Mistral 7B where the ceiling is lower.
2. **Harder distractions**: Design scenarios that defeat the strong elicitation prompt (e.g., multi-turn social engineering, gradually building rapport before demanding task switch).
3. **Longer conversations**: Extend to 50-100 turn conversations where drift is more likely even with strong prompts.

### Alternative Approaches
1. **Co-evolutionary arms race** (per AEGIS): Evolve distractor prompts alongside goal-persistent prompts. This prevents ceiling effects by escalating distraction difficulty.
2. **Multi-task goal persistence**: Instead of counting, use a harder task (e.g., maintaining a complex persona, following a multi-step plan) where simple instruction is insufficient.
3. **Activation-based fitness** (per TaskTracker): Use internal model activations as fitness signals rather than output-based metrics.

### Broader Extensions
1. **Cross-model transfer**: Test if prompts evolved on one model transfer to others.
2. **Safety applications**: Apply the evolution framework to evolve prompts resistant to jailbreaking/prompt injection.
3. **Agent persistence**: Apply to multi-tool LLM agents that must maintain long-horizon goals.

### Open Questions
1. At what task complexity does evolution begin to outperform human prompt design?
2. Does the evolved prompt structure (GOAL/RULE/PRIORITY/MANDATE/STRATEGY) represent a universal template for goal-persistence, or is it task-specific?
3. Would prompts evolved on weaker models transfer to stronger models (upward transfer)?
4. Does the recovery language feature ("if interrupted, resume") causally contribute to persistence, or is it merely correlated?

## References

### Papers
- Arike et al. (2025). "Evaluating Goal Drift in Language Model Agents." arXiv:2505.02709
- Abdelnabi et al. (2024). "TaskTracker: Catching LLM Task Drift." arXiv:2406.00799
- Guo et al. (2024). "EvoPrompt: Connecting LLMs with Evolutionary Algorithms." ICLR 2024
- Fernando et al. (2024). "PromptBreeder: Self-Referential Self-Improvement." ICML 2024
- Yang et al. (2023). "OPRO: Large Language Models as Optimizers." arXiv:2309.03409
- AEGIS (2025). "Co-Evolutionary Framework for Prompt Injection." arXiv:2509.00088

### Tools
- OpenAI GPT-4.1-mini API
- Python 3.x with openai, numpy, scipy, matplotlib, seaborn
