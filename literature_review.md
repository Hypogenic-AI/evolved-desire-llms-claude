# Literature Review: Evolved Desire in LLMs

## Research Hypothesis

> Prompts evolved under selection pressure for goal-persistence in LLMs will result in less drift from the original task compared to standard prompts, suggesting that persistent "desire" cannot be simply instructed but must be evolved.

---

## Executive Summary

This literature review examines two converging research areas: (1) goal drift and task persistence in LLM agents, and (2) evolutionary approaches to prompt optimization. The intersection of these fields provides the theoretical foundation for testing whether evolved prompts can produce more persistent goal-following behavior than manually designed prompts.

---

## 1. Goal Drift and Task Persistence

### 1.1 The Problem of Goal Drift

Goal drift occurs when an LLM agent deviates from its original instructions during task execution. This is particularly problematic in:
- Multi-turn interactions where context accumulates
- Agentic systems processing external data
- Scenarios vulnerable to prompt injection attacks

**Key Paper**: "Evaluating Goal Drift in Language Model Agents" (2505.02709, 2025)
- Proposes methods to detect and measure goal drift in autonomous agents
- Establishes metrics for quantifying deviation from original instructions
- Provides evaluation framework applicable to our hypothesis testing

### 1.2 Task Drift Detection

**Key Paper**: "Are You Still on Track!? Catching LLM Task Drift with Activations" (2406.00799, 2024)
- Introduces TaskTracker: activation-based task drift detection
- Dataset of 500K+ instances across multiple drift scenarios
- Achieves >0.99 ROC AUC for detecting prompt injections and jailbreaks
- Key insight: Internal activations reveal drift before output manifestation

**Implications for Hypothesis**: TaskTracker provides both the methodology (activation analysis) and benchmark data for measuring whether evolved prompts reduce detectable drift.

### 1.3 Instruction Adherence

**Key Paper**: "When Models Can't Follow: Testing Instruction Adherence Across 256 LLMs" (2510.18892, 2024)
- Large-scale benchmark of instruction-following across models
- Identifies systematic patterns in instruction adherence failures
- Provides evaluation framework for measuring adherence

**Key Finding**: Instruction adherence is not uniform across instruction types, suggesting that specific prompt formulations may influence persistence.

---

## 2. Prompt Evolution and Optimization

### 2.1 Evolutionary Algorithms for Prompts

**Key Paper**: "EvoPrompt: Connecting LLMs with Evolutionary Algorithms" (2309.08532, ICLR 2024)
- Applies genetic algorithms and differential evolution to prompt optimization
- Up to 25% improvement on Big-Bench Hard benchmarks
- Key insight: Evolutionary approaches discover prompt structures that manual design misses

**Methodology**:
1. Initialize population of prompts
2. Evaluate fitness on task performance
3. Apply crossover and mutation via LLM operations
4. Select fittest prompts for next generation

### 2.2 Self-Referential Prompt Evolution

**Key Paper**: "PromptBreeder: Self-Referential Self-Improvement via Prompt Evolution" (2309.16797, ICML 2024)
- Evolves both task-prompts AND mutation-prompts simultaneously
- Self-referential: the system improves its own improvement mechanism
- Demonstrates emergent prompt structures not anticipated by designers

**Key Insight for Hypothesis**: If evolved prompts develop unexpected structures that improve task performance, they may also develop structures that enhance goal persistence—a property not explicitly optimized for.

### 2.3 LLM-Based Optimization

**Key Paper**: "Large Language Models as Optimizers (OPRO)" (2309.03409)
- Uses LLMs to generate and evaluate prompt variations
- Up to 50% improvement on Big-Bench Hard tasks
- Up to 8% improvement on GSM8K

**Optimization Process**:
1. Natural language description of optimization goal
2. LLM generates candidate solutions
3. Evaluation on training set
4. Best solutions added to prompt for next iteration

### 2.4 Tournament-Based Evolution

**Key Paper**: "Tournament of Prompts: Evolving LLM Instructions Through Structured Debates" (2506.00178, 2025)
- Uses Elo rating systems for prompt selection
- Structured debates for comparative evaluation
- Recent work showing continued innovation in evolutionary approaches

### 2.5 Gradient-Inspired Optimization

**Key Paper**: "GPO: Gradient-inspired LLM-based Prompt Optimizer" (2402.17564, 2024)
- Analyzes prompt optimization through gradient-based lens
- Up to 56.8% improvement on Big-Bench Hard
- Provides theoretical framework connecting discrete prompt optimization to continuous optimization

---

## 3. Long-Term Goal Tracking

**Key Paper**: "Prompt Reinforcing for Long-term Planning of LLMs" (2510.05921, 2025)
- Addresses multi-turn interactions and long-term goal tracking
- Uses RL-inspired prompt optimization
- Directly relevant to goal persistence across extended interactions

**Relevance**: This paper bridges prompt optimization and goal persistence, providing methodology for evaluating prompts in multi-turn settings where drift is most likely.

---

## 4. Theoretical Framework for Hypothesis

### 4.1 Why Evolution Might Produce Persistence

The hypothesis rests on several theoretical grounds:

1. **Selection Pressure Argument**: If prompts are selected based on goal-persistence metrics, evolution will discover features that enhance persistence, even if those features are not explicitly specified.

2. **Emergent Properties Argument**: PromptBreeder demonstrates that evolved prompts contain structures not anticipated by designers. Goal-persistence may be such an emergent property.

3. **Robustness Argument**: Evolutionary processes tend to produce robust solutions. Prompts evolved under varied conditions may generalize better than manually designed prompts.

### 4.2 Why Simple Instruction May Be Insufficient

Several observations suggest that instructing persistence is inadequate:

1. **Prompt Injection Vulnerability**: Despite explicit instructions to follow only user commands, LLMs are susceptible to prompt injection (TaskTracker data).

2. **Instruction Following Limits**: Even state-of-the-art models achieve only ~80% on verifiable instruction following (IFEval results).

3. **Drift in Multi-Turn**: Goal drift increases with conversation length, suggesting instructions lose influence over time.

### 4.3 Testable Predictions

1. **Primary**: Evolved prompts will show lower drift metrics (TaskTracker) than baseline prompts with equivalent explicit goal statements.

2. **Secondary**: Evolved prompts will maintain higher instruction adherence (IFEval-style metrics) across multi-turn interactions.

3. **Exploratory**: Evolved prompt structures will contain identifiable features that correlate with persistence (interpretability analysis).

---

## 5. Methodological Considerations

### 5.1 Experimental Design

Based on the literature, a robust experimental design would:

1. **Evolution Phase**:
   - Use EvoPrompt or OPRO framework
   - Define fitness function including goal-persistence metrics
   - Evolve prompts over multiple generations

2. **Evaluation Phase**:
   - Compare evolved vs. baseline prompts
   - Use TaskTracker for drift detection
   - Apply IFEval-style verifiable instructions
   - Test across multiple LLMs (generalization)

3. **Analysis Phase**:
   - Quantify drift reduction
   - Analyze evolved prompt structures
   - Identify transferable insights

### 5.2 Baselines

- Manually designed prompts with explicit goal statements
- Prompts optimized for task performance only (no persistence fitness)
- Zero-shot prompts

### 5.3 Metrics

| Metric | Source | Purpose |
|--------|--------|---------|
| Task Drift Score | TaskTracker | Primary outcome |
| Instruction Adherence | IFEval | Secondary outcome |
| Task Performance | BBH | Control variable |
| Activation Similarity | TaskTracker | Mechanistic insight |

---

## 6. Gaps and Opportunities

### 6.1 Current Gaps

1. **No direct study** of evolution for goal-persistence (vs. task performance)
2. **Limited understanding** of prompt features that cause persistence
3. **Lack of standardized** goal-drift benchmarks for evolved prompts

### 6.2 Research Opportunities

1. **Novel fitness functions** incorporating persistence metrics
2. **Cross-model generalization** of evolved persistent prompts
3. **Interpretability analysis** of evolved prompt structures

---

## 7. Key References

### Core Papers

| Paper | arXiv | Key Contribution |
|-------|-------|------------------|
| Goal Drift in LM Agents | 2505.02709 | Drift measurement methods |
| TaskTracker | 2406.00799 | Activation-based drift detection |
| Instruction Adherence | 2510.18892 | Large-scale adherence evaluation |
| EvoPrompt | 2309.08532 | EA-based prompt evolution |
| PromptBreeder | 2309.16797 | Self-referential evolution |
| OPRO | 2309.03409 | LLM-based optimization |
| Tournament of Prompts | 2506.00178 | Competitive prompt evolution |
| GPO | 2402.17564 | Gradient-inspired optimization |
| Prompt Reinforcing | 2510.05921 | Long-term goal tracking |

### Supplementary Resources

- TaskTracker Dataset: https://github.com/microsoft/TaskTracker
- IFEval Benchmark: HuggingFace google/IFEval
- Big-Bench Hard: https://github.com/suzgunmirac/BIG-Bench-Hard

---

## 8. Conclusion

The literature provides strong support for investigating whether evolutionary approaches to prompt design can enhance goal-persistence in LLMs. Key findings suggest:

1. **Goal drift is measurable** using activation-based methods (TaskTracker)
2. **Prompt evolution discovers** non-obvious solutions (EvoPrompt, PromptBreeder)
3. **Long-term goal tracking** is an active research area with established methods

The hypothesis that evolved prompts will show greater goal-persistence than instructed prompts is both theoretically motivated and empirically testable using existing tools and datasets.

---

*Literature review prepared for: Evolved Desire in LLMs Research Project*
*Date: December 2024*
