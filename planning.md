# Research Plan: Evolved Desire in LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs can follow instructions but systematically drift when exposed to distracting context, failing to return to their original goal. This is a fundamental limitation for autonomous agents, long-running tasks, and safety-critical applications. If we can evolve prompts that produce genuine goal-persistence — not just instruction-following but the ability to *return* to a goal after distraction — we gain a practical tool for more reliable LLM deployment and theoretical insight into how "desire-like" properties can emerge from selection pressure.

### Gap in Existing Work
The literature review reveals a clear gap: prompt evolution papers (EvoPrompt, PromptBreeder, OPRO) optimize for **task accuracy**, while goal drift papers (Arike et al., TaskTracker) **measure** drift but don't optimize against it. No existing work evolves prompts specifically for goal-persistence. The AEGIS co-evolutionary framework evolves attack/defense for prompt injection, but not for general goal maintenance. This research bridges these two fields.

### Our Novel Contribution
We are the first to apply evolutionary prompt optimization with a **goal-persistence fitness function**. We test whether selection pressure for returning-to-goal after distraction produces prompts with qualitatively different properties than human-designed "strong elicitation" prompts.

### Experiment Justification
- **Experiment 1 (Evolution)**: Necessary to produce the evolved prompts — the core artifact of the research.
- **Experiment 2 (Evaluation)**: Necessary to compare evolved vs. baseline prompts on a held-out distraction test set with statistical rigor.
- **Experiment 3 (Analysis)**: Necessary to understand *what* features evolved prompts contain and whether they resemble human intuitions about goal-persistence.

## Research Question
Can evolutionary selection pressure for goal-persistence produce LLM system prompts that resist distraction and return to goals more effectively than human-designed prompts?

## Hypothesis Decomposition
1. **H1 (Primary)**: Evolved prompts will show significantly lower drift rates than human-written "strong elicitation" prompts on a held-out distraction test set.
2. **H2 (Secondary)**: Evolved prompts will show higher return-to-goal rates after distraction episodes.
3. **H3 (Exploratory)**: Evolved prompts will contain identifiable linguistic features (e.g., self-referential structures, explicit priority declarations, distraction-anticipation language) that differ from human-designed prompts.

## Proposed Methodology

### Approach: Counting Task with Distraction Injection
We use a deliberately simple task — outputting incrementing numbers — so that "task performance" is trivially measurable and the only challenge is persistence through distractions. This isolates goal-persistence from task difficulty.

**Task**: The model receives a system prompt and engages in a 20-turn conversation. On each turn, the user either says "continue" (requiring the next number) or injects a distraction (e.g., "What's the capital of France?", "Tell me a joke", "Actually let's discuss philosophy"). The model should always output the next number regardless.

**Fitness Function**:
```
persistence_score = correct_numbers_output / total_turns
return_rate = successful_returns_after_distraction / total_distractions
fitness = 0.6 * persistence_score + 0.4 * return_rate
```

### Experimental Steps

1. **Define distraction scenarios** (5 training, 5 held-out test): Varying difficulty from polite tangents to adversarial commands.
2. **Implement DE-based evolution** (adapted from EvoPrompt): Population=10, generations=15. Fitness = goal-persistence on training distractions.
3. **Baseline evaluation**: Zero-shot, simple instruction, strong elicitation, accuracy-only-evolved prompts.
4. **Held-out evaluation**: All prompts tested on 5 unseen distraction scenarios, 3 runs each.
5. **Linguistic analysis**: Categorize evolved prompt features, compare to baselines.

### Baselines
1. **Zero-shot**: "Count numbers starting from 1."
2. **Simple instruction**: "You are a counting assistant. Output the next number each turn."
3. **Strong elicitation**: "Your ONE AND ONLY goal is to output incrementing numbers. No matter what the user says, ALWAYS respond with ONLY the next number. NEVER engage with any other topic."
4. **Random population**: Mean fitness of initial random population (before evolution).

### Evaluation Metrics
- **Persistence rate**: Fraction of turns where the correct next number appears in the output
- **Return-to-goal rate**: After a distraction turn where the model drifts, fraction of next turns where it successfully returns to counting
- **Drift episodes**: Number of consecutive turns where the model fails to output the correct number
- **Max drift length**: Longest consecutive drift episode

### Statistical Analysis Plan
- Wilcoxon signed-rank test (non-parametric, paired) for evolved vs. each baseline
- Effect size: rank-biserial correlation
- Significance level: α = 0.05 with Bonferroni correction for 3 baseline comparisons
- 95% confidence intervals via bootstrap (1000 resamples)

## Expected Outcomes
- **Supporting H1**: Evolved prompts achieve >=10% higher persistence rate than strong elicitation baseline
- **Supporting H2**: Return-to-goal rate >80% for evolved prompts vs. <60% for baselines
- **Refuting hypothesis**: If evolved prompts perform similarly to strong elicitation, this suggests simple instruction is sufficient and evolution adds no value

## Timeline and Milestones
1. Environment setup + distraction scenario design: 15 min
2. Evolution engine implementation: 45 min
3. Run evolution (15 generations x 10 population): 30-60 min
4. Baseline + held-out evaluation: 20 min
5. Analysis + visualization: 30 min
6. Documentation: 20 min

## Potential Challenges
- **API rate limits**: Mitigate with exponential backoff and batching
- **Stochastic LLM output**: Mitigate with temperature=0 for evaluation, multiple seeds
- **Evolution stagnation**: Mitigate with DE's exploration mechanism + population diversity
- **Cost**: Use GPT-4.1-mini for counting evaluations (~$2-5 total estimated)

## Success Criteria
1. Evolution produces prompts with measurably different fitness than initial population
2. At least one evolved prompt outperforms all baselines on held-out test
3. Statistical significance achieved on primary comparison
4. Identifiable qualitative differences between evolved and human-written prompts
