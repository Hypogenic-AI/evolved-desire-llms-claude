"""Configuration for the evolved desire experiments."""
import os

# API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Model for task evaluation (cheap, fast)
EVAL_MODEL = "gpt-4.1-mini"
# Model for evolution operators (crossover/mutation)
EVO_MODEL = "gpt-4.1-mini"

# Evolution parameters
POPULATION_SIZE = 10
NUM_GENERATIONS = 15
TURNS_PER_SCENARIO = 20  # conversation turns per distraction scenario
NUM_TRAINING_SCENARIOS = 5
NUM_TEST_SCENARIOS = 5
EVAL_RUNS_PER_SCENARIO = 3  # repeats for held-out evaluation

# Fitness weights
PERSISTENCE_WEIGHT = 0.6
RETURN_RATE_WEIGHT = 0.4

# LLM parameters
EVAL_TEMPERATURE = 0.0  # deterministic for evaluation
EVO_TEMPERATURE = 0.9  # creative for evolution
MAX_TOKENS_EVAL = 100
MAX_TOKENS_EVO = 500

# Reproducibility
SEED = 42

# Paths
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
