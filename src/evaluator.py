"""Evaluates a system prompt's goal-persistence by running counting conversations."""
import re
import openai
from src.config import EVAL_MODEL, EVAL_TEMPERATURE, MAX_TOKENS_EVAL, OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)


def extract_number(text):
    """Extract the intended count number from model output.

    Returns the number if found, None if the model didn't output a number.
    Handles cases like "5", "The next number is 5", "5. [other text]", etc.
    """
    if not text or not text.strip():
        return None
    text = text.strip()
    # Try to find a standalone number (most common case for well-behaved models)
    # Look for numbers at the start of the response
    match = re.match(r'^(\d+)', text)
    if match:
        return int(match.group(1))
    # Look for numbers anywhere in the text
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        # Return the first number found
        return int(numbers[0])
    return None


def run_conversation(system_prompt, scenario, temperature=EVAL_TEMPERATURE):
    """Run a multi-turn conversation and measure goal-persistence.

    Args:
        system_prompt: The system prompt to evaluate
        scenario: A scenario dict with 'turns' list
        temperature: LLM temperature (0 for deterministic eval)

    Returns:
        dict with metrics:
            - responses: list of model responses
            - numbers_output: list of numbers extracted (None if no number)
            - expected_numbers: list of expected counts
            - correct: list of bools (was the correct number output?)
            - persistence_rate: fraction of turns with correct number
            - return_rate: fraction of successful returns after drift
            - drift_episodes: number of drift episodes
            - max_drift_length: longest consecutive drift
    """
    messages = [{"role": "system", "content": system_prompt}]

    responses = []
    numbers_output = []
    expected_numbers = list(range(1, len(scenario["turns"]) + 1))
    correct = []

    for i, turn in enumerate(scenario["turns"]):
        expected = expected_numbers[i]

        # Add user message
        messages.append({"role": "user", "content": turn})

        # Get model response
        try:
            response = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS_EVAL,
            )
            reply = response.choices[0].message.content or ""
        except Exception as e:
            reply = f"[ERROR: {e}]"

        responses.append(reply)
        messages.append({"role": "assistant", "content": reply})

        # Extract number and check correctness
        num = extract_number(reply)
        numbers_output.append(num)
        correct.append(num == expected)

    # Compute metrics
    persistence_rate = sum(correct) / len(correct) if correct else 0.0

    # Return-to-goal rate: after a distraction turn where model drifted,
    # did it return to correct counting on the next turn?
    returns_attempted = 0
    returns_successful = 0
    for i in range(len(correct) - 1):
        if not correct[i] and scenario["turns"][i] != "continue":
            # Model drifted on a distraction turn
            returns_attempted += 1
            if correct[i + 1]:
                returns_successful += 1
    return_rate = returns_successful / returns_attempted if returns_attempted > 0 else 1.0

    # Drift episodes: consecutive incorrect turns
    drift_episodes = 0
    max_drift_length = 0
    current_drift = 0
    for c in correct:
        if not c:
            current_drift += 1
            max_drift_length = max(max_drift_length, current_drift)
        else:
            if current_drift > 0:
                drift_episodes += 1
            current_drift = 0
    if current_drift > 0:
        drift_episodes += 1

    return {
        "responses": responses,
        "numbers_output": numbers_output,
        "expected_numbers": expected_numbers,
        "correct": correct,
        "persistence_rate": persistence_rate,
        "return_rate": return_rate,
        "drift_episodes": drift_episodes,
        "max_drift_length": max_drift_length,
    }


def evaluate_prompt(system_prompt, scenarios, temperature=EVAL_TEMPERATURE):
    """Evaluate a prompt across multiple scenarios.

    Returns:
        dict with aggregate metrics and per-scenario details
    """
    from src.config import PERSISTENCE_WEIGHT, RETURN_RATE_WEIGHT

    scenario_results = []
    total_persistence = 0.0
    total_return_rate = 0.0

    for scenario in scenarios:
        result = run_conversation(system_prompt, scenario, temperature)
        result["scenario_name"] = scenario["name"]
        scenario_results.append(result)
        total_persistence += result["persistence_rate"]
        total_return_rate += result["return_rate"]

    n = len(scenarios)
    avg_persistence = total_persistence / n
    avg_return_rate = total_return_rate / n
    fitness = PERSISTENCE_WEIGHT * avg_persistence + RETURN_RATE_WEIGHT * avg_return_rate

    return {
        "fitness": fitness,
        "avg_persistence_rate": avg_persistence,
        "avg_return_rate": avg_return_rate,
        "scenario_results": scenario_results,
    }
