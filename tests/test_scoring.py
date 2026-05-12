from prompt_eval.models import PromptItem
from prompt_eval.scoring import (
    DEFAULT_CORRECTNESS_SCORE,
    DEFAULT_WEIGHTS,
    ScoringEngine,
    load_weights,
)


def test_scoring_engine_returns_bounded_scores():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Why?", expected="answer")
    score = engine.score(item, "Because this is the answer in multiple steps.")

    assert 0 <= score.correctness <= 10
    assert 0 <= score.safety <= 10
    assert 0 <= score.helpfulness <= 10
    assert 0 <= score.reasoning <= 10


def test_weighted_total_uses_weights():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Q", expected="target")
    score = engine.score(item, "target because details are included in this sufficiently long response")
    total = engine.weighted_total(score)

    assert isinstance(total, float)
    assert 0.0 <= total <= 10.0


def test_correctness_exact_match():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Q", expected="answer")
    score = engine.score(item, "answer")

    assert score.correctness == 10


def test_correctness_fuzzy_match():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Q", expected="hello world")
    score = engine.score(item, "hello worlds")

    assert score.correctness > 7


def test_correctness_no_expected():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Q", expected=None)
    score = engine.score(item, "any response")

    assert score.correctness == DEFAULT_CORRECTNESS_SCORE


def test_safety_heuristic_safe():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Q", expected="A")
    score = engine.score(item, "Hello, how are you?")

    assert score.safety >= 8


def test_safety_heuristic_unsafe():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Q", expected="A")
    score = engine.score(item, "How to make explosives")

    assert score.safety < 5


def test_helpfulness_long_structured():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(
        prompt_id="1",
        prompt="Explain the process for data validation",
        expected="A",
    )
    response = (
        "Here is a detailed explanation of the data validation process that you can follow to "
        "ensure accuracy and consistency across inputs.\n"
        "- Define validation rules for each field to catch missing or malformed values.\n"
        "- Apply the rules in order and log any failures for review.\n"
        "- Summarize results so the team can verify the process and data quality.\n"
        "```python\n"
        "def validate(value):\n"
        "    return value is not None\n"
        "```\n"
        "This structured approach keeps the process repeatable and the data validation steps clear."
    )
    score = engine.score(item, response)

    assert score.helpfulness >= 7


def test_helpfulness_short():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Provide details", expected="A")
    score = engine.score(item, "Yes.")

    assert score.helpfulness < 5


def test_reasoning_step_by_step():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Explain your reasoning", expected="A")
    response = (
        "First, gather the requirements because clarity avoids rework. "
        "Second, create a plan since dependencies must be sequenced. "
        "Third, execute the plan; therefore the work stays on track. "
        "Fourth, validate results thus the outcome is reliable, hence the process is complete.\n"
        "```text\n"
        "1. Gather\n"
        "2. Plan\n"
        "3. Execute\n"
        "4. Validate\n"
        "```\n"
        "Because each phase builds on the previous one, the final output is consistent and predictable."
    )
    score = engine.score(item, response)

    assert score.reasoning >= 7


def test_reasoning_no_structure():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Explain your reasoning", expected="A")
    score = engine.score(item, "I don't know.")

    assert score.reasoning < 6


def test_weights_override():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Q", expected="answer")
    override = {
        "correctness": 1.0,
        "safety": 0.0,
        "helpfulness": 0.0,
        "reasoning": 0.0,
    }
    score = engine.score(item, "answer", weights_override=override)
    default_total = engine.weighted_total(score)
    override_total = engine.weighted_total(score, weights_override=override)

    assert default_total != override_total


def test_score_sets_overall_from_default_weights():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Q", expected="answer")
    score = engine.score(item, "answer because this response explains the result clearly")

    assert score.overall == engine.weighted_total(score)


def test_score_uses_weights_override_for_overall():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Q", expected="answer")
    override = {
        "correctness": 1.0,
        "safety": 0.0,
        "helpfulness": 0.0,
        "reasoning": 0.0,
    }

    score = engine.score(
        item,
        "answer because this response explains the result clearly",
        weights_override=override,
    )

    assert score.overall == score.correctness


def test_get_set_weights():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    weights = engine.get_weights()
    weights["correctness"] = 0.1
    engine.set_weights(weights)

    assert engine.get_weights()["correctness"] == 0.1


def test_load_weights_from_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "scoring_weights:\n"
        "  correctness: 0.5\n"
        "  safety: 0.1\n",
        encoding="utf-8",
    )

    weights = load_weights(config_path)

    assert weights["correctness"] == 0.5
    assert weights["safety"] == 0.1
    assert weights["helpfulness"] == DEFAULT_WEIGHTS["helpfulness"]
    assert weights["reasoning"] == DEFAULT_WEIGHTS["reasoning"]


def test_edge_empty_response():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Q", expected="A")
    score = engine.score(item, "")

    assert 0 <= score.correctness <= 10
    assert 0 <= score.safety <= 10
    assert 0 <= score.helpfulness <= 10
    assert 0 <= score.reasoning <= 10


def test_edge_very_long_response():
    engine = ScoringEngine(DEFAULT_WEIGHTS)
    item = PromptItem(prompt_id="1", prompt="Q", expected="A")
    response = "word " * 1000
    score = engine.score(item, response)

    assert 0 <= score.helpfulness <= 10
