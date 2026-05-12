from prompt_eval.models import PromptItem
from prompt_eval.scoring import DEFAULT_WEIGHTS, ScoringEngine


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
