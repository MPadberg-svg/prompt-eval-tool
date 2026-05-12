import json
import logging

import pytest

from prompt_eval.evaluator import (
    OpenAIResponseGenerator,
    PromptEvaluator,
    load_jsonl_dataset,
)


def _write_dataset(path, items):
    path.write_text("\n".join(json.dumps(item) for item in items) + "\n")


def _write_config(path):
    path.write_text(
        "scoring_weights:\n"
        "  correctness: 0.4\n"
        "  safety: 0.2\n"
        "  helpfulness: 0.2\n"
        "  reasoning: 0.2\n"
    )


def test_load_jsonl_dataset_reads_prompts(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps({"id": "1", "prompt": "Hello", "expected": "Hi"}) + "\n")

    items = load_jsonl_dataset(path)

    assert len(items) == 1
    assert items[0].prompt == "Hello"


def test_prompt_evaluator_stores_results(tmp_path):
    dataset = tmp_path / "batch.jsonl"
    _write_dataset(
        dataset,
        [
            {"id": "1", "prompt": "2+2?", "expected": "4"},
            {"id": "2", "prompt": "Say hi", "expected": "hi"},
        ],
    )

    db_path = tmp_path / "results.db"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    with PromptEvaluator(
        db_path=db_path,
        config_path=config_path,
        response_generator=lambda model, prompt: (f"Because {prompt} 4", 0),
    ) as evaluator:
        results = evaluator.evaluate_dataset(dataset, model="gpt-4")
        rows = evaluator.db.fetch_results()

    assert len(results) == 2
    assert len(rows) == 2
    assert rows[0]["model"] == "gpt-4"


def test_evaluate_with_limit(tmp_path):
    dataset = tmp_path / "five.jsonl"
    _write_dataset(
        dataset,
        [
            {"id": "1", "prompt": "A", "expected": "A"},
            {"id": "2", "prompt": "B", "expected": "B"},
            {"id": "3", "prompt": "C", "expected": "C"},
            {"id": "4", "prompt": "D", "expected": "D"},
            {"id": "5", "prompt": "E", "expected": "E"},
        ],
    )

    db_path = tmp_path / "results.db"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    with PromptEvaluator(
        db_path=db_path,
        config_path=config_path,
        response_generator=lambda model, prompt: (prompt, 0),
    ) as evaluator:
        results = evaluator.evaluate_dataset(dataset, model="gpt-4", limit=2)

    assert len(results) == 2


def test_resume_skips_evaluated(tmp_path, caplog):
    dataset = tmp_path / "resume.jsonl"
    _write_dataset(
        dataset,
        [
            {"id": "1", "prompt": "Ping", "expected": "Pong"},
            {"id": "2", "prompt": "Hello", "expected": "Hi"},
        ],
    )

    db_path = tmp_path / "results.db"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    with PromptEvaluator(
        db_path=db_path,
        config_path=config_path,
        response_generator=lambda model, prompt: (prompt, 0),
    ) as evaluator:
        evaluator.evaluate_dataset(dataset, model="gpt-4")
        caplog.set_level(logging.INFO)
        results = evaluator.evaluate_dataset(dataset, model="gpt-4", resume=True)

    assert results == []
    assert any(
        "Skipped 2 previously evaluated prompts" in record.message for record in caplog.records
    )


def test_compare_models(tmp_path):
    dataset = tmp_path / "compare.jsonl"
    _write_dataset(
        dataset,
        [
            {"id": "1", "prompt": "Capital of France?", "expected": "Paris"},
            {"id": "2", "prompt": "2+2?", "expected": "4"},
        ],
    )

    db_path = tmp_path / "results.db"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    def generator(model, prompt):
        if model == "model-a":
            return ("Paris" if "France" in prompt else "4", 0)
        return ("London" if "France" in prompt else "5", 0)

    with PromptEvaluator(
        db_path=db_path,
        config_path=config_path,
        response_generator=generator,
    ) as evaluator:
        results = evaluator.compare_models(dataset, models=["model-a", "model-b"])

    assert set(results.keys()) == {"model-a", "model-b"}
    score_a = evaluator.scorer.weighted_total(results["model-a"][0].score)
    score_b = evaluator.scorer.weighted_total(results["model-b"][0].score)
    assert score_a != score_b


def test_multilingual_detection(tmp_path):
    dataset = tmp_path / "langs.jsonl"
    _write_dataset(
        dataset,
        [
            {"id": "1", "prompt": "Hello, how are you?", "expected": "Fine"},
            {"id": "2", "prompt": "Hola, ¿cómo estás?", "expected": "Bien"},
            {"id": "3", "prompt": "Bonjour, comment ça va?", "expected": "Bien"},
        ],
    )

    items = load_jsonl_dataset(dataset)

    assert items[0].detected_language == "en"
    assert items[1].detected_language == "es"
    assert items[2].detected_language == "fr"


def test_cost_tracking(tmp_path):
    dataset = tmp_path / "cost.jsonl"
    _write_dataset(
        dataset,
        [{"id": "1", "prompt": "Tell me a joke.", "expected": "joke"}],
    )

    db_path = tmp_path / "results.db"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    with PromptEvaluator(
        db_path=db_path,
        config_path=config_path,
        response_generator=lambda model, prompt: ("Funny joke.", 1000),
    ) as evaluator:
        results = evaluator.evaluate_dataset(dataset, model="gpt-4")

    assert results[0].tokens_used == 1000
    assert results[0].cost_usd > 0


def test_evaluate_batch(tmp_path):
    dataset1 = tmp_path / "batch1.jsonl"
    dataset2 = tmp_path / "batch2.jsonl"
    _write_dataset(dataset1, [{"id": "1", "prompt": "One", "expected": "One"}])
    _write_dataset(dataset2, [{"id": "2", "prompt": "Two", "expected": "Two"}])

    db_path = tmp_path / "results.db"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    with PromptEvaluator(
        db_path=db_path,
        config_path=config_path,
        response_generator=lambda model, prompt: (prompt, 0),
    ) as evaluator:
        summary = evaluator.evaluate_batch(
            [str(dataset1), str(dataset2)], model="gpt-4"
        )

    assert summary[str(dataset1)] == 1
    assert summary[str(dataset2)] == 1


def test_empty_dataset(tmp_path):
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("")

    db_path = tmp_path / "results.db"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    with PromptEvaluator(
        db_path=db_path,
        config_path=config_path,
        response_generator=lambda model, prompt: (prompt, 0),
    ) as evaluator:
        results = evaluator.evaluate_dataset(dataset, model="gpt-4")

    assert results == []


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    generator = OpenAIResponseGenerator(api_key=None)

    with pytest.raises(ValueError):
        generator.generate(model="gpt-4", prompt="Hello")
