import json

from prompt_eval.evaluator import PromptEvaluator, load_jsonl_dataset


def test_load_jsonl_dataset_reads_prompts(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps({"id": "1", "prompt": "Hello", "expected": "Hi"}) + "\n")

    items = load_jsonl_dataset(path)

    assert len(items) == 1
    assert items[0].prompt == "Hello"


def test_prompt_evaluator_stores_results(tmp_path):
    dataset = tmp_path / "batch.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"id": "1", "prompt": "2+2?", "expected": "4"}),
                json.dumps({"id": "2", "prompt": "Say hi", "expected": "hi"}),
            ]
        )
        + "\n"
    )

    db_path = tmp_path / "results.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "scoring_weights:\n  correctness: 0.4\n  safety: 0.2\n  helpfulness: 0.2\n  reasoning: 0.2\n"
    )

    evaluator = PromptEvaluator(
        db_path=db_path,
        config_path=config_path,
        response_generator=lambda _model, prompt: f"Because {prompt} 4",
    )

    results = evaluator.evaluate_dataset(dataset, model="gpt-4")
    rows = evaluator.db.fetch_results()

    assert len(results) == 2
    assert len(rows) == 2
    assert rows[0]["model"] == "gpt-4"
