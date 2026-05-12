from prompt_eval.database import DatabaseManager
from prompt_eval.models import EvaluationResult, PromptItem, Score


def test_insert_and_fetch(tmp_path):
    db_path = tmp_path / "results.db"
    with DatabaseManager(db_path) as db:
        result = EvaluationResult(
            dataset="sample",
            prompt_item=PromptItem(
                prompt_id="1",
                prompt="Hello",
                expected="Hi",
                detected_language="en",
            ),
            model="gpt-4",
            response="Hi there",
            score=Score(correctness=8, safety=10, helpfulness=8, reasoning=6),
            tokens_used=42,
            cost_usd=0.0125,
        )
        result.detected_language = "en"
        db.insert_result(result, total_score=8.5)

        rows = db.fetch_results()

    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "sample"
    assert row["prompt_id"] == "1"
    assert row["prompt"] == "Hello"
    assert row["expected"] == "Hi"
    assert row["model"] == "gpt-4"
    assert row["response"] == "Hi there"
    assert row["correctness"] == 8
    assert row["safety"] == 10
    assert row["helpfulness"] == 8
    assert row["reasoning"] == 6
    assert row["total_score"] == 8.5
    assert row["tokens_used"] == 42
    assert row["cost_usd"] == 0.0125
    assert row["detected_language"] == "en"


def test_get_evaluated_prompt_ids(tmp_path):
    db_path = tmp_path / "results.db"
    with DatabaseManager(db_path) as db:
        for prompt_id, model in [("1", "gpt-4"), ("2", "gpt-4o"), ("3", "claude")]:
            result = EvaluationResult(
                dataset="sample",
                prompt_item=PromptItem(prompt_id=prompt_id, prompt="Hello"),
                model=model,
                response="Hi there",
                score=Score(correctness=7, safety=8, helpfulness=7, reasoning=6),
            )
            db.insert_result(result, total_score=7.0)

        assert db.get_evaluated_prompt_ids("sample", "gpt-4") == {("1", "gpt-4")}
        assert db.get_evaluated_prompt_ids("sample", "gpt-4o") == {("2", "gpt-4o")}
        assert db.get_evaluated_prompt_ids("sample", "claude") == {("3", "claude")}


def test_fetch_by_dataset(tmp_path):
    db_path = tmp_path / "results.db"
    with DatabaseManager(db_path) as db:
        for dataset in ["alpha", "beta"]:
            result = EvaluationResult(
                dataset=dataset,
                prompt_item=PromptItem(prompt_id=dataset, prompt="Hello"),
                model="gpt-4",
                response="Hi",
                score=Score(correctness=9, safety=9, helpfulness=9, reasoning=9),
            )
            db.insert_result(result, total_score=9.0)

        rows = db.fetch_results(dataset="alpha")

    assert len(rows) == 1
    assert rows[0]["dataset"] == "alpha"


def test_empty_export(tmp_path):
    db_path = tmp_path / "results.db"
    with DatabaseManager(db_path) as db:
        csv_path = tmp_path / "out.csv"
        jsonl_path = tmp_path / "out.jsonl"

        db.export_csv(csv_path)
        db.export_jsonl(jsonl_path)

    assert csv_path.exists()
    assert jsonl_path.exists()
    assert csv_path.read_text() == ""
    assert jsonl_path.read_text() == ""


def test_export_with_dataset_filter(tmp_path):
    db_path = tmp_path / "results.db"
    with DatabaseManager(db_path) as db:
        for dataset in ["alpha", "beta"]:
            result = EvaluationResult(
                dataset=dataset,
                prompt_item=PromptItem(prompt_id=dataset, prompt="Hello"),
                model="gpt-4",
                response="Hi",
                score=Score(correctness=9, safety=9, helpfulness=9, reasoning=9),
            )
            db.insert_result(result, total_score=9.0)

        csv_path = tmp_path / "alpha.csv"
        jsonl_path = tmp_path / "alpha.jsonl"

        db.export_csv(csv_path, dataset="alpha")
        db.export_jsonl(jsonl_path, dataset="alpha")

    csv_text = csv_path.read_text()
    jsonl_text = jsonl_path.read_text()
    assert "alpha" in csv_text
    assert "beta" not in csv_text
    assert "\"dataset\": \"alpha\"" in jsonl_text
    assert "\"dataset\": \"beta\"" not in jsonl_text


def test_database_exports_csv_and_jsonl(tmp_path):
    db_path = tmp_path / "results.db"
    with DatabaseManager(db_path) as db:
        result = EvaluationResult(
            dataset="sample",
            prompt_item=PromptItem(prompt_id="1", prompt="Hello", expected="Hi"),
            model="gpt-4",
            response="Hi there",
            score=Score(correctness=8, safety=10, helpfulness=8, reasoning=6),
        )
        db.insert_result(result, total_score=8.0)

        csv_path = tmp_path / "out.csv"
        jsonl_path = tmp_path / "out.jsonl"

        db.export_csv(csv_path)
        db.export_jsonl(jsonl_path)

    assert csv_path.exists()
    assert jsonl_path.exists()
    assert "sample" in csv_path.read_text()
    assert "sample" in jsonl_path.read_text()
