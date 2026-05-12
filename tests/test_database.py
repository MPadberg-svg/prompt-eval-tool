from prompt_eval.database import DatabaseManager
from prompt_eval.models import EvaluationResult, PromptItem, Score


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
