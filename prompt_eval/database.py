from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Optional

from .models import EvaluationResult


class DatabaseManager:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def _create_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                dataset TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                expected TEXT,
                model TEXT NOT NULL,
                response TEXT NOT NULL,
                correctness INTEGER NOT NULL,
                safety INTEGER NOT NULL,
                helpfulness INTEGER NOT NULL,
                reasoning INTEGER NOT NULL,
                total_score REAL NOT NULL,
                tokens_used INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0.0,
                detected_language TEXT
            )
            """
        )
        self.conn.commit()

    def insert_result(self, result: EvaluationResult, total_score: float) -> None:
        self.conn.execute(
            """
            INSERT INTO evaluations (
                dataset, prompt_id, prompt, expected, model, response,
                correctness, safety, helpfulness, reasoning, total_score,
                tokens_used, cost_usd, detected_language
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.dataset,
                result.prompt_item.prompt_id,
                result.prompt_item.prompt,
                result.prompt_item.expected,
                result.model,
                result.response,
                result.score.correctness,
                result.score.safety,
                result.score.helpfulness,
                result.score.reasoning,
                total_score,
                result.tokens_used,
                result.cost_usd,
                result.prompt_item.detected_language,
            ),
        )
        self.conn.commit()

    def fetch_results(self, dataset: Optional[str] = None) -> list[dict]:
        if dataset:
            cursor = self.conn.execute(
                "SELECT * FROM evaluations WHERE dataset = ? ORDER BY rowid ASC",
                (dataset,),
            )
        else:
            cursor = self.conn.execute("SELECT * FROM evaluations ORDER BY rowid ASC")
        return [dict(row) for row in cursor.fetchall()]

    def get_evaluated_prompt_ids(self, dataset: str, model: str) -> set[tuple[str, str]]:
        cursor = self.conn.execute(
            """
            SELECT prompt_id, model
            FROM evaluations
            WHERE dataset = ? AND model = ?
            """,
            (dataset, model),
        )
        return {(row["prompt_id"], row["model"]) for row in cursor.fetchall()}

    def export_csv(self, output_path: str | Path, dataset: Optional[str] = None) -> None:
        rows = self.fetch_results(dataset=dataset)
        path = Path(output_path)
        if not rows:
            path.write_text("")
            return

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def export_jsonl(self, output_path: str | Path, dataset: Optional[str] = None) -> None:
        rows = self.fetch_results(dataset=dataset)
        path = Path(output_path)
        if not rows:
            path.write_text("")
            return

        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
