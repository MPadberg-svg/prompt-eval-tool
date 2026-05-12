from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import EvaluationResult


class DatabaseManager:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def insert_result(self, result: EvaluationResult, total_score: float) -> None:
        self.conn.execute(
            """
            INSERT INTO evaluations (
                dataset, prompt_id, prompt, expected, model, response,
                correctness, safety, helpfulness, reasoning, total_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        self.conn.commit()

    def fetch_results(self, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
        if dataset:
            rows = self.conn.execute(
                "SELECT * FROM evaluations WHERE dataset = ? ORDER BY id", (dataset,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM evaluations ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    def export_csv(self, output_path: str | Path, dataset: Optional[str] = None) -> None:
        rows = self.fetch_results(dataset=dataset)
        output = Path(output_path)
        if not rows:
            output.write_text("")
            return

        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def export_jsonl(self, output_path: str | Path, dataset: Optional[str] = None) -> None:
        rows = self.fetch_results(dataset=dataset)
        output = Path(output_path)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.conn.close()
