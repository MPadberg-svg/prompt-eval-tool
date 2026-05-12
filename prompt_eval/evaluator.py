from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .database import DatabaseManager
from .models import EvaluationResult, PromptItem
from .scoring import ScoringEngine, load_weights


class OpenAIResponseGenerator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None

    def generate(self, model: str, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for live API calls")
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""


def load_jsonl_dataset(dataset_path: str | Path) -> List[PromptItem]:
    path = Path(dataset_path)
    items: List[PromptItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if not prompt:
                raise ValueError(f"Missing 'prompt' in {dataset_path} line {idx}")
            items.append(
                PromptItem(
                    prompt_id=str(record.get("id", idx)),
                    prompt=prompt,
                    expected=record.get("expected"),
                    metadata=record.get("metadata") or {},
                )
            )
    return items


class PromptEvaluator:
    def __init__(
        self,
        db_path: str | Path,
        config_path: str | Path = "config.yaml",
        response_generator: Optional[Callable[[str, str], str]] = None,
    ):
        weights = load_weights(config_path)
        self.scorer = ScoringEngine(weights)
        self.db = DatabaseManager(db_path)
        if response_generator is None:
            live_generator = OpenAIResponseGenerator()
            self.response_generator = live_generator.generate
        else:
            self.response_generator = response_generator

    def evaluate_dataset(
        self,
        dataset_path: str | Path,
        model: str,
        limit: Optional[int] = None,
    ) -> List[EvaluationResult]:
        dataset_name = Path(dataset_path).stem
        prompts = load_jsonl_dataset(dataset_path)
        if limit is not None:
            prompts = prompts[:limit]

        results: List[EvaluationResult] = []
        for item in prompts:
            response = self.response_generator(model, item.prompt)
            score = self.scorer.score(item, response)
            result = EvaluationResult(
                dataset=dataset_name,
                prompt_item=item,
                model=model,
                response=response,
                score=score,
            )
            total = self.scorer.weighted_total(score)
            self.db.insert_result(result, total_score=total)
            results.append(result)
        return results

    def evaluate_batch(self, dataset_paths: Iterable[str], model: str, limit: Optional[int] = None) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for path in dataset_paths:
            results = self.evaluate_dataset(path, model=model, limit=limit)
            summary[path] = len(results)
        return summary
