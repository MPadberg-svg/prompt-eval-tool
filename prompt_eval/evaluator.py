from __future__ import annotations

import json
import os
import time
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from langdetect import detect
from collections import Counter

from .database import DatabaseManager
from .models import EvaluationResult, PromptItem
from .scoring import ScoringEngine, load_weights

logger = logging.getLogger(__name__)

def load_jsonl_dataset(dataset_path: str | Path) -> List[PromptItem]:
    """
    Loads a JSON Lines dataset and returns a list of PromptItem objects.
    Detects the language of each prompt and adds it to PromptItem.
    """
    from langdetect.lang_detect_exception import LangDetectException

    path = Path(dataset_path)
    items: List[PromptItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if not prompt:
                raise ValueError(
                    f"Missing 'prompt' in {str(dataset_path)}, line {idx}. "
                    f"Found keys: {sorted(record.keys())}"
                )
            try:
                detected_language = detect(prompt)
            except LangDetectException:
                detected_language = "unknown"
            items.append(
                PromptItem(
                    prompt_id=str(record.get("id", idx)),
                    prompt=prompt,
                    expected=record.get("expected"),
                    metadata=record.get("metadata") or {},
                    detected_language=detected_language,
                )
            )
    return items

def _load_temperature_from_config(config_path: str | Path) -> float:
    import yaml

    path = Path(config_path)
    if not path.exists():
        return 0.0
    with path.open("r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            return float(config.get("temperature", 0.0))
        except Exception:
            return 0.0


def _load_multilingual_weights_from_config(
    config_path: str | Path,
) -> Dict[str, Dict[str, float]]:
    import yaml

    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f) or {}
        except Exception:
            return {}

    multilingual_weights = config.get("multilingual_weights") or {}
    allowed_keys = {"correctness", "safety", "helpfulness", "reasoning"}
    normalized: Dict[str, Dict[str, float]] = {}
    for language, weights in multilingual_weights.items():
        if not isinstance(weights, dict):
            continue
        normalized[str(language)] = {
            key: float(value)
            for key, value in weights.items()
            if key in allowed_keys
        }
    return normalized

class OpenAIResponseGenerator:
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.0):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.client = None

    def generate(self, model: str, prompt: str) -> Tuple[str, int]:
        """
        Generates a response using OpenAI API with exponential backoff (up to 3 attempts)
        Returns (content, tokens_used) on success.
        """
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for live API calls")
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)

        from openai import APIError, RateLimitError

        max_attempts = 3
        backoff = [1, 2, 4]
        last_err: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    timeout=60,
                )
                content = response.choices[0].message.content or ""
                tokens_used = getattr(response.usage, "total_tokens", None)
                if tokens_used is None and hasattr(response, 'usage'):
                    tokens_used = response.usage.get('total_tokens', 0)
                if tokens_used is None:
                    tokens_used = 0
                return (content, tokens_used)
            except (RateLimitError, APIError) as e:
                logger.warning(f"OpenAI API error ({type(e).__name__}) on attempt {attempt+1}/{max_attempts}: {e}")
                last_err = e
                if attempt < max_attempts - 1:
                    time.sleep(backoff[attempt])
                else:
                    break
            except Exception as e:
                logger.error(f"Unexpected OpenAI API exception: {e}")
                last_err = e
                break

        raise RuntimeError(f"OpenAI API call failed after {max_attempts} attempts: {last_err}")

class PromptEvaluator:
    def __init__(
        self,
        db_path: str | Path,
        config_path: str | Path = "config.yaml",
        response_generator: Optional[Callable[[str, str], Tuple[str, int]]] = None,
    ):
        """
        Initializes the evaluator, scoring engine, and database. Reads temperature from config.
        """
        self.temperature = _load_temperature_from_config(config_path)
        weights = load_weights(config_path)
        self.scorer = ScoringEngine(weights)
        self.multilingual_weights = _load_multilingual_weights_from_config(config_path)
        self.db = DatabaseManager(db_path)
        if response_generator is None:
            live_generator = OpenAIResponseGenerator(temperature=self.temperature)
            self.response_generator = live_generator.generate
        else:
            self.response_generator = response_generator

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "PromptEvaluator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _get_cost_usd(self, model: str, tokens: int) -> float:
        """
        Estimates cost in USD for a given model and number of tokens.
        """
        model_lower = model.lower()
        if "gpt-4" in model_lower:
            return tokens * 0.03 / 1000
        elif "gpt-3.5" in model_lower:
            return tokens * 0.002 / 1000
        else:
            # Default to GPT-4 price if unknown
            return tokens * 0.03 / 1000

    def evaluate_dataset(
        self,
        dataset_path: str | Path,
        model: str,
        limit: Optional[int] = None,
        resume: bool = False,
    ) -> List[EvaluationResult]:
        """
        Evaluates the dataset, optionally resuming from previous results.
        Adds cost tracking, detected language, progress logging, and skips already evaluated prompts if resume=True.
        Returns list of EvaluationResult.
        """
        dataset_name = Path(dataset_path).stem
        prompts = load_jsonl_dataset(dataset_path)
        N = len(prompts)
        logger.info(f"Evaluating {N} prompts from [{dataset_name}]")
        if limit is not None:
            prompts = prompts[:limit]
            N = len(prompts)

        # Resume/skipping already evaluated prompts
        skipped = 0
        if resume:
            evaluated_set = set(
                self.db.get_evaluated_prompt_ids(dataset_name, model)
            )
        else:
            evaluated_set = set()

        results: List[EvaluationResult] = []
        language_counter: Counter[str] = Counter()
        total_tokens = 0
        total_cost = 0.0

        for idx, item in enumerate(prompts, start=1):
            if resume and (item.prompt_id, model) in evaluated_set:
                skipped += 1
                continue

            language = getattr(item, "detected_language", None)
            language_counter[language] += 1

            response, tokens = self.response_generator(model, item.prompt)
            total_tokens += tokens
            cost_usd = self._get_cost_usd(model, tokens)
            total_cost += cost_usd

            original_weights = self.scorer.get_weights()
            language_weights = self.multilingual_weights.get(language or "")
            active_weights = original_weights
            if language_weights:
                weights = dict(original_weights)
                weights.update(language_weights)
                active_weights = weights
                self.scorer.set_weights(weights)
            score = self.scorer.score(item, response)

            result = EvaluationResult(
                dataset=dataset_name,
                prompt_item=item,
                model=model,
                response=response,
                score=score,
                tokens_used=tokens,
                cost_usd=cost_usd,
            )
            total = self.scorer.weighted_total(score, weights_override=active_weights)
            if language_weights:
                self.scorer.set_weights(original_weights)
            self.db.insert_result(result, total_score=total)
            results.append(result)

            if idx % 10 == 0:
                logger.info(f"Progress: {idx}/{N}")

        logger.info(f"Completed {len(results)} evaluations")
        if skipped > 0:
            logger.info(f"Skipped {skipped} previously evaluated prompts (resume enabled)")

        # Language distribution
        lang_dist_str = ", ".join(f"{lang}: {count}" for lang, count in language_counter.items())
        logger.info(f"Detected language distribution: {lang_dist_str}")
        logger.info(f"Total cost for [{dataset_name}] ({model}): ${total_cost:.4f}")

        return results

    def evaluate_batch(
        self, dataset_paths: Iterable[str], model: str, limit: Optional[int] = None, resume: bool = False
    ) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for path in dataset_paths:
            results = self.evaluate_dataset(path, model=model, limit=limit, resume=resume)
            summary[path] = len(results)
        return summary

    def compare_models(
        self, dataset_path: str | Path, models: List[str], limit: Optional[int] = None, resume: bool = False
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Evaluates dataset against each model provided, returns evaluation results.
        Logs average score per model.
        """
        all_results: Dict[str, List[EvaluationResult]] = {}
        for model in models:
            results = self.evaluate_dataset(dataset_path, model, limit=limit, resume=resume)
            all_results[model] = results
            # Average score
            if results:
                avg_score = sum(self.scorer.weighted_total(r.score) for r in results) / len(results)
                logger.info(f"Average score for model '{model}': {avg_score:.4f}")
        return all_results
