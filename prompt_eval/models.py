from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PromptItem:
    prompt_id: str
    prompt: str
    expected: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    detected_language: Optional[str] = None


@dataclass
class Score:
    correctness: int
    safety: int
    helpfulness: int
    reasoning: int
    overall: float = 0.0

    def as_dict(self) -> Dict[str, int | float]:
        return {
            "correctness": self.correctness,
            "safety": self.safety,
            "helpfulness": self.helpfulness,
            "reasoning": self.reasoning,
            "overall": self.overall,
        }


@dataclass
class EvaluationResult:
    dataset: str
    prompt_item: PromptItem
    model: str
    response: str
    score: Score
    tokens_used: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset,
            "prompt_id": self.prompt_item.prompt_id,
            "prompt": self.prompt_item.prompt,
            "expected": self.prompt_item.expected,
            "metadata": self.prompt_item.metadata,
            "detected_language": self.prompt_item.detected_language,
            "model": self.model,
            "response": self.response,
            "correctness": self.score.correctness,
            "safety": self.score.safety,
            "helpfulness": self.score.helpfulness,
            "reasoning": self.score.reasoning,
            "overall": self.score.overall,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
        }
