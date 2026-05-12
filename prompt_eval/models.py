from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PromptItem:
    prompt_id: str
    prompt: str
    expected: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Score:
    correctness: int
    safety: int
    helpfulness: int
    reasoning: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "correctness": self.correctness,
            "safety": self.safety,
            "helpfulness": self.helpfulness,
            "reasoning": self.reasoning,
        }


@dataclass
class EvaluationResult:
    dataset: str
    prompt_item: PromptItem
    model: str
    response: str
    score: Score
