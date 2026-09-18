from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional


AnswerFn = Callable[[str], str]
def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


@dataclass
class TriggerResult:
    is_uncertain: bool
    signal_name: str
    raw_value: float           # disagreement rate, entropy, or estimated logprob-based score
    samples: list[str] = field(default_factory=list)  # the raw answers collected, for logging


class SelfConsistencyTrigger:
    def __init__(self, k: int = 3, majority_threshold: float = 1.0):
        assert k >= 2, "need at least 2 generations to measure disagreement"
        self.k = k
        self.majority_threshold = majority_threshold  # fraction that must agree to call it "confident"

    def is_uncertain(self, question: str, answer_fn: AnswerFn) -> TriggerResult:
        samples = [normalize_answer(answer_fn(question)) for _ in range(self.k)]
        counts = Counter(samples)
        top_answer, top_count = counts.most_common(1)[0]
        agreement_rate = top_count / self.k
        uncertain = agreement_rate < self.majority_threshold
        return TriggerResult(
            is_uncertain=uncertain,
            signal_name="self_consistency",
            raw_value=1.0 - agreement_rate,   # disagreement rate
            samples=samples,
        )


class EntropyTrigger:
    def __init__(self, k: int = 5, threshold: float = 0.4):
        assert k >= 2
        self.k = k
        self.threshold = threshold

    @staticmethod
    def _normalized_entropy(counts: Counter, k: int) -> float:
        if len(counts) <= 1:
            return 0.0
        h = 0.0
        for c in counts.values():
            p = c / k
            h -= p * math.log(p + 1e-12)
        h_max = math.log(len(counts))
        return h / h_max if h_max > 0 else 0.0

    def is_uncertain(self, question: str, answer_fn: AnswerFn) -> TriggerResult:
        samples = [normalize_answer(answer_fn(question)) for _ in range(self.k)]
        counts = Counter(samples)
        entropy = self._normalized_entropy(counts, self.k)
        uncertain = entropy > self.threshold
        return TriggerResult(
            is_uncertain=uncertain,
            signal_name="entropy",
            raw_value=entropy,
            samples=samples,
        )


class LogProbTrigger:
    def __init__(self, threshold: float = -0.75):
        self.threshold = threshold

    def is_uncertain(
        self,
        question: str,
        logprob_fn: Callable[[str], tuple[str, float]],
    ) -> TriggerResult:
        answer, mean_logprob = logprob_fn(question)
        uncertain = mean_logprob < self.threshold
        return TriggerResult(
            is_uncertain=uncertain,
            signal_name="logprob",
            raw_value=mean_logprob,
            samples=[normalize_answer(answer)],
        )


def trigger_error_correlation(trigger_flags: list[bool], correctness_flags: list[bool]) -> dict:
    assert len(trigger_flags) == len(correctness_flags)
    tp = sum(1 for t, c in zip(trigger_flags, correctness_flags) if t and not c)   # triggered AND wrong
    fp = sum(1 for t, c in zip(trigger_flags, correctness_flags) if t and c)       # triggered AND right
    fn = sum(1 for t, c in zip(trigger_flags, correctness_flags) if not t and not c)  # missed AND wrong
    tn = sum(1 for t, c in zip(trigger_flags, correctness_flags) if not t and c)   # not triggered AND right

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision_trigger_predicts_error": precision,
        "recall_trigger_predicts_error": recall,
        "f1_trigger_predicts_error": f1,
        "trigger_rate": sum(trigger_flags) / len(trigger_flags),
        "base_error_rate": sum(1 for c in correctness_flags if not c) / len(correctness_flags),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
