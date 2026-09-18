from __future__ import annotations

from functools import lru_cache


class TokenCounter:
    def __init__(self, model_name_or_path: str):
        from transformers import AutoTokenizer  # local import: keep this file importable
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_name_or_path = model_name_or_path

    @lru_cache(maxsize=4096)
    def _count_cached(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count(self, text: str) -> int:
        if not text:
            return 0
        return self._count_cached(text)

    def count_many(self, texts: list[str]) -> int:
        return sum(self.count(t) for t in texts)


def total_pipeline_tokens(counter: TokenCounter, *text_fields: str) -> int:
    """
    Sum tokens across every text field produced for one sample (prompt(s) +
    generation(s)). Pass every prompt AND every generated string for the
    condition being measured -- e.g. for adaptive debate on an escalated
    sample: initial_reasoning + critique_prompt + critique + revise_prompt +
    revised_reasoning. Undercounting by only measuring the final answer text
    will make adaptive/debate look artificially cheap.
    """
    return counter.count_many([t for t in text_fields if t])
