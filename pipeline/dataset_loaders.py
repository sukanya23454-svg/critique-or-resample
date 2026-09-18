from __future__ import annotations

import json
import random
from datetime import datetime
from typing import Iterator


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _get_field(example: dict, candidates: list[str], dataset_name: str):
    """Try several possible field names in order; raise with the actual
    available keys if none match, rather than silently returning None or
    crashing with a bare KeyError -- useful since we haven't confirmed the
    exact schema of your locally-downloaded files."""
    for c in candidates:
        if c in example:
            return example[c]
    raise KeyError(
        f"None of {candidates} found in a {dataset_name} example. "
        f"Actual available fields: {list(example.keys())}. "
        f"Update the candidate list in dataset_loaders.py to match your file's real schema."
    )


def load_strategyqa(
    path: str = "data/strategyqa/dev.jsonl",
    n: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[dict]:
    """
    Yields {"question": ..., "gold": "yes"/"no"} dicts.

    FIXED: was json.load() expecting a single JSON array -- your actual file
    is .jsonl (one JSON object per line), same format as HotpotQA. Reads
    line-by-line now, matching the real file format.

    Answer field handling is defensive: some StrategyQA dumps store answer as
    a Python bool (True/False), others as the string "yes"/"no" already --
    both are handled below without assuming which one your download used.
    """
    log(f"Loading StrategyQA from {path}...")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            question = _get_field(ex, ["question"], "StrategyQA")
            answer = _get_field(ex, ["answer"], "StrategyQA")
            if isinstance(answer, bool):
                gold = "yes" if answer else "no"
            else:
                gold = str(answer).strip().lower()
            examples.append({"question": question, "gold": gold})

    if shuffle:
        random.Random(seed).shuffle(examples)
    if n is not None:
        examples = examples[:n]

    log(f"StrategyQA: loaded {len(examples)} samples")
    for ex in examples:
        yield ex


def load_hotpotqa(
    path: str = "data/hotpotqa/dev.jsonl",
    n: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[dict]:
    """
    Yields {"question": ..., "gold": ...} dicts.

    Path fixed to point at your real local file (data/hotpotqa/dev.jsonl).
    NEVER point this at train.jsonl -- that's the ~536MB file that broke your
    git push; dev.jsonl is what you actually want for evaluation anyway.
    """
    log(f"Loading HotpotQA from {path}...")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            question = _get_field(ex, ["question"], "HotpotQA")
            answer = _get_field(ex, ["answer"], "HotpotQA")
            examples.append({"question": question, "gold": answer})

    if shuffle:
        random.Random(seed).shuffle(examples)
    if n is not None:
        examples = examples[:n]

    log(f"HotpotQA: loaded {len(examples)} samples")
    for ex in examples:
        yield ex


def build_lookup(
    strategyqa_path: str = "data/strategyqa/dev.jsonl",
    hotpotqa_path: str = "data/hotpotqa/dev.jsonl",
) -> dict:
    """Merged normalized-question -> gold-answer lookup across both datasets,
    kept from your original pattern. Not used by run_pipeline.py directly."""
    lookup = {}

    log("Loading StrategyQA...")
    with open(strategyqa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            answer = ex["answer"]
            gold = "yes" if isinstance(answer, bool) and answer else "no" if isinstance(answer, bool) else str(answer).strip().lower()
            lookup[normalize(ex["question"])] = gold

    log("Loading HotpotQA...")
    with open(hotpotqa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            lookup[normalize(ex["question"])] = ex["answer"]

    return lookup


if __name__ == "__main__":
    
    print("=== StrategyQA sample ===")
    for ex in load_strategyqa(n=1):
        print(ex)

    print("\n=== HotpotQA sample ===")
    for ex in load_hotpotqa(n=1):
        print(ex)