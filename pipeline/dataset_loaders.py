from __future__ import annotations

import json
import random
from datetime import datetime
from typing import Iterator


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def load_strategyqa(
    path: str = "strategyqa_dev.json",
    n: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[dict]:
    """
    Yields {"question": ..., "gold": "yes"/"no"} dicts, matching your original
    ex["answer"] boolean -> "yes"/"no" mapping.
    """
    log("Loading StrategyQA...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    examples = [
        {"question": ex["question"], "gold": "yes" if ex["answer"] else "no"}
        for ex in data
    ]

    if shuffle:
        random.Random(seed).shuffle(examples)
    if n is not None:
        examples = examples[:n]

    log(f"StrategyQA: loaded {len(examples)} samples")
    for ex in examples:
        yield ex


def load_hotpotqa(
    path: str = "hotpotqa_fullwiki.jsonl",
    n: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[dict]:
    
    log("Loading HotpotQA...")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            examples.append({"question": ex["question"], "gold": ex["answer"]})

    if shuffle:
        random.Random(seed).shuffle(examples)
    if n is not None:
        examples = examples[:n]

    log(f"HotpotQA: loaded {len(examples)} samples")
    for ex in examples:
        yield ex


def build_lookup(
    strategyqa_path: str = "strategyqa_dev.json",
    hotpotqa_path: str = "hotpotqa_fullwiki.jsonl",
) -> dict:

    lookup = {}

    log("Loading StrategyQA...")
    with open(strategyqa_path, encoding="utf-8") as f:
        data = json.load(f)
        for ex in data:
            lookup[normalize(ex["question"])] = "yes" if ex["answer"] else "no"

    log("Loading HotpotQA...")
    with open(hotpotqa_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            lookup[normalize(ex["question"])] = ex["answer"]

    return lookup
