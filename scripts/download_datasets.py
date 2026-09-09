from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

STRATEGYQA_DIR = DATA_DIR / "strategyqa"
HOTPOTQA_DIR = DATA_DIR / "hotpotqa"

STRATEGYQA_DIR.mkdir(parents=True, exist_ok=True)
HOTPOTQA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# StrategyQA
# ============================================================

strategyqa_train_path = STRATEGYQA_DIR / "train.jsonl"
strategyqa_dev_path = STRATEGYQA_DIR / "dev.jsonl"

if strategyqa_train_path.exists() and strategyqa_dev_path.exists():
    print("StrategyQA already exists. Skipping download.")
else:
    print("Downloading StrategyQA...")

    strategyqa = load_dataset(
        "wics/strategy-qa",
        trust_remote_code=True,
    )

    strategyqa_split = strategyqa["test"].train_test_split(
        test_size=0.1,
        seed=42,
    )

    strategyqa_split["train"].to_json(
        strategyqa_train_path,
        orient="records",
        lines=True,
    )

    strategyqa_split["test"].to_json(
        strategyqa_dev_path,
        orient="records",
        lines=True,
    )

    print(f"Saved StrategyQA train -> {strategyqa_train_path}")
    print(f"Saved StrategyQA dev   -> {strategyqa_dev_path}")


# ============================================================
# HotpotQA
# ============================================================

hotpotqa_train_path = HOTPOTQA_DIR / "train.jsonl"
hotpotqa_dev_path = HOTPOTQA_DIR / "dev.jsonl"

if hotpotqa_train_path.exists() and hotpotqa_dev_path.exists():
    print("HotpotQA already exists. Skipping download.")
else:
    print("\nDownloading HotpotQA...")

    hotpotqa = load_dataset(
        "hotpotqa/hotpot_qa",
        "fullwiki",
    )

    hotpotqa["train"].to_json(
        hotpotqa_train_path,
        orient="records",
        lines=True,
    )

    hotpotqa["validation"].to_json(
        hotpotqa_dev_path,
        orient="records",
        lines=True,
    )

    print(f"Saved HotpotQA train -> {hotpotqa_train_path}")
    print(f"Saved HotpotQA dev   -> {hotpotqa_dev_path}")


# ============================================================
# Done
# ============================================================

print("\n========================================")
print("Dataset setup complete!")
print("========================================")

print("\nData directory:", DATA_DIR)

print("\nExpected files:")
print(f"  {strategyqa_train_path}")
print(f"  {strategyqa_dev_path}")
print(f"  {hotpotqa_train_path}")
print(f"  {hotpotqa_dev_path}")

