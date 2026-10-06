"""
model_backend.py

This is the file that actually loads and calls models -- the part that was
previously just a typed placeholder in run_pipeline.py. Two separate models:

  - BASE model (answers every condition): Qwen3-7B, per your model choice.
    This is the ONLY model that ever produces or revises an answer, in every
    condition (baseline / always_on / adaptive_rereason / adaptive_debate).

  - JUDGE model (scores correctness only, never generates/revises answers):
    a larger local checkpoint, loaded separately, called only to output a
    correct/incorrect judgment. This is what keeps the bias fix real -- if
    you accidentally let the judge model "help" generate the answer, you're
    back to the original judge-asymmetry confound.

Both are loaded once (model loading is slow) and exposed as plain callables
so run_pipeline.py can pass them straight into `base_generate_fn` and
`judge_correctness_fn`.

VRAM note: Qwen3-7B (bf16) needs ~16GB. A 32B-class local judge (bf16) needs
~65GB, or ~18-20GB at 4-bit. If your GPU budget doesn't have room for both
loaded at once, load the base model, run ALL base-model generation for a full
condition first, free it (`del model; torch.cuda.empty_cache()`), then load
the judge model to score the whole CSV in a second pass -- see
`judge_pass_from_csv()` at the bottom for that two-pass pattern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Base model (answers every condition)
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str


def load_base_model(model_name: str = "Qwen/Qwen3-8B", dtype=torch.bfloat16) -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    return LoadedModel(model=model, tokenizer=tokenizer, device=device)


def make_base_generate_fn(loaded: LoadedModel, max_new_tokens: int = 512, temperature: float = 0.7):
    """
    Returns a `base_generate_fn(prompt: str) -> str` closure using the loaded
    base model. Temperature > 0 matters here: the self-consistency /entropy
    triggers rely on independent samples actually varying -- greedy decoding
    (temperature=0) will make every "independent" generation identical and the
    trigger will never fire. Keep sampling on for base_generate_fn.
    """
    model, tokenizer = loaded.model, loaded.tokenizer

    def _generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(loaded.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print("\n========== RAW MODEL OUTPUT ==========")
        print(text)
        print("======================================\n")
        return text

    return _generate


def make_extract_answer_fn(dataset: str = "generic"):
    """
    Dataset-aware answer extractor.

    StrategyQA is a yes/no task, so normalize the model output to:
    "yes", "no", or "unknown".

    Other datasets retain the original generic extraction behavior.
    """
    pattern = re.compile(r"final answer[:\-]?\s*(.+)", re.IGNORECASE)

    def _extract_strategyqa(raw_text: str) -> str:
        # Prefer an explicit final answer.
        match = pattern.search(raw_text)
        if match:
            tail = match.group(1).strip().lower()
            if re.search(r"\byes\b", tail):
                return "yes"
            if re.search(r"\bno\b", tail):
                return "no"

        # Otherwise look for yes/no near the end of the response.
        tail = raw_text[-500:]

        yes_matches = list(re.finditer(r"\byes\b", tail, re.IGNORECASE))
        no_matches = list(re.finditer(r"\bno\b", tail, re.IGNORECASE))

        last_yes = yes_matches[-1].start() if yes_matches else -1
        last_no = no_matches[-1].start() if no_matches else -1

        if last_yes == -1 and last_no == -1:
            return "unknown"

        return "yes" if last_yes > last_no else "no"

    def _extract_generic(raw_text: str) -> str:
        match = pattern.search(raw_text)
        if match:
            return match.group(1).strip().split("\n")[0]
        lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else raw_text.strip()

    return _extract_strategyqa if dataset.lower() == "strategyqa" else _extract_generic


# ---------------------------------------------------------------------------
# Judge model (correctness scoring ONLY -- never generates/revises an answer)
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """You are a strict grader. Given a question, the gold (correct) answer, and a
model's predicted answer, decide if the prediction is correct. Minor wording
differences are fine; the underlying answer must match.

Question: {question}
Gold answer: {gold}
Predicted answer: {prediction}

Respond with exactly one word: CORRECT or INCORRECT."""


def load_judge_model(model_name: str = "Qwen/Qwen2.5-32B-Instruct", dtype=torch.bfloat16,
                      use_4bit: bool = False) -> LoadedModel:
    """
    Default is a 32B-class local checkpoint -- big enough to be a meaningfully
    stronger grader than the 7B base model, without requiring 70B-scale VRAM.

    use_4bit: REQUIRED True on cards under ~70GB free VRAM -- 32B in bf16 is
        ~65GB and will not fit on a 40GB card at all, not even alone. 4-bit
        (via bitsandbytes) brings this down to ~18-20GB. On a 40GB card, load
        this AFTER unloading the base model (see unload() + the two-pass
        pattern in judge_pass_from_csv below) rather than holding both models
        in memory at once -- even 16GB (base) + ~19GB (4-bit judge) plus
        generation activation overhead is too tight to trust for a multi-hour
        run on 40GB.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant_config, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
    model.eval()
    device = next(model.parameters()).device
    return LoadedModel(model=model, tokenizer=tokenizer, device=device)


def make_judge_correctness_fn(loaded: LoadedModel, max_new_tokens: int = 8):
    """
    Returns `judge_correctness_fn(question, gold, prediction) -> bool`.
    Greedy decoding on purpose here (do_sample=False) -- you want the judge
    to be deterministic/reproducible, unlike the base model's sampling used
    for the uncertainty trigger.
    """
    model, tokenizer = loaded.model, loaded.tokenizer

    def _judge(question: str, gold: str, prediction: str) -> bool:
        prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, gold=gold, prediction=prediction)
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )

        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids["input_ids"]

        input_ids = input_ids.to(loaded.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        verdict = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()
        return verdict.startswith("CORRECT")

    return _judge


def unload(loaded: LoadedModel):
    """Free GPU memory -- call this before loading the second model if VRAM is tight."""
    del loaded.model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Two-pass pattern for tight VRAM budgets: generate everything with the base
# model first (no judge loaded), write raw predictions to CSV, THEN load the
# judge and score the whole file in a second pass.
# ---------------------------------------------------------------------------

def judge_pass_from_csv(csv_path: str, judge_model_name: str = "Qwen/Qwen2.5-32B-Instruct",
                         use_4bit: bool = True):
    """
    Two-pass scoring: run generation with --skip_judge first (base model only
    loaded), THEN call this separately once the base model process has exited
    and its VRAM is free. use_4bit defaults True since this is the pattern
    you need on cards where the base and judge models can't both fit at once
    (e.g. 40GB cards, where 32B judge alone needs 4-bit regardless).

    Handles both CSV schemas: the simple run_condition schema (one
    "final_answer" column) and the adaptive_paired schema (three answer
    columns: baseline_answer, rereason_answer, critique_answer -- the latter
    two may be blank on non-triggered rows, which are skipped).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    judge_loaded = load_judge_model(judge_model_name, use_4bit=use_4bit)
    judge_fn = make_judge_correctness_fn(judge_loaded)

    if "final_answer" in df.columns:
        df["correct"] = [
            judge_fn(row["question"], row["gold_answer"], row["final_answer"])
            for _, row in df.iterrows()
        ]
    else:
        # adaptive_paired schema
        for col in ["baseline_answer", "rereason_answer", "critique_answer"]:
            correct_col = col.replace("_answer", "_correct")
            df[correct_col] = [
                judge_fn(row["question"], row["gold_answer"], row[col])
                if isinstance(row.get(col), str) and row[col].strip() else ""
                for _, row in df.iterrows()
            ]

    df.to_csv(csv_path, index=False)
    unload(judge_loaded)
