from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Callable, Iterable

from pipeline.uncertainty import SelfConsistencyTrigger, EntropyTrigger, trigger_error_correlation
from pipeline.critique_loop import CritiqueDebateLoop, AdaptiveRereasonLoop
from pipeline.token_utils import TokenCounter, total_pipeline_tokens


CSV_FIELDS = [
    "sample_id", "dataset", "mode", "question",
    "gold_answer", "final_answer", "correct",
    "trigger_fired", "trigger_signal", "trigger_raw_value",
    "n_tokens", "wall_seconds",
]


def build_trigger(trigger_type: str):
    if trigger_type == "self_consistency":
        return SelfConsistencyTrigger(k=3)
    if trigger_type == "entropy":
        return EntropyTrigger(k=5, threshold=0.4)
    raise ValueError(f"unknown trigger_type: {trigger_type}")


PAIRED_CSV_FIELDS = [
    "sample_id", "dataset", "question", "gold_answer",
    "baseline_answer", "baseline_correct", "baseline_tokens",
    "trigger_fired", "trigger_signal", "trigger_raw_value", "trigger_tokens",
    "rereason_answer", "rereason_correct", "rereason_tokens",
    "critique_answer", "critique_correct", "critique_tokens",
    "wall_seconds",
]


def run_paired_condition(
    dataset: Iterable[dict],
    dataset_name: str,
    base_generate_fn: Callable[[str], str],
    extract_answer_fn: Callable[[str], str],
    judge_correctness_fn: Callable[[str, str, str], bool],
    token_counter: TokenCounter,
    trigger_type: str,
    out_csv_path: str,
    stage_generate_fn: Callable[[str], str] | None = None,
):
    """
    THE mode that actually answers RQ3. Runs the uncertainty trigger ONCE per
    question, then -- if and only if triggered -- applies BOTH escalation arms
    (AdaptiveRereasonLoop and CritiqueDebateLoop) to that SAME triggered
    instance, so per-question outcomes are directly paired: for every
    triggered question you get both a rereason_correct and a critique_correct
    value, letting you build the (A fixed/B fixed, A fixed/B wrong, ...) 2x2
    table rather than comparing two independently-triggered subsets.

    Use this instead of running --mode adaptive_rereason and --mode
    adaptive_debate as two separate invocations -- those independently
    re-sample the trigger (temperature > 0), so nothing guarantees the same
    question triggers in both runs, which silently breaks the paired analysis.
    """
    trigger = build_trigger(trigger_type)
    rereason_loop = AdaptiveRereasonLoop(base_generate_fn, extract_answer_fn)
    critique_loop = CritiqueDebateLoop(base_generate_fn, extract_answer_fn, stage_generate_fn=stage_generate_fn)

    write_header = not os.path.exists(out_csv_path)
    with open(out_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIRED_CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for i, sample in enumerate(dataset):
            t0 = time.time()
            question, gold = sample["question"], sample["gold"]

            print(f"\\n[{i + 1}/{len(dataset)}] Starting question...", flush=True)
            print(f"[{i + 1}/{len(dataset)}] Question: {question[:200]}", flush=True)

            base_prompt = f"Question: {question}\nAnswer with only yes or no."

            print(f"[{i + 1}/{len(dataset)}] Generating answer...", flush=True)
            initial_reasoning = base_generate_fn(base_prompt)
            print(f"[{i + 1}/{len(dataset)}] Generation complete.", flush=True)
            baseline_answer = extract_answer_fn(initial_reasoning)
            baseline_correct = judge_correctness_fn(question, gold, baseline_answer)
            baseline_tokens = total_pipeline_tokens(token_counter, base_prompt, initial_reasoning)

            # Trigger evaluated exactly ONCE per question -- this is the shared
            # decision both arms below are conditioned on.
            answer_only_fn = lambda q: extract_answer_fn(base_generate_fn(
                f"Question: {q}\nAnswer with only yes or no."
            ))
            result = trigger.is_uncertain(question, answer_only_fn)
            trigger_fired = result.is_uncertain
            trigger_tokens = total_pipeline_tokens(token_counter, *result.samples)

            row = {
                "sample_id": i, "dataset": dataset_name, "question": question,
                "gold_answer": gold,
                "baseline_answer": baseline_answer, "baseline_correct": baseline_correct,
                "baseline_tokens": baseline_tokens,
                "trigger_fired": trigger_fired, "trigger_signal": result.signal_name,
                "trigger_raw_value": result.raw_value, "trigger_tokens": trigger_tokens,
                "rereason_answer": "", "rereason_correct": "", "rereason_tokens": 0,
                "critique_answer": "", "critique_correct": "", "critique_tokens": 0,
                "wall_seconds": 0,
            }

            if trigger_fired:
                # Both arms applied to the SAME initial_reasoning + trigger
                # outcome -- this is what makes the comparison paired.
                rereason_trace = rereason_loop.run(question, initial_reasoning, is_uncertain=True, sampled_answers=result.samples)
                rereason_correct = judge_correctness_fn(question, gold, rereason_trace.final_answer)
                rereason_tokens = total_pipeline_tokens(token_counter, rereason_trace.revised_reasoning)

                critique_trace = critique_loop.run(question, initial_reasoning, is_uncertain=True)
                critique_correct = judge_correctness_fn(question, gold, critique_trace.final_answer)
                critique_tokens = total_pipeline_tokens(
                    token_counter, critique_trace.critique, critique_trace.revised_reasoning
                )

                row.update({
                    "rereason_answer": rereason_trace.final_answer, "rereason_correct": rereason_correct,
                    "rereason_tokens": rereason_tokens,
                    "critique_answer": critique_trace.final_answer, "critique_correct": critique_correct,
                    "critique_tokens": critique_tokens,
                })

            row["wall_seconds"] = round(time.time() - t0, 3)
            writer.writerow(row)


def run_condition(
    dataset: Iterable[dict],
    dataset_name: str,
    mode: str,
    base_generate_fn: Callable[[str], str],
    extract_answer_fn: Callable[[str], str],
    judge_correctness_fn: Callable[[str, str, str], bool],
    token_counter: TokenCounter,
    trigger_type: str,
    out_csv_path: str,
    stage_generate_fn: Callable[[str], str] | None = None,
):
    """
    stage_generate_fn: passed through to CritiqueDebateLoop for the critique
        and revise stages specifically. Construct this with roughly HALF the
        max_new_tokens of base_generate_fn's budget (see model_backend.py's
        make_base_generate_fn max_new_tokens param) so the nominal token
        ceiling for the critique arm matches the re-reasoning arm's single
        pass, rather than silently doubling it. See critique_loop.py docstring.
    """
    assert mode in {"baseline", "always_on", "adaptive_rereason", "adaptive_debate"}

    trigger = build_trigger(trigger_type) if mode.startswith("adaptive") else None
    if mode in ("adaptive_debate", "always_on"):
        loop = CritiqueDebateLoop(base_generate_fn, extract_answer_fn, stage_generate_fn=stage_generate_fn)
    elif mode == "adaptive_rereason":
        loop = AdaptiveRereasonLoop(base_generate_fn, extract_answer_fn)
    else:
        loop = None  # baseline: no escalation loop needed

    write_header = not os.path.exists(out_csv_path)
    with open(out_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for i, sample in enumerate(dataset):
            t0 = time.time()
            question, gold = sample["question"], sample["gold"]
            base_prompt = f"Question: {question}\nAnswer with only yes or no."

            initial_reasoning = base_generate_fn(base_prompt)

            trigger_fired, trigger_signal, trigger_raw = False, "none", 0.0
            tokens_used = [base_prompt, initial_reasoning]

            if mode == "baseline":
                final_answer = extract_answer_fn(initial_reasoning)

            elif mode == "always_on":
                trace = loop.run(question, initial_reasoning, is_uncertain=True)
                final_answer = trace.final_answer
                trigger_fired, trigger_signal = True, "always_on"
                tokens_used += [trace.critique, trace.revised_reasoning]

            else:  # adaptive_rereason or adaptive_debate
                # Uncertainty check re-uses base_generate_fn to draw extra
                # independent samples -- these count toward token cost.
                answer_only_fn = lambda q: extract_answer_fn(base_generate_fn(
                    f"Question: {q}\nAnswer with only yes or no."
                ))
                result = trigger.is_uncertain(question, answer_only_fn)
                trigger_fired = result.is_uncertain
                trigger_signal = result.signal_name
                trigger_raw = result.raw_value
                tokens_used += result.samples  # rough accounting; refine if you log full trigger prompts/outputs

                trace = loop.run(question, initial_reasoning, is_uncertain=trigger_fired,
                                  trigger_info={"signal": trigger_signal, "value": trigger_raw}, sampled_answers=result.samples)
                final_answer = trace.final_answer
                if trigger_fired:
                    tokens_used += [trace.critique, trace.revised_reasoning]

            print(f"[{i + 1}/{len(dataset)}] Answer extracted: {final_answer[:200]}", flush=True)

            correct = judge_correctness_fn(question, gold, final_answer)
            n_tokens = total_pipeline_tokens(token_counter, *tokens_used)

            writer.writerow({
                "sample_id": i,
                "dataset": dataset_name,
                "mode": mode,
                "question": question,
                "gold_answer": gold,
                "final_answer": final_answer,
                "correct": correct,
                "trigger_fired": trigger_fired,
                "trigger_signal": trigger_signal,
                "trigger_raw_value": trigger_raw,
                "n_tokens": n_tokens,
                "wall_seconds": round(time.time() - t0, 3),
            })

            print(
                f"[{i + 1}/{len(dataset)}] DONE "
                f"({round(time.time() - t0, 2)}s)",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                         choices=["baseline", "always_on", "adaptive_rereason", "adaptive_debate", "adaptive_paired"],
                         help="adaptive_paired is the mode that actually answers RQ3 -- "
                              "runs the trigger once and applies both escalation arms to "
                              "the same triggered questions. Use this, not separate "
                              "adaptive_rereason/adaptive_debate runs, for the RQ3 comparison.")
    parser.add_argument("--dataset", required=True, choices=["strategyqa", "hotpotqa"])
    parser.add_argument("--trigger_type", default="self_consistency",
                         choices=["self_consistency", "entropy"])
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B",
                         help="base model: answers every condition")
    parser.add_argument("--judge_model_name", default="Qwen/Qwen2.5-32B-Instruct",
                         help="judge model: scores correctness only, never generates/revises")
    parser.add_argument("--out_csv", default="results.csv")
    parser.add_argument("--skip_judge", action="store_true",
                         help="generate only; run judge_pass_from_csv separately later "
                              "if VRAM can't hold both models at once (recommended on "
                              "40GB cards -- see README two-process workflow)")
    parser.add_argument("--judge_bf16", action="store_true",
                         help="load judge in full bf16 instead of 4-bit. Only use this "
                              "if you've confirmed your card has ~65GB+ free for the judge "
                              "alone; default is 4-bit since that's what fits on 40GB cards.")
    args = parser.parse_args()

    from pipeline.dataset_loaders import load_strategyqa, load_hotpotqa

    if args.dataset == "strategyqa":
        dataset = list(load_strategyqa(n=args.n_samples))
    else:
        dataset = list(load_hotpotqa(n=args.n_samples))

    from pipeline.model_backend import (
        load_base_model, make_base_generate_fn, make_extract_answer_fn,
        load_judge_model, make_judge_correctness_fn,
    )

    base_loaded = load_base_model(args.model_name)
    base_generate_fn = make_base_generate_fn(base_loaded, max_new_tokens=768)
    # Half the single-pass budget for critique/revise stages, so the critique
    # arm's nominal token ceiling (256 + 256) matches the re-reasoning arm's
    # single pass (512), rather than silently doubling it. See critique_loop.py.
    stage_generate_fn = make_base_generate_fn(base_loaded, max_new_tokens=256)
    extract_answer_fn = make_extract_answer_fn(dataset=args.dataset)
    token_counter = TokenCounter(args.model_name)

    if args.skip_judge:
        # Two-pass mode: score correctness later with judge_pass_from_csv().
        judge_correctness_fn = lambda q, g, p: None
    elif args.dataset == "strategyqa":
        # StrategyQA is binary yes/no: use exact match instead of the 32B judge.
        judge_correctness_fn = lambda q, g, p: (
            p.strip().lower() == g.strip().lower()
        )
    else:
        judge_loaded = load_judge_model(
            args.judge_model_name,
            use_4bit=not args.judge_bf16
        )
        judge_correctness_fn = make_judge_correctness_fn(judge_loaded)

    if args.mode == "adaptive_paired":
        run_paired_condition(
            dataset=dataset,
            dataset_name=args.dataset,
            base_generate_fn=base_generate_fn,
            extract_answer_fn=extract_answer_fn,
            judge_correctness_fn=judge_correctness_fn,
            token_counter=token_counter,
            trigger_type=args.trigger_type,
            out_csv_path=args.out_csv,
            stage_generate_fn=stage_generate_fn,
        )
        return

    run_condition(
        dataset=dataset,
        dataset_name=args.dataset,
        mode=args.mode,
        base_generate_fn=base_generate_fn,
        extract_answer_fn=extract_answer_fn,
        judge_correctness_fn=judge_correctness_fn,
        token_counter=token_counter,
        trigger_type=args.trigger_type,
        out_csv_path=args.out_csv,
        stage_generate_fn=stage_generate_fn,
    )


if __name__ == "__main__":
    main()
