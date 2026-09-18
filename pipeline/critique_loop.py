from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

GenerateFn = Callable[[str], str]


CRITIQUE_PROMPT_TEMPLATE = """You are reviewing another model's reasoning for correctness.

Question: {question}

Reasoning and answer to review:
{reasoning}

Identify the SPECIFIC step (quote or number it) where the reasoning is most likely to be
wrong, unsupported, or a non-sequitur. If you believe every step is fully correct, say
"NO ISSUE FOUND" and briefly say why you're confident. Do not just say "this looks wrong" --
point to the exact step and explain the specific error.

Critique:"""

REVISE_PROMPT_TEMPLATE = """You previously answered a question. Another reviewer critiqued your reasoning.
Revise your answer, taking the critique into account. If the critique doesn't identify a real
problem, you may keep your original answer -- but say so explicitly.

Question: {question}

Your original reasoning and answer:
{reasoning}

Reviewer's critique:
{critique}

Provide your revised reasoning and final answer:"""


@dataclass
class DebateTrace:
    question: str
    initial_reasoning: str
    critique: str
    revised_reasoning: str
    final_answer: str
    was_escalated: bool
    trigger_info: dict = field(default_factory=dict)


class CritiqueDebateLoop:
    def __init__(self, generate_fn: GenerateFn, extract_answer_fn: Callable[[str], str],
                 stage_generate_fn: GenerateFn | None = None):
        self.generate_fn = generate_fn
        self.stage_generate_fn = stage_generate_fn or generate_fn
        self.extract_answer_fn = extract_answer_fn
        if stage_generate_fn is None:
            import warnings
            warnings.warn(
                "CritiqueDebateLoop created without stage_generate_fn: critique "
                "and revision will use the same max_new_tokens as a full "
                "generation each, giving this arm roughly 2x the re-reasoning "
                "arm's nominal token ceiling. Pass a budget-halved "
                "stage_generate_fn to fix this before running real comparisons.",
                stacklevel=2,
            )

    def run(self, question: str, initial_reasoning: str, is_uncertain: bool,
            trigger_info: dict | None = None) -> DebateTrace:
        if not is_uncertain:
            return DebateTrace(
                question=question,
                initial_reasoning=initial_reasoning,
                critique="",
                revised_reasoning="",
                final_answer=self.extract_answer_fn(initial_reasoning),
                was_escalated=False,
                trigger_info=trigger_info or {},
            )

        critique_prompt = CRITIQUE_PROMPT_TEMPLATE.format(
            question=question, reasoning=initial_reasoning
        )
        critique = self.stage_generate_fn(critique_prompt)

        revise_prompt = REVISE_PROMPT_TEMPLATE.format(
            question=question, reasoning=initial_reasoning, critique=critique
        )
        revised_reasoning = self.stage_generate_fn(revise_prompt)

        return DebateTrace(
            question=question,
            initial_reasoning=initial_reasoning,
            critique=critique,
            revised_reasoning=revised_reasoning,
            final_answer=self.extract_answer_fn(revised_reasoning),
            was_escalated=True,
            trigger_info=trigger_info or {},
        )


class AdaptiveRereasonLoop:
    
    def __init__(self, generate_fn: GenerateFn, extract_answer_fn: Callable[[str], str]):
        self.generate_fn = generate_fn
        self.extract_answer_fn = extract_answer_fn

    def run(self, question: str, initial_reasoning: str, is_uncertain: bool,
            trigger_info: dict | None = None) -> DebateTrace:
        if not is_uncertain:
            return DebateTrace(
                question=question,
                initial_reasoning=initial_reasoning,
                critique="",
                revised_reasoning="",
                final_answer=self.extract_answer_fn(initial_reasoning),
                was_escalated=False,
                trigger_info=trigger_info or {},
            )

        second_pass_prompt = f"Question: {question}\nThink through this again from scratch, step by step, then give a final answer."
        second_reasoning = self.generate_fn(second_pass_prompt)

        return DebateTrace(
            question=question,
            initial_reasoning=initial_reasoning,
            critique="(no critique -- blind re-reasoning baseline)",
            revised_reasoning=second_reasoning,
            final_answer=self.extract_answer_fn(second_reasoning),
            was_escalated=True,
            trigger_info=trigger_info or {},
        )
