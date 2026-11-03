"""
Multi-agent debate (Du et al., 2023) vs a single pass.

single_answer: one agent, one shot — the baseline.

debate: N agents answer independently (round 0), then for R rounds each agent
is shown the OTHER agents' latest answers and reasoning and asked to reconsider
and either defend or revise. The final answer is a majority vote over the
agents' last answers. The mechanism is simple: seeing a peer reach a different
conclusion forces each agent to re-examine its own — which is exactly what
catches a confident first-instinct error.

Both paths return (final_answer, usage_tokens) so we can score accuracy AND
compare cost.
"""

import os
import re
from collections import Counter

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# Debaters run on a cheaper model by default so there's headroom to show
# whether debate improves accuracy. Override with DEBATE_MODEL.
MODEL = os.environ.get("DEBATE_MODEL", "claude-haiku-4-5")
N_AGENTS = int(os.environ.get("DEBATE_AGENTS", "3"))
N_ROUNDS = int(os.environ.get("DEBATE_ROUNDS", "2"))

client = anthropic.Anthropic()


class Answer(BaseModel):
    reasoning: str = Field(description="brief step-by-step reasoning")
    final_answer: str = Field(description="the final answer only, as concisely as possible")


SOLVE_SYSTEM = ("You are a careful problem solver. Think step by step, watch "
                "for traps and intuitive-but-wrong answers, then give your final "
                "answer as concisely as possible.")


def _ask(question: str, context: str, usage_acc: dict) -> Answer:
    msg = question if not context else (
        f"{question}\n\nHere is how other solvers answered. Consider their "
        f"reasoning carefully. If they caught something you missed, revise; if "
        f"you believe you are right, defend your answer.\n\n{context}\n\n"
        "Give your reconsidered final answer.")
    r = client.messages.parse(
        model=MODEL, max_tokens=800, system=SOLVE_SYSTEM,
        messages=[{"role": "user", "content": msg}], output_format=Answer)
    usage_acc["in"] += r.usage.input_tokens
    usage_acc["out"] += r.usage.output_tokens
    return r.parsed_output


def single_answer(question: str) -> tuple[str, dict]:
    usage = {"in": 0, "out": 0}
    ans = _ask(question, "", usage)
    return ans.final_answer, usage


def debate(question: str) -> tuple[str, dict]:
    usage = {"in": 0, "out": 0}
    # Round 0 — independent answers.
    answers = [_ask(question, "", usage) for _ in range(N_AGENTS)]

    # Rounds 1..R — each agent sees the others and reconsiders.
    for _ in range(N_ROUNDS):
        new_answers = []
        for i in range(N_AGENTS):
            others = "\n".join(
                f"- Solver {j+1} concluded: {a.final_answer}  (reasoning: {a.reasoning[:200]})"
                for j, a in enumerate(answers) if j != i)
            new_answers.append(_ask(question, others, usage))
        answers = new_answers

    # Majority vote over final answers (normalized loosely for grouping).
    def norm(s): return re.sub(r"[^a-z0-9.]", "", s.lower())
    votes = Counter(norm(a.final_answer) for a in answers)
    winner_norm, _ = votes.most_common(1)[0]
    final = next(a.final_answer for a in answers if norm(a.final_answer) == winner_norm)
    return final, usage
