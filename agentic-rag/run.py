"""
Benchmark naive RAG vs agentic RAG on the NovaTech knowledge base.

An Opus judge grades each answer against ground truth. Crucially, for
out-of-scope questions the ONLY correct answer is to abstain — so a confident
answer there is marked wrong (that's the hallucination we care about).

    python run.py
"""

import os
from collections import defaultdict

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import agentic_rag
import naive_rag
from corpus import QUESTIONS

load_dotenv()
JUDGE = "claude-opus-4-8"
HAIKU_IN, HAIKU_OUT = 1.0 / 1e6, 5.0 / 1e6
JEV_IN = 0.042 / 1e6
judge_client = anthropic.Anthropic()


class Verdict(BaseModel):
    correct: bool = Field(description="true if the answer matches the ground truth")
    reason: str = Field(description="one short sentence")


def judge(question, ground_truth, answer) -> bool:
    r = judge_client.messages.parse(
        model=JUDGE, max_tokens=250,
        system="Grade the answer against the ground truth. For out-of-scope "
        "questions the ground truth says to abstain — mark correct=true ONLY if "
        "the answer admits it doesn't know / isn't in the knowledge base, and "
        "correct=false if it confidently answers anyway (that's a hallucination).",
        messages=[{"role": "user", "content":
                   f"<question>{question}</question>\n"
                   f"<ground_truth>{ground_truth}</ground_truth>\n"
                   f"<answer>{answer}</answer>"}],
        output_format=Verdict)
    return r.parsed_output.correct


def cost(u):
    return (u.get("in", 0) * HAIKU_IN + u.get("out", 0) * HAIKU_OUT
            + u.get("jev", 0) * JEV_IN)


if __name__ == "__main__":
    print(f"Naive RAG vs Agentic RAG · {len(QUESTIONS)} questions · "
          f"grader={JUDGE}\n")
    by_type = defaultdict(lambda: {"n": 0, "naive": 0, "agentic": 0})
    n_cost = a_cost = 0.0
    flip = None

    for qd in QUESTIONS:
        q, gt, typ = qd["q"], qd["ground_truth"], qd["type"]
        n_ans, n_u = naive_rag.answer(q)
        a_ans, a_u = agentic_rag.answer(q)
        n_ok, a_ok = judge(q, gt, n_ans), judge(q, gt, a_ans)
        n_cost += cost(n_u); a_cost += cost(a_u)
        by_type[typ]["n"] += 1
        by_type[typ]["naive"] += n_ok
        by_type[typ]["agentic"] += a_ok
        m = lambda ok: "✓" if ok else "✗"
        tag = " (abstained)" if a_u.get("abstained") else ""
        print(f"  {qd['id']:4} {typ:14} naive={m(n_ok)} agentic={m(a_ok)}{tag}")
        if flip is None and a_ok and not n_ok and typ == "out_of_scope":
            flip = {"q": q, "naive": n_ans, "agentic": a_ans}

    n = len(QUESTIONS)
    n_acc = sum(by_type[t]["naive"] for t in by_type) / n
    a_acc = sum(by_type[t]["agentic"] for t in by_type) / n

    print(f"\n{'='*58}\n  By category          naive   agentic")
    for t, s in by_type.items():
        print(f"  {t:18} {s['naive']}/{s['n']}      {s['agentic']}/{s['n']}")
    print(f"{'='*58}")
    print(f"  Overall accuracy:  naive {n_acc:.0%}   agentic {a_acc:.0%}  ({a_acc-n_acc:+.0%})")
    print(f"  Cost:              naive ${n_cost:.4f}   agentic ${a_cost:.4f}  "
          f"({a_cost/n_cost:.1f}x)")

    if flip:
        print(f"\n--- the out-of-scope hallucinate→abstain flip ---")
        print(f"  Q:       {flip['q']}")
        print(f"  Naive:   {flip['naive'][:150]}")
        print(f"  Agentic: {flip['agentic'][:150]}")
