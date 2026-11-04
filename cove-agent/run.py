"""
Benchmark CoVe against a single baseline pass, and test the paper's central
claim that INDEPENDENT (factored) verification beats CONTAMINATED (joint).

An Opus judge grades each answer against the question's ground truth — stronger
than the Haiku answerer, and calibrated on obvious cases before we trust it.

    python run.py
"""

import asyncio
import json
import os
import time

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from cove import baseline, cove
from questions import QUESTIONS

load_dotenv()
import anthropic  # noqa: E402

JUDGE_MODEL = "claude-opus-4-8"
judge_client = anthropic.AsyncAnthropic()


class Verdict(BaseModel):
    correct: bool = Field(description="true if the answer is factually correct "
                          "and free of hallucinated/added wrong details")
    reason: str = Field(description="one short sentence")


async def judge(question: str, ground_truth: str, answer: str) -> bool:
    r = await judge_client.messages.parse(
        model=JUDGE_MODEL, max_tokens=300,
        system="You grade factual answers. Mark correct=true ONLY if the answer "
        "matches the ground truth AND adds no incorrect/hallucinated details "
        "(for list questions, every listed item must be correct and the set "
        "complete; for false-premise questions, the answer must reject the "
        "premise rather than confabulate).",
        messages=[{"role": "user", "content":
                   f"<question>{question}</question>\n"
                   f"<ground_truth>{ground_truth}</ground_truth>\n"
                   f"<answer>{answer}</answer>"}],
        output_format=Verdict)
    return r.parsed_output.correct


CALIBRATION = [
    ("What is the capital of Australia?", "Canberra.", "Canberra.", True),
    ("What is the capital of Australia?", "Canberra.", "Sydney.", False),
    ("Which states border Mexico?", "California, Arizona, New Mexico, Texas.",
     "California, Arizona, New Mexico, Texas, and Nevada.", False),  # added wrong
]


async def calibrate():
    oks = await asyncio.gather(*(judge(q, g, a) for q, g, a, _ in CALIBRATION))
    agree = sum(1 for ok, (_, _, _, exp) in zip(oks, CALIBRATION) if ok == exp)
    return agree, len(CALIBRATION)


async def run():
    agree, total = await calibrate()
    print(f"Judge calibration: {agree}/{total} "
          f"({'trustworthy' if agree == total else 'DO NOT TRUST'})\n")

    print(f"Running {len(QUESTIONS)} questions: baseline vs CoVe(factored) vs CoVe(joint)\n")
    rows = []
    first_catch = None
    t0 = time.perf_counter()
    for qd in QUESTIONS:
        q, gt = qd["q"], qd["ground_truth"]
        base = await baseline(q)
        cv_f = await cove(q, "factored")
        cv_j = await cove(q, "joint")
        b_ok, f_ok, j_ok = await asyncio.gather(
            judge(q, gt, base),
            judge(q, gt, cv_f["final"]),
            judge(q, gt, cv_j["final"]))
        rows.append({"id": qd["id"], "type": qd["type"],
                     "baseline_ok": b_ok, "factored_ok": f_ok, "joint_ok": j_ok})
        mark = lambda ok: "✓" if ok else "✗"
        print(f"  {qd['id']:4} {qd['type']:14} base={mark(b_ok)} "
              f"cove_factored={mark(f_ok)} cove_joint={mark(j_ok)}")
        if first_catch is None and f_ok and not b_ok:
            first_catch = {"q": q, "baseline": base, "final": cv_f["final"],
                           "checks": cv_f["verifications"]}
    wall = time.perf_counter() - t0

    n = len(rows)
    b_acc = sum(r["baseline_ok"] for r in rows) / n
    f_acc = sum(r["factored_ok"] for r in rows) / n
    j_acc = sum(r["joint_ok"] for r in rows) / n

    print(f"\n{'='*60}")
    print(f"  Baseline (single pass):   {b_acc:.0%}")
    print(f"  CoVe — joint (contaminated verification): {j_acc:.0%}")
    print(f"  CoVe — factored (independent verification): {f_acc:.0%}")
    print(f"\n  CoVe lift (factored vs baseline):  {f_acc - b_acc:+.0%}")
    print(f"  Independence matters (factored − joint): {f_acc - j_acc:+.0%}")
    print(f"  Wall clock: {wall:.0f}s")

    if first_catch:
        print(f"\n--- a hallucination CoVe caught ---")
        print(f"  Q:        {first_catch['q']}")
        print(f"  Baseline: {first_catch['baseline'][:160]}")
        print(f"  Verified: {first_catch['final'][:160]}")

    json.dump({"baseline": b_acc, "cove_factored": f_acc, "cove_joint": j_acc,
               "rows": rows}, open("results.json", "w"), indent=1)
    print("\n  Wrote results.json")


if __name__ == "__main__":
    asyncio.run(run())
