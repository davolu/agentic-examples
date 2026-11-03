"""
Benchmark: single pass vs multi-agent debate on the reasoning-trap set.

    python run.py           # default debaters: claude-haiku-4-5, 3 agents, 2 rounds
    DEBATE_MODEL=claude-sonnet-5 python run.py

Reports accuracy and cost for both, plus the per-question flip (where debate
rescued a wrong single answer — and where it made one worse).
"""

import json
import time
from pathlib import Path

from debate import MODEL, N_AGENTS, N_ROUNDS, debate, single_answer
from questions import QUESTIONS, check

# haiku-4-5 pricing $/token (input, output)
PRICE = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-5": (3.0, 15.0),
         "claude-opus-4-8": (5.0, 25.0)}
P_IN, P_OUT = [p / 1e6 for p in PRICE.get(MODEL, (1.0, 5.0))]


def cost(u): return u["in"] * P_IN + u["out"] * P_OUT


if __name__ == "__main__":
    print(f"Debaters: {MODEL} · {N_AGENTS} agents · {N_ROUNDS} rounds · "
          f"{len(QUESTIONS)} questions\n")
    rows = []
    s_cost = d_cost = 0.0
    t0 = time.perf_counter()
    for q in QUESTIONS:
        s_ans, s_u = single_answer(q["q"])
        d_ans, d_u = debate(q["q"])
        s_ok, d_ok = check(s_ans, q), check(d_ans, q)
        s_cost += cost(s_u); d_cost += cost(d_u)
        flip = ("✅ rescued" if d_ok and not s_ok else
                "❌ broke" if s_ok and not d_ok else
                "  both ok" if s_ok else "  both wrong")
        print(f"  {q['id']:6} single={'✓' if s_ok else '✗'} "
              f"debate={'✓' if d_ok else '✗'}  {flip}"
              f"   (single:{s_ans[:18]!r} debate:{d_ans[:18]!r})")
        rows.append({"id": q["id"], "single_ok": s_ok, "debate_ok": d_ok,
                     "single": s_ans, "debate": d_ans})
    wall = time.perf_counter() - t0

    n = len(QUESTIONS)
    s_acc = sum(r["single_ok"] for r in rows) / n
    d_acc = sum(r["debate_ok"] for r in rows) / n
    rescued = sum(1 for r in rows if r["debate_ok"] and not r["single_ok"])
    broke = sum(1 for r in rows if r["single_ok"] and not r["debate_ok"])

    print(f"\n{'='*56}")
    print(f"  Single pass:  {s_acc:.0%}  ·  ${s_cost:.4f}")
    print(f"  Debate:       {d_acc:.0%}  ·  ${d_cost:.4f}  "
          f"({d_cost/s_cost:.1f}x cost)")
    print(f"  Net: {d_acc-s_acc:+.0%}  (rescued {rescued}, broke {broke})")
    print(f"  Wall clock: {wall:.0f}s")

    Path("results.json").write_text(json.dumps(
        {"model": MODEL, "agents": N_AGENTS, "rounds": N_ROUNDS,
         "single_accuracy": s_acc, "debate_accuracy": d_acc,
         "single_cost": round(s_cost, 4), "debate_cost": round(d_cost, 4),
         "rescued": rescued, "broke": broke, "rows": rows}, indent=1))
    print("  Wrote results.json")
