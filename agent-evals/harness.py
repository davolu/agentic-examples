"""
The eval harness — your agent's CI.

Runs the triage agent over the golden dataset, scores every case with all
three evaluators, aggregates overall + PER-CATEGORY metrics, tracks cost and
latency, and diffs against a saved baseline so a regression can't sneak in.

    python harness.py v1 --save-baseline     # record the baseline
    python harness.py v2                      # compare a change against it
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from agent import triage
from dataset import CASES, CATEGORIES
from evaluators import (JEV_LIVE, calibrate_judge, exact_match, jev_judge,
                        llm_judge)

BASELINE = Path(__file__).resolve().parent / "baseline.json"
LLM_IN, LLM_OUT = 5.00 / 1e6, 25.00 / 1e6   # claude-opus-4-8 $/token
JEV_IN = 0.042 / 1e6
REGRESSION_THRESHOLD = 0.05   # flag a per-metric drop larger than this


def run(version: str) -> dict:
    ledger = {"llm_in": 0, "llm_out": 0, "jev_in": 0}
    rows, per_cat = [], defaultdict(lambda: {"n": 0, "cat_ok": 0})

    print(f"\nRunning {len(CASES)} cases on prompt {version}...")
    t0 = time.perf_counter()
    for case in CASES:
        pred, usage = triage(case["email"], version)
        ledger["llm_in"] += usage.input_tokens
        ledger["llm_out"] += usage.output_tokens
        em = exact_match(pred, case)
        judged = llm_judge(case["email"], pred.reply, case["reply_gist"], ledger).acceptable
        jev_ok = jev_judge(case["email"], pred.reply, case["reply_gist"], ledger)
        rows.append({"id": case["id"], "category": case["category"],
                     "pred_category": pred.category, **em,
                     "reply_ok": judged, "jev_reply_ok": jev_ok})
        per_cat[case["category"]]["n"] += 1
        per_cat[case["category"]]["cat_ok"] += int(em["category_ok"])
    wall = time.perf_counter() - t0

    n = len(rows)
    metrics = {
        "version": version,
        "category_accuracy": sum(r["category_ok"] for r in rows) / n,
        "is_refund_accuracy": sum(r["is_refund_ok"] for r in rows) / n,
        "urgency_accuracy": sum(r["urgency_ok"] for r in rows) / n,
        "reply_acceptable_rate": sum(r["reply_ok"] for r in rows) / n,
        "per_category_accuracy": {
            c: round(per_cat[c]["cat_ok"] / per_cat[c]["n"], 3)
            for c in CATEGORIES if per_cat[c]["n"]},
        "cost_usd": round(ledger["llm_in"] * LLM_IN + ledger["llm_out"] * LLM_OUT
                          + ledger["jev_in"] * JEV_IN, 4),
        "wall_clock_s": round(wall, 1),
        "n": n,
    }
    if JEV_LIVE:
        metrics["jev_reply_agreement"] = round(
            sum(1 for r in rows if r["jev_reply_ok"] == r["reply_ok"]) / n, 3)
    return {"metrics": metrics, "rows": rows}


def print_metrics(m):
    print(f"\n  Category accuracy:     {m['category_accuracy']:.1%}")
    print(f"  is_refund accuracy:    {m['is_refund_accuracy']:.1%}")
    print(f"  Urgency accuracy (±1): {m['urgency_accuracy']:.1%}")
    print(f"  Reply acceptable rate: {m['reply_acceptable_rate']:.1%}  (LLM judge)")
    if "jev_reply_agreement" in m:
        print(f"  Jev↔LLM judge agreement: {m['jev_reply_agreement']:.1%}")
    print("  Per-category accuracy:")
    for c, acc in m["per_category_accuracy"].items():
        print(f"    {c:10} {acc:.0%}")
    print(f"  Cost: ${m['cost_usd']:.4f} · {m['wall_clock_s']}s")


def diff_baseline(m):
    if not BASELINE.exists():
        print("\n  (no baseline to compare against — run with --save-baseline)")
        return
    base = json.loads(BASELINE.read_text())["metrics"]
    print(f"\n  REGRESSION CHECK vs baseline ({base['version']}):")
    regressed = False
    for key in ("category_accuracy", "is_refund_accuracy", "urgency_accuracy",
                "reply_acceptable_rate"):
        delta = m[key] - base[key]
        flag = ""
        if delta < -REGRESSION_THRESHOLD:
            flag, regressed = "  🔴 REGRESSION", True
        elif delta > REGRESSION_THRESHOLD:
            flag = "  🟢 improved"
        print(f"    {key:22} {base[key]:.1%} → {m[key]:.1%}  ({delta:+.1%}){flag}")
    print("    per-category:")
    for c in m["per_category_accuracy"]:
        b = base["per_category_accuracy"].get(c)
        if b is None:
            continue
        delta = m["per_category_accuracy"][c] - b
        flag = ""
        if delta < -REGRESSION_THRESHOLD:
            flag, regressed = "  🔴 REGRESSION", True
        print(f"      {c:10} {b:.0%} → {m['per_category_accuracy'][c]:.0%}  ({delta:+.0%}){flag}")
    print(f"\n  {'🔴 REGRESSIONS DETECTED — do not ship' if regressed else '🟢 No regressions — safe to ship'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("version", nargs="?", default="v1", choices=["v1", "v2"])
    ap.add_argument("--save-baseline", action="store_true")
    args = ap.parse_args()

    cal = calibrate_judge()
    print(f"Judge calibration: {cal['agreement']}/{cal['total']} "
          f"({'trustworthy' if cal['trustworthy'] else 'DO NOT TRUST'})")

    result = run(args.version)
    print_metrics(result["metrics"])

    if args.save_baseline:
        BASELINE.write_text(json.dumps(result, indent=1))
        print(f"\n  Saved baseline ({args.version}) → baseline.json")
    else:
        diff_baseline(result["metrics"])
