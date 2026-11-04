"""
Benchmark: plain RAG vs Corrective RAG on the NovaTech corpus.

Both answer the same 10 questions. Five are in the corpus; five are real-world
facts the corpus knows nothing about. An Opus judge grades each answer against
the ground truth. The headline is the web_fact group: plain RAG can only see
the (irrelevant) retrieved docs, so it abstains or guesses; CRAG's evaluator
flags the retrieval as junk and falls back to the web, recovering the answer.
"""

import os

import anthropic
from dotenv import load_dotenv

from corpus import QUESTIONS
from crag import crag
from plain_rag import answer as plain_answer

load_dotenv()
JUDGE = os.environ.get("JUDGE_MODEL", "claude-opus-4-8")
_judge = anthropic.Anthropic()


def judge(question: str, truth: str, got: str) -> bool:
    r = _judge.messages.create(
        model=JUDGE, max_tokens=5,
        system="You grade an answer against the ground truth. Reply YES if the "
        "answer is factually correct and contains the key fact, NO otherwise. "
        "An abstention ('I don't have that information') is NO. Reply only YES or NO.",
        messages=[{"role": "user", "content":
                   f"Question: {question}\nGround truth: {truth}\nAnswer: {got}"}])
    txt = next((b.text for b in r.content if b.type == "text"), "NO")
    return txt.strip().upper().startswith("YES")


def main():
    rows = []
    for q in QUESTIONS:
        p_ans, p_use = plain_answer(q["q"])
        c = crag(q["q"])
        p_ok = judge(q["q"], q["ground_truth"], p_ans)
        c_ok = judge(q["q"], q["ground_truth"], c["answer"])
        rows.append({
            "id": q["id"], "type": q["type"], "tier": c["tier"],
            "plain_ok": p_ok, "crag_ok": c_ok,
            "p_tok": p_use["in"] + p_use["out"],
            "c_tok": c["usage"]["in"] + c["usage"]["out"]})
        mark = lambda b: "✓" if b else "✗"
        print(f'{q["id"]:>3} [{q["type"]:<9}] retrieval={c["tier"]:<9} '
              f'plain {mark(p_ok)}  crag {mark(c_ok)}')

    print("\n" + "=" * 64)
    for grp in ("in_corpus", "web_fact"):
        g = [r for r in rows if r["type"] == grp]
        p = sum(r["plain_ok"] for r in g)
        c = sum(r["crag_ok"] for r in g)
        print(f'{grp:<10} ({len(g)})   plain {p}/{len(g)}   CRAG {c}/{len(g)}')

    allp = sum(r["plain_ok"] for r in rows)
    allc = sum(r["crag_ok"] for r in rows)
    n = len(rows)
    print("-" * 64)
    print(f'{"overall":<10} ({n})  plain {allp}/{n} ({allp/n:.0%})   '
          f'CRAG {allc}/{n} ({allc/n:.0%})')

    ptok = sum(r["p_tok"] for r in rows)
    ctok = sum(r["c_tok"] for r in rows)
    print(f"\ntokens      plain {ptok:,}   CRAG {ctok:,}   "
          f"({ctok/ptok:.1f}x)")

    tiers = {}
    for r in rows:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    print("routing     " + "  ".join(f"{k}={v}" for k, v in sorted(tiers.items())))


if __name__ == "__main__":
    main()
