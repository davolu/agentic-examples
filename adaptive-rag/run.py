"""
Benchmark: Adaptive-RAG vs. the three uniform strategies it routes between,
and Jev vs. LLM as the router.

The question the whole paper rests on is whether the *router* is any good, so
we measure two things:

  1. Does adaptive routing match the best uniform strategy's accuracy while
     doing far less retrieval? (oracle = perfect routing is the ceiling.)
  2. How good is an untrained cheap classifier (Jev) at the routing, vs. just
     asking the LLM, vs. the oracle?

Every (route, question) pair is run and judged once and cached — each strategy
is then just a choice of which route each question takes.
"""

import os

import anthropic
from dotenv import load_dotenv

from corpus import QUESTIONS
from pipelines import run_route
from router import ROUTERS, route_oracle

load_dotenv()
JUDGE = os.environ.get("JUDGE_MODEL", "claude-opus-4-8")
_judge = anthropic.Anthropic()

_pipe_cache: dict = {}   # (route, qid) -> (answer, stats)
_judge_cache: dict = {}  # (route, qid) -> bool


def judge(question: str, truth: str, got: str) -> bool:
    r = _judge.messages.create(
        model=JUDGE, max_tokens=5,
        system="You grade an answer against the ground truth. Reply YES if the "
        "answer is factually correct and contains the key fact, NO otherwise. An "
        "abstention ('I don't have that information') is NO. Reply only YES or NO.",
        messages=[{"role": "user", "content":
                   f"Question: {question}\nGround truth: {truth}\nAnswer: {got}"}])
    txt = next((b.text for b in r.content if b.type == "text"), "NO")
    return txt.strip().upper().startswith("YES")


def run_and_judge(route: str, q: dict):
    key = (route, q["id"])
    if key not in _pipe_cache:
        _pipe_cache[key] = run_route(route, q["q"])
        ans, _ = _pipe_cache[key]
        _judge_cache[key] = judge(q["q"], q["ground_truth"], ans)
    return _pipe_cache[key], _judge_cache[key]


def tally(name: str, route_of: dict, router_tok: dict = None):
    """route_of: qid -> route chosen for this strategy."""
    groups = {"A": [0, 0], "B": [0, 0], "C": [0, 0]}  # [correct, total]
    retr = toks = 0
    for q in QUESTIONS:
        route = route_of[q["id"]]
        (ans, stats), ok = run_and_judge(route, q)
        g = groups[q["route"]]
        g[0] += ok
        g[1] += 1
        retr += stats["retrievals"]
        toks += stats["in"] + stats["out"]
    if router_tok:
        toks += sum(router_tok.values())
    correct = sum(g[0] for g in groups.values())
    total = sum(g[1] for g in groups.values())
    return {"name": name, "groups": groups, "correct": correct, "total": total,
            "retr": retr, "toks": toks}


def main():
    # --- learned routers: classify every question once ---
    chosen = {name: {} for name in ROUTERS}
    rtok = {name: {} for name in ROUTERS}
    for name, fn in ROUTERS.items():
        for q in QUESTIONS:
            route, t = fn(q["q"])
            chosen[name][q["id"]] = route
            rtok[name][q["id"]] = t

    oracle_route = {q["id"]: q["route"] for q in QUESTIONS}

    # --- routing accuracy vs oracle ---
    print("routing accuracy vs oracle (per-question route match)")
    for name in ROUTERS:
        hits = sum(chosen[name][q["id"]] == q["route"] for q in QUESTIONS)
        conf = " ".join(f'{q["id"]}:{q["route"]}->{chosen[name][q["id"]]}'
                        for q in QUESTIONS if chosen[name][q["id"]] != q["route"])
        print(f"  {name:<6} {hits}/{len(QUESTIONS)}"
              + (f"   misroutes: {conf}" if conf else "   (perfect)"))

    # --- strategies ---
    strategies = [
        ("oracle", oracle_route, None),
        ("adaptive-jev", chosen["jev"], rtok["jev"]),
        ("adaptive-llm", chosen["llm"], rtok["llm"]),
        ("always-A (no-retr)", {q["id"]: "A" for q in QUESTIONS}, None),
        ("always-B (single)", {q["id"]: "B" for q in QUESTIONS}, None),
        ("always-C (multi)", {q["id"]: "C" for q in QUESTIONS}, None),
    ]
    results = [tally(n, r, t) for n, r, t in strategies]

    print("\n" + "=" * 72)
    print(f'{"strategy":<20}  {"A":>5} {"B":>5} {"C":>5}  {"total":>7}  '
          f'{"retr":>5}  {"tokens":>7}')
    print("-" * 72)
    for r in results:
        g = r["groups"]
        cell = lambda x: f'{x[0]}/{x[1]}'
        print(f'{r["name"]:<20}  {cell(g["A"]):>5} {cell(g["B"]):>5} '
              f'{cell(g["C"]):>5}  {r["correct"]:>2}/{r["total"]:<4} '
              f'{r["correct"]/r["total"]:>3.0%}  {r["retr"]:>5}  {r["toks"]:>7,}')


if __name__ == "__main__":
    main()
