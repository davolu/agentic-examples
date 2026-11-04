"""
The same résumé-review DAG, with LangGraph.

Everything we hand-rolled in dag_scratch.py — topological ordering, running
independent nodes in parallel, fan-in waiting for all branches — LangGraph
does for you. You declare the edges; LangGraph executes nodes in "super-steps",
running everything in a step concurrently and making a fan-in node wait for all
its upstream branches.

The one wrinkle vs. the scratch version: three parallel nodes all write to the
state at once, so the field they write to needs a REDUCER (how to merge
concurrent writes). That's the `Annotated[dict, merge]` below.

Note: this uses the REAL LangGraph API (StateGraph) — not the `GraphLangChain`
from the canonical blog screenshot, which isn't a real class.

    python dag_langgraph.py
"""

import asyncio
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from nodes import (check_ats, market_compare, parse_resume, skill_gaps,
                   synthesize)


def merge(a: dict, b: dict) -> dict:
    """Reducer: merge concurrent writes from the parallel analysis nodes."""
    return {**a, **b}


class State(TypedDict):
    parsed: str
    analyses: Annotated[dict, merge]   # ats/skills/market write here in parallel
    review: str


# Node wrappers adapt our shared async nodes to LangGraph's state in/out shape.
async def n_parse(s: State):
    return {"parsed": await parse_resume({})}

async def n_ats(s: State):
    return {"analyses": {"ats": await check_ats({"parse": s["parsed"]})}}

async def n_skills(s: State):
    return {"analyses": {"skills": await skill_gaps({"parse": s["parsed"]})}}

async def n_market(s: State):
    return {"analyses": {"market": await market_compare({"parse": s["parsed"]})}}

async def n_synth(s: State):
    a = s["analyses"]
    return {"review": await synthesize(
        {"ats": a["ats"], "skills": a["skills"], "market": a["market"]})}


def build():
    g = StateGraph(State)
    for name, fn in [("parse", n_parse), ("ats", n_ats), ("skills", n_skills),
                     ("market", n_market), ("synth", n_synth)]:
        g.add_node(name, fn)
    g.add_edge(START, "parse")
    # fan-out: parse → the three independent analyses (run in parallel)
    g.add_edge("parse", "ats")
    g.add_edge("parse", "skills")
    g.add_edge("parse", "market")
    # fan-in: synth waits for all three
    g.add_edge("ats", "synth")
    g.add_edge("skills", "synth")
    g.add_edge("market", "synth")
    g.add_edge("synth", END)
    return g.compile()


async def main():
    graph = build()
    print("LangGraph diamond: parse → [ats ∥ skills ∥ market] → synth\n")
    t = time.perf_counter()
    result = await graph.ainvoke({"parsed": "", "analyses": {}, "review": ""})
    elapsed = time.perf_counter() - t
    print(f"⏱  Ran in {elapsed:.1f}s (the 3 analyses ran concurrently in one super-step)")
    print("\n" + "=" * 60 + "\nRÉSUMÉ REVIEW\n" + "=" * 60)
    print(result["review"][:1000])


if __name__ == "__main__":
    asyncio.run(main())
