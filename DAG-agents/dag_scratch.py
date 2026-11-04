"""
A DAG scheduler from scratch — ~50 lines, nothing hidden.

The point: once a workflow is a declared dependency graph, three things fall
out of the *structure* automatically, with no orchestration logic:

  PARALLELISM  — group nodes into topological "levels"; every node in a level
                 has its dependencies already satisfied, so the whole level
                 runs concurrently (asyncio.gather).
  TERMINATION  — if we ever reach a point where no node's dependencies are
                 satisfiable, the graph has a cycle and we refuse it. An
                 acyclic graph is guaranteed to finish.
  ORDER        — the levels ARE the correct execution order; we never hand-wrote
                 "run parse first, then..." — the graph said so.

    python dag_scratch.py            # run the résumé review (parallel)
    python dag_scratch.py --cycle    # show cycle detection reject a bad graph
"""

import asyncio
import sys
import time

from nodes import GRAPH


def topological_levels(graph: dict) -> list[list[str]]:
    """Group nodes into dependency levels. Raises on a cycle (not a DAG)."""
    deps = {name: set(d) for name, (_, d) in graph.items()}
    done, levels = set(), []
    while len(done) < len(graph):
        # A level = every not-yet-done node whose deps are all satisfied.
        level = [n for n in graph if n not in done and deps[n] <= done]
        if not level:
            stuck = [n for n in graph if n not in done]
            raise ValueError(f"Cycle detected — not a DAG. Stuck on: {stuck}")
        levels.append(sorted(level))
        done |= set(level)
    return levels


async def _run_node(graph, name, results):
    func, dep_names = graph[name]
    inputs = {d: results[d] for d in dep_names}
    return name, await func(inputs)


async def run_parallel(graph: dict) -> dict:
    """Run each topological level concurrently."""
    results = {}
    for level in topological_levels(graph):
        done = await asyncio.gather(*(_run_node(graph, n, results) for n in level))
        results.update(dict(done))
    return results


async def run_sequential(graph: dict) -> dict:
    """Run nodes one at a time in topological order (baseline)."""
    results = {}
    for level in topological_levels(graph):
        for n in level:
            _, out = await _run_node(graph, n, results)
            results[n] = out
    return results


async def main():
    if "--cycle" in sys.argv:
        print("Adding a cycle (synth → parse) and asking the scheduler to run it:\n")
        bad = dict(GRAPH)
        bad["parse"] = (GRAPH["parse"][0], ["synth"])   # parse now needs synth → cycle
        try:
            topological_levels(bad)
        except ValueError as e:
            print(f"  ✅ Refused: {e}")
            print("  (A cyclic graph can never terminate — the structure catches it.)")
        return

    levels = topological_levels(GRAPH)
    print("DAG levels (each level runs in parallel):")
    for i, lvl in enumerate(levels):
        print(f"  level {i}: {lvl}")

    print("\n⏱  Sequential (one node at a time)...")
    t = time.perf_counter()
    await run_sequential(GRAPH)
    seq = time.perf_counter() - t
    print(f"   {seq:.1f}s")

    print("⏱  DAG-parallel (levels run concurrently)...")
    t = time.perf_counter()
    results = await run_parallel(GRAPH)
    par = time.perf_counter() - t
    print(f"   {par:.1f}s")

    print(f"\n⚡ Speedup: {seq/par:.1f}x  "
          f"(the 3 analyses ran at once instead of back-to-back)")
    print("\n" + "=" * 60 + "\nRÉSUMÉ REVIEW\n" + "=" * 60)
    print(results["synth"][:1000])


if __name__ == "__main__":
    asyncio.run(main())
