"""
Benchmark: vector RAG vs GraphRAG, on a GLOBAL question and a LOCAL one.

Global question ("what are the main recurring problems?") — the answer is spread
across the whole corpus. We score THEME COVERAGE: of the 7 seeded recurring
themes, how many does each approach surface? An Opus judge decides, per theme,
whether the answer substantively covers it.

Local question ("what about data export?") — the answer sits in a few tickets.
Both approaches should do fine; the point is that GraphRAG does NOT win here, and
costs more, so its advantage is scoped to global sensemaking.

We also report GraphRAG's one-time INDEXING cost — the price of the graph.
"""

import os

import anthropic
from dotenv import load_dotenv

from corpus import QUESTIONS, THEMES, TICKETS
from graph_index import build
from graph_query import global_search, local_search
from vector_rag import answer as vector_answer

load_dotenv()
JUDGE = os.environ.get("JUDGE_MODEL", "claude-opus-4-8")
CHAT = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
_judge = anthropic.Anthropic()


def stuff_all(question: str) -> tuple[str, dict]:
    """Full-context baseline: put the ENTIRE corpus in the prompt. Only possible
    because this corpus is tiny — the honest point of comparison."""
    ctx = "\n\n".join(f"[{t}] {d['text']}" for t, d in TICKETS.items())
    r = _judge.messages.create(
        model=CHAT, max_tokens=900,
        system="Answer using ONLY the provided tickets. Be specific and concise.",
        messages=[{"role": "user", "content":
                   f"<tickets>\n{ctx}\n</tickets>\n\nQuestion: {question}"}])
    text = next((b.text for b in r.content if b.type == "text"), "")
    return text, {"in": r.usage.input_tokens, "out": r.usage.output_tokens, "calls": 1}


def covered_themes(answer: str) -> set:
    """Opus decides which seeded themes the answer substantively covers."""
    themes = "\n".join(f"{k}: {v}" for k, v in THEMES.items())
    r = _judge.messages.create(
        model=JUDGE, max_tokens=200,
        system="You check which themes an answer covers. For EACH theme, output "
        "a line 'key: YES' if the answer substantively raises that problem, else "
        "'key: NO'. Output only those lines.",
        messages=[{"role": "user", "content":
                   f"THEMES:\n{themes}\n\nANSWER:\n{answer}"}])
    txt = next((b.text for b in r.content if b.type == "text"), "")
    out = set()
    for line in txt.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            if k.strip() in THEMES and v.strip().upper().startswith("YES"):
                out.add(k.strip())
    return out


def tok(c):
    return c["in"] + c["out"]


def main():
    index = build()
    s = index["stats"]
    ic = index["cost"]
    print(f'INDEX: {s["entities"]} entities, {s["edges"]} edges, '
          f'{s["communities"]} communities')
    print(f'  one-time indexing cost: {ic["calls"]} LLM calls, '
          f'{tok(ic):,} tokens\n')

    gq = next(q for q in QUESTIONS if q["type"] == "global")
    lq = next(q for q in QUESTIONS if q["type"] == "local")
    n_themes = len(THEMES)

    # ---- GLOBAL: theme coverage ----
    v_ans, v_cost = vector_answer(gq["q"])
    g_ans, g_cost = global_search(gq["q"], index)
    s_ans, s_cost = stuff_all(gq["q"])
    v_cov, g_cov, s_cov = covered_themes(v_ans), covered_themes(g_ans), covered_themes(s_ans)

    print("GLOBAL question — theme coverage (of 7 recurring themes)")
    print(f'  vector RAG (k={v_cost["k"]})       {len(v_cov)}/{n_themes}  '
          f'{{{", ".join(sorted(v_cov))}}}   {v_cost["calls"]} call / {tok(v_cost):,} tok')
    print(f'  GraphRAG (global)      {len(g_cov)}/{n_themes}  '
          f'{{{", ".join(sorted(g_cov))}}}   {g_cost["calls"]} calls / {tok(g_cost):,} tok '
          f'(kept {g_cost["kept"]}/{g_cost["communities"]})')
    print(f'  stuff-all (full ctx)   {len(s_cov)}/{n_themes}  '
          f'{{{", ".join(sorted(s_cov))}}}   {s_cost["calls"]} call / {tok(s_cost):,} tok')
    print(f'  themes vector RAG missed: {{{", ".join(sorted(set(THEMES) - v_cov))}}}\n')

    # ---- LOCAL: both should answer; GraphRAG shouldn't win, and costs more ----
    lv_ans, lv_cost = vector_answer(lq["q"])
    lg_ans, lg_cost = local_search(lq["q"], index)
    lv_ok = covers_truth(lq["q"], lq["truth"], lv_ans)
    lg_ok = covers_truth(lq["q"], lq["truth"], lg_ans)
    print("LOCAL question — correct vs ground truth")
    print(f'  vector RAG   {"OK" if lv_ok else "miss"}   '
          f'{lv_cost["calls"]} calls / {tok(lv_cost):,} tok')
    print(f'  GraphRAG     {"OK" if lg_ok else "miss"}   '
          f'{lg_cost["calls"]} calls / {tok(lg_cost):,} tok')


def covers_truth(question, truth, got) -> bool:
    r = _judge.messages.create(
        model=JUDGE, max_tokens=5,
        system="Reply YES if the answer is correct and contains the key facts of "
        "the ground truth, else NO. Reply only YES or NO.",
        messages=[{"role": "user", "content":
                   f"Q: {question}\nGround truth: {truth}\nAnswer: {got}"}])
    txt = next((b.text for b in r.content if b.type == "text"), "NO")
    return txt.strip().upper().startswith("YES")


if __name__ == "__main__":
    main()
