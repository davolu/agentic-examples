"""
The GraphRAG index — the expensive part, built once.

Pipeline (Edge et al., 2024):
  1. For each ticket, an LLM extracts entities (features/problems) + the
     relationships between them.
  2. Entity resolution: one LLM pass canonicalizes the raw entity names so the
     same concept ("invoices", "invoice clarity", "cryptic charges") collapses
     to one node. Without this, a 25-ticket graph shatters into singletons and
     the theme communities never form — entity resolution is what makes the
     graph cohere at small scale.
  3. Those become a knowledge graph (networkx): nodes = canonical entities,
     edges = intra-ticket co-occurrence, weighted by how often a pair appears
     together (relationships add extra weight).
  4. Community detection (Louvain) partitions the graph into clusters of
     closely-related entities — these end up tracking the complaint themes.
  5. An LLM writes a summary of each community from its entities + source
     tickets.

Every LLM call's tokens are tallied into `index["cost"]` — the upfront price
you pay for the ability to answer global questions. The whole index is cached
to graph_index.json so reruns are instant; delete it (or pass force=True) to
rebuild live.
"""

import json
import os

import anthropic
import networkx as nx
from dotenv import load_dotenv
from networkx.algorithms.community import louvain_communities

from corpus import TICKETS

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
CACHE = os.path.join(os.path.dirname(__file__), "graph_index.json")
_client = anthropic.Anthropic()


def _llm(system, user, cost, max_tokens=700):
    r = _client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    cost["in"] += r.usage.input_tokens
    cost["out"] += r.usage.output_tokens
    cost["calls"] += 1
    return next((b.text for b in r.content if b.type == "text"), "")


def _extract(ticket_text, cost):
    """LLM → {entities:[{name,type}], relationships:[{source,target,desc}]}."""
    raw = _llm(
        "Extract a small knowledge graph from a customer support ticket. "
        "Identify entities — the product features, areas, and problems it "
        "mentions (e.g. 'data export', 'download link', 'billing', '429 error', "
        "'salesforce'). Then the relationships between them. Return ONLY JSON: "
        '{"entities":[{"name":"...","type":"feature|problem|other"}],'
        '"relationships":[{"source":"...","target":"...","desc":"..."}]}. '
        "Use short lowercase entity names so the same concept matches across "
        "tickets.",
        ticket_text, cost, max_tokens=500)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"entities": [], "relationships": []}


def _norm(name: str) -> str:
    return name.strip().lower()


def _canonicalize(names, cost):
    """One LLM pass: map every raw entity name to a canonical name (entity
    resolution). Collapses synonyms so the graph coheres."""
    raw = _llm(
        "These are entity names extracted from many support tickets, with "
        "duplicates and near-synonyms. Map each to a short canonical name so "
        "the same concept shares one name (e.g. 'invoices', 'invoice clarity', "
        "'cryptic charges' → 'billing clarity'). Keep names specific enough to "
        "stay meaningful — aim for roughly 15–20 canonical names total. Return "
        'ONLY JSON: {"raw name": "canonical name", ...} covering every input.',
        "\n".join(sorted(names)), cost, max_tokens=1500)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()
    try:
        m = json.loads(raw)
        return {_norm(k): _norm(v) for k, v in m.items() if v}
    except json.JSONDecodeError:
        return {}


def build(force: bool = False) -> dict:
    if not force and os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)

    cost = {"in": 0, "out": 0, "calls": 0}

    # 1. Extract a raw entity set + relationship pairs per ticket.
    ticket_ents: dict = {}   # tid -> set of raw entity names
    ticket_rels: dict = {}   # tid -> list of (raw source, raw target)
    raw_names: set = set()
    for tid, t in TICKETS.items():
        kg = _extract(t["text"], cost)
        ents = {_norm(e["name"]) for e in kg.get("entities", []) if e.get("name")}
        rels = [(_norm(r.get("source", "")), _norm(r.get("target", "")))
                for r in kg.get("relationships", [])]
        rels = [(s, o) for s, o in rels if s and o and s != o]
        for s, o in rels:
            ents.add(s)
            ents.add(o)
        ticket_ents[tid] = ents
        ticket_rels[tid] = rels
        raw_names |= ents

    # 2. Entity resolution — canonicalize names globally.
    canon = _canonicalize(raw_names, cost)
    c = lambda n: canon.get(n, n)

    # 3. Build the graph: intra-ticket co-occurrence (+ relationship) edges.
    G = nx.Graph()
    ent_tickets: dict = {}
    for tid in TICKETS:
        ents = sorted({c(n) for n in ticket_ents[tid]})
        for n in ents:
            G.add_node(n)
            ent_tickets.setdefault(n, set()).add(tid)
        for i in range(len(ents)):          # co-occurrence clique
            for j in range(i + 1, len(ents)):
                a, b = ents[i], ents[j]
                w = G.get_edge_data(a, b, {}).get("weight", 0) + 1
                G.add_edge(a, b, weight=w)
        for s, o in ticket_rels[tid]:       # relationships add weight
            a, b = c(s), c(o)
            if a != b:
                w = G.get_edge_data(a, b, {}).get("weight", 0) + 1
                G.add_edge(a, b, weight=w)

    # 4. Community detection over the entity graph.
    # resolution > 1 favors more, finer communities (the co-occurrence cliques
    # otherwise merge the graph into just a few coarse blobs).
    communities = (louvain_communities(G, weight="weight", resolution=1.8, seed=7)
                   if G.number_of_edges() else [])

    comm_records = []
    for i, nodes in enumerate(sorted(communities, key=len, reverse=True)):
        nodes = sorted(nodes)
        tickets = sorted({tid for n in nodes for tid in ent_tickets.get(n, [])})
        src = "\n\n".join(f"[{t}] {TICKETS[t]['text']}" for t in tickets)
        summary = _llm(
            "Summarize what this cluster of related support-ticket topics is "
            "about, focusing on the customer problem(s) it represents. 2–3 "
            "sentences.",
            f"Topics: {', '.join(nodes)}\n\nSource tickets:\n{src}",
            cost, max_tokens=250)
        comm_records.append({"id": i, "entities": nodes, "tickets": tickets,
                             "summary": summary.strip()})

    index = {
        "communities": comm_records,
        "entity_tickets": {k: sorted(v) for k, v in ent_tickets.items()},
        "stats": {"entities": G.number_of_nodes(), "edges": G.number_of_edges(),
                  "communities": len(comm_records)},
        "cost": cost,
    }
    with open(CACHE, "w") as f:
        json.dump(index, f, indent=2)
    return index


if __name__ == "__main__":
    idx = build(force=True)
    s = idx["stats"]
    print(f"entities={s['entities']} edges={s['edges']} "
          f"communities={s['communities']}")
    print(f"indexing cost: {idx['cost']['calls']} calls, "
          f"{idx['cost']['in'] + idx['cost']['out']:,} tokens")
    for c in idx["communities"]:
        print(f"\n[community {c['id']}] ({len(c['tickets'])} tickets) "
              f"{', '.join(c['entities'][:8])}")
        print("  " + c["summary"])
