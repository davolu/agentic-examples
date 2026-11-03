"""
Human-in-the-loop refund agent — LangGraph edition.

The same agent as refund_agent.py, but the durable pause/resume we hand-rolled
(persist checkpoint + exit; reload + inject decision) is exactly what
LangGraph's `interrupt()` + a checkpointer give you as built-ins:

  - `interrupt(payload)` suspends the graph, persists its full state to the
    checkpointer, and surfaces `payload` to the caller.
  - The caller inspects the interrupt, then resumes with
    `graph.invoke(Command(resume=decision), config=<same thread_id>)`.
  - `interrupt()` returns `decision` and the node continues from that line.

State lives in the checkpointer keyed by thread_id, so "resume days later from
another process" is a SqliteSaver away. We hand-rolled all of this in ~40
lines; here it's the framework's job.

    python refund_agent_langgraph.py
"""

import json
import os
from typing import TypedDict

import anthropic
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import tickets as T
from refund_agent import (SYSTEM, TOOLS, execute_tool, jev_second_opinion,
                          needs_human, PauseForHuman)

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-opus-4-8")
client = anthropic.Anthropic()


class State(TypedDict):
    ticket: dict
    messages: list
    outcome: dict


def agent_node(state: State):
    """Run the model/tool loop; when a refund needs a human, interrupt()."""
    messages = state["messages"]
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=1500, system=SYSTEM, tools=TOOLS, messages=messages)
        if response.stop_reason != "tool_use":
            text = next((b.text for b in response.content if b.type == "text"), "")
            return {"messages": messages, "outcome": {"status": "done", "summary": text}}

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "issue_refund":
                gate, why = needs_human(block.input["order_id"], block.input["amount"])
                if gate:
                    jev_p = jev_second_opinion(
                        state["ticket"], T.get_order(block.input["order_id"]),
                        block.input["amount"])
                    # ---- the one line that replaces our whole checkpoint dance ----
                    decision = interrupt({
                        "proposal": {**block.input, "gate_reason": why,
                                     "jev_needs_review": jev_p}})
                    # execution resumes HERE when the human responds
                    out = _apply_decision(decision, block.input)
                else:
                    out = execute_tool(block.name, block.input, state["ticket"])
            else:
                out = execute_tool(block.name, block.input, state["ticket"])
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        messages.append({"role": "user", "content": results})


def _apply_decision(decision: dict, proposed: dict) -> str:
    kind = decision.get("decision")
    if kind == "approve":
        return json.dumps({"refunded": True, "amount": proposed["amount"],
                           "order_id": proposed["order_id"], "approved_by": "human"})
    if kind == "edit":
        return json.dumps({"refunded": True, "amount": decision["amount"],
                           "order_id": proposed["order_id"], "approved_by": "human",
                           "note": f"adjusted from ${proposed['amount']:.2f}"})
    return json.dumps({"refunded": False, "rejected_by": "human",
                       "note": decision.get("note", "")})


def build_graph():
    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    return g.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    graph = build_graph()
    human_decisions = {
        "T-502": {"decision": "approve"},
        "T-503": {"decision": "reject",
                  "note": "Outside return window; offer store credit instead."},
    }

    for ticket in T.TICKETS:
        print(f"\n{'='*64}\n  {ticket['id']} — order {ticket['order_id']}\n{'='*64}")
        config = {"configurable": {"thread_id": ticket["id"]}}
        init = {"ticket": ticket, "messages": [{"role": "user", "content":
                f"Ticket {ticket['id']} (order {ticket['order_id']}): {ticket['message']}"}]}
        result = graph.invoke(init, config=config)

        if "__interrupt__" in result:
            proposal = result["__interrupt__"][0].value["proposal"]
            jev = (f", Jev p={proposal['jev_needs_review']:.2f}"
                   if proposal.get("jev_needs_review") is not None else "")
            print(f"  ⏸️  INTERRUPTED — gate: {proposal['gate_reason']}{jev}")
            print(f"     Proposed: refund ${proposal['amount']:.2f} on {proposal['order_id']}")
            print(f"     (graph state persisted in checkpointer under thread_id={ticket['id']})")
            decision = human_decisions.get(ticket["id"], {"decision": "approve"})
            print(f"  👤 Human decision: {decision['decision'].upper()}")
            # Resume the SAME thread — interrupt() returns `decision` and continues.
            result = graph.invoke(Command(resume=decision), config=config)
            print(f"  ▶️  RESUMED — {result['outcome']['summary'][:180]}")
        else:
            print(f"  ✅ AUTO-COMPLETED — {result['outcome']['summary'][:180]}")
