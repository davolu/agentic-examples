"""
Human-in-the-loop refund agent — from scratch, no framework.

The whole point of HITL, stripped of magic: an interrupt is just PERSISTED
STATE + an EXIT, and a resume is a RELOAD. When the agent wants to do
something high-stakes, it doesn't call the tool — it writes its full
conversation state to a checkpoint file and stops. A human (maybe minutes,
maybe days later) reviews the proposed action and decides approve / edit /
reject. We reload the checkpoint, inject the decision as the tool result,
and the agent continues as if nothing happened.

The gate policy decides WHICH actions need a human: here, refunds over a
dollar limit OR outside the return window. A Jev Noul gives a cheap second
opinion ("does this refund warrant human review?") — the series' decision
model as a policy input.

Run the demo:
    python refund_agent.py            # processes all tickets, pausing when needed
"""

import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

import tickets as T

load_dotenv()

MODEL = os.environ.get("CHAT_MODEL", "claude-opus-4-8")
JEV_LIVE = bool(os.environ.get("TYPESAFE_API_KEY"))
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
if JEV_LIVE:
    from typesafe_sdk import Noul, TypeSafeClient

client = anthropic.Anthropic()
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

SYSTEM = """You are a customer-support refund agent for an electronics store.
For each ticket: look up the order, check it against the refund policy, then —
if a refund is warranted — call issue_refund with the amount you believe is
correct and a one-line reason.

Do NOT pre-judge whether a refund is "too large" or "out of policy" to issue:
a separate policy layer automatically routes high-value or out-of-window
refunds to a human reviewer. Your job is simply to propose the correct refund.
Only use escalate for tickets that are not refund requests at all (e.g. fraud
reports). Always look up the order and check policy first. Be concise."""

TOOLS = [
    {"name": "lookup_order", "description": "Get order details by id.",
     "input_schema": {"type": "object",
                      "properties": {"order_id": {"type": "string"}},
                      "required": ["order_id"]}},
    {"name": "check_policy",
     "description": "Check an order against the refund policy. Returns whether "
     "it's within the return window and the auto-approve limit.",
     "input_schema": {"type": "object",
                      "properties": {"order_id": {"type": "string"}},
                      "required": ["order_id"]}},
    {"name": "issue_refund",
     "description": "Issue a refund for an order. HIGH-STAKES: may require "
     "human approval depending on amount and policy.",
     "input_schema": {"type": "object",
                      "properties": {"order_id": {"type": "string"},
                                     "amount": {"type": "number"},
                                     "reason": {"type": "string"}},
                      "required": ["order_id", "amount", "reason"]}},
    {"name": "escalate",
     "description": "Escalate the ticket to a human with a note (no refund issued).",
     "input_schema": {"type": "object",
                      "properties": {"note": {"type": "string"}},
                      "required": ["note"]}},
]


# ---------------------------------------------------------------------------
# The gate policy — which refunds need a human
# ---------------------------------------------------------------------------

def needs_human(order_id: str, amount: float) -> tuple[bool, str]:
    order = T.get_order(order_id)
    if not order:
        return True, "unknown order"
    if amount > T.AUTO_APPROVE_LIMIT:
        return True, f"amount ${amount:.2f} exceeds auto-approve limit ${T.AUTO_APPROVE_LIMIT:.0f}"
    if order["days_since_delivery"] > T.RETURN_WINDOW_DAYS:
        return True, f"outside {T.RETURN_WINDOW_DAYS}-day return window"
    return False, "within policy and under limit"


def jev_second_opinion(ticket: dict, order: dict, amount: float) -> float | None:
    if not JEV_LIVE:
        return None
    with TypeSafeClient(model=JEV_MODEL) as c:
        r = c.system_one(
            state={"customer_message": ticket["message"], "order": order,
                   "proposed_refund": amount,
                   "return_window_days": T.RETURN_WINDOW_DAYS},
            questions={"needs_review": Noul(
                instructions="This refund should be reviewed by a human agent "
                "rather than auto-approved (high value, outside policy, unusual "
                "claim, or possible abuse)")},
        )
    return r.nouls["needs_review"].noul


# ---------------------------------------------------------------------------
# Tool execution — issue_refund is the one that can pause
# ---------------------------------------------------------------------------

class PauseForHuman(Exception):
    """Raised to suspend the agent pending a human decision."""
    def __init__(self, proposal):
        self.proposal = proposal


def execute_tool(name, args, ticket):
    if name == "lookup_order":
        order = T.get_order(args["order_id"])
        return json.dumps(order or {"error": "not found"})
    if name == "check_policy":
        order = T.get_order(args["order_id"])
        if not order:
            return json.dumps({"error": "not found"})
        return json.dumps({
            "within_return_window": order["days_since_delivery"] <= T.RETURN_WINDOW_DAYS,
            "days_since_delivery": order["days_since_delivery"],
            "return_window_days": T.RETURN_WINDOW_DAYS,
            "auto_approve_limit": T.AUTO_APPROVE_LIMIT,
            "order_price": order["price"],
        })
    if name == "issue_refund":
        gate, why = needs_human(args["order_id"], args["amount"])
        if gate:
            jev_p = jev_second_opinion(ticket, T.get_order(args["order_id"]), args["amount"])
            raise PauseForHuman({
                "ticket_id": ticket["id"], "order_id": args["order_id"],
                "amount": args["amount"], "reason": args["reason"],
                "gate_reason": why, "jev_needs_review": jev_p,
            })
        return json.dumps({"refunded": True, "amount": args["amount"],
                           "order_id": args["order_id"], "auto_approved": True})
    if name == "escalate":
        return json.dumps({"escalated": True, "note": args["note"]})
    return json.dumps({"error": f"unknown tool {name}"})


# ---------------------------------------------------------------------------
# The agent loop, with durable pause and resume
# ---------------------------------------------------------------------------

def _step(messages, ticket):
    """Run the model/tool loop until done or a human pause is needed."""
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=1500, system=SYSTEM, tools=TOOLS, messages=messages)
        if response.stop_reason != "tool_use":
            text = next((b.text for b in response.content if b.type == "text"), "")
            return {"done": True, "summary": text}

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                out = execute_tool(block.name, block.input, ticket)
            except PauseForHuman as pause:
                # Persist EVERYTHING needed to resume, then stop.
                return {"done": False, "pause": pause.proposal,
                        "pending_tool_use_id": block.id, "messages": messages}
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        messages.append({"role": "user", "content": results})


def process_ticket(ticket):
    messages = [{"role": "user", "content":
                 f"Ticket {ticket['id']} (order {ticket['order_id']}): {ticket['message']}"}]
    state = _step(messages, ticket)
    if state["done"]:
        return {"status": "auto-completed", "summary": state["summary"]}
    # Paused: write a checkpoint a human (or another process) can act on later.
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    cp = CHECKPOINT_DIR / f"{ticket['id']}.json"
    cp.write_text(json.dumps({
        "ticket": ticket, "proposal": state["pause"],
        "pending_tool_use_id": state["pending_tool_use_id"],
        "messages": _serialize(state["messages"]),
    }, indent=1))
    return {"status": "paused", "proposal": state["pause"], "checkpoint": str(cp)}


def resume(ticket_id, decision, edited_amount=None, note=""):
    """Resume a paused ticket. decision ∈ approve | edit | reject."""
    cp = CHECKPOINT_DIR / f"{ticket_id}.json"
    data = json.loads(cp.read_text())
    messages = data["messages"]
    proposal = data["proposal"]

    if decision == "approve":
        result = {"refunded": True, "amount": proposal["amount"],
                  "order_id": proposal["order_id"], "approved_by": "human"}
    elif decision == "edit":
        result = {"refunded": True, "amount": edited_amount,
                  "order_id": proposal["order_id"], "approved_by": "human",
                  "note": f"amount adjusted from ${proposal['amount']:.2f}"}
    elif decision == "reject":
        result = {"refunded": False, "rejected_by": "human", "note": note}
    else:
        raise ValueError(decision)

    # Inject the human decision as the pending tool's result and continue.
    messages.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": data["pending_tool_use_id"],
         "content": json.dumps(result)}]})
    state = _step(messages, data["ticket"])
    cp.unlink(missing_ok=True)
    return {"status": f"resumed ({decision})", "result": result,
            "summary": state.get("summary", "")}


def _serialize(messages):
    """Make the message list JSON-safe (content blocks → dicts)."""
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
        else:
            blocks = []
            for b in c:
                blocks.append(b if isinstance(b, dict) else b.model_dump())
            out.append({"role": m["role"], "content": blocks})
    return out


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A scripted human for the demo: approve the monitor, reject the keyboard.
    human_decisions = {
        "T-502": ("approve", None, ""),
        "T-503": ("reject", None, "Outside return window and 6 weeks is beyond "
                  "a reasonable defect claim — offer store credit instead."),
    }

    for ticket in T.TICKETS:
        print(f"\n{'='*64}\n  {ticket['id']} — order {ticket['order_id']}\n{'='*64}")
        outcome = process_ticket(ticket)
        if outcome["status"] == "auto-completed":
            print(f"  ✅ AUTO-COMPLETED (no human needed)")
            print(f"     {outcome['summary'][:200]}")
        else:
            p = outcome["proposal"]
            jev = f", Jev needs-review p={p['jev_needs_review']:.2f}" if p["jev_needs_review"] is not None else ""
            print(f"  ⏸️  PAUSED FOR HUMAN — gate: {p['gate_reason']}{jev}")
            print(f"     Proposed: refund ${p['amount']:.2f} on {p['order_id']} — {p['reason']}")
            print(f"     Checkpoint written: {Path(outcome['checkpoint']).name}")
            print(f"     (state persisted — the process could exit here and resume days later)")

            decision, amount, note = human_decisions.get(ticket["id"], ("approve", None, ""))
            print(f"  👤 Human decision: {decision.upper()}")
            resumed = resume(ticket["id"], decision, amount, note)
            print(f"  ▶️  RESUMED — {resumed['result']}")
            print(f"     {resumed['summary'][:200]}")
