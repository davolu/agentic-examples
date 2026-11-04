"""
The copilot pipeline — one agentic turn, composed from the series' patterns.

    recall ─▶ route ─▶ retrieve ─▶ draft ─▶ reflect ─▶ respond ─▶ consolidate
   (memory)  (Jev)   (adaptive)          (self-check)          (write memory)

Every step is a LangSmith span (@traceable) and all Claude calls go through a
wrapped client, so a full trace tree shows up in LangSmith per message — route
decision, what memory was recalled, which KB docs grounded the answer, the
reflection revise, and token/latency for each. If LANGSMITH_API_KEY isn't set,
tracing is a silent no-op and everything still runs.
"""

import os

import anthropic
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_anthropic

from kb import retrieve
from memory import get_store
from router import route

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")


class Copilot:
    def __init__(self):
        # wrap_anthropic → every messages.create becomes a traced LLM span.
        self.client = wrap_anthropic(anthropic.Anthropic())
        self.store = get_store()

    def _chat(self, system, user, cost, max_tokens=600):
        r = self.client.messages.create(
            model=MODEL, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        cost["in"] += r.usage.input_tokens
        cost["out"] += r.usage.output_tokens
        cost["calls"] += 1
        return next((b.text for b in r.content if b.type == "text"), "").strip()

    @traceable(name="recall-memory", run_type="retriever")
    def _recall(self, customer_id, message):
        return self.store.recall(customer_id, message, k=3)

    @traceable(name="retrieve-kb", run_type="retriever")
    def _retrieve(self, message, k):
        return retrieve(message, k)

    @traceable(name="draft", run_type="chain")
    def _draft(self, message, kb_docs, memories, cost):
        kb = "\n".join(f"- {d['text']}" for d in kb_docs)
        mem = "\n".join(f"- {m}" for m in memories) or "(none)"
        return self._chat(
            "You are NovaTech's support copilot. Answer the customer using ONLY "
            "the knowledge-base facts provided. Use the remembered context to "
            "personalize (e.g. acknowledge a prior issue) but never invent facts. "
            "If the facts don't cover it, say you'll escalate to a human. Be warm "
            "and concise.",
            f"<kb_facts>\n{kb}\n</kb_facts>\n\n<remembered_about_customer>\n{mem}"
            f"\n</remembered_about_customer>\n\nCustomer message: {message}", cost)

    @traceable(name="reflect", run_type="chain")
    def _reflect(self, message, draft, kb_docs, memories, cost):
        kb = "\n".join(f"- {d['text']}" for d in kb_docs)
        mem = "\n".join(f"- {m}" for m in memories) or "(none)"
        out = self._chat(
            "Review a draft support reply before it is sent. Check that: (1) every "
            "PRODUCT FACT is grounded in the KB facts, (2) it answers the question, "
            "(3) it promises no action we can't verify (refunds, account changes). "
            "Using the remembered customer context to interpret a vague message or "
            "personalize the reply is CORRECT and desired — do not treat it as an "
            "unfounded assumption. Output EXACTLY 'READY:' if it passes or "
            "'REVISED:' if you changed it, followed by ONLY the final reply text — "
            "no explanation, no reasoning.",
            f"<kb_facts>\n{kb}\n</kb_facts>\n\n<remembered_about_customer>\n{mem}\n"
            f"</remembered_about_customer>\n\nCustomer: {message}\n\nDraft: {draft}",
            cost, max_tokens=600)
        revised = out.upper().startswith("REVISED")
        reply = out.split(":", 1)[1].strip() if ":" in out else draft
        return reply, revised

    @traceable(name="consolidate-memory", run_type="chain")
    def _consolidate(self, customer_id, message):
        # Write the interaction back so a future, separate session can recall it.
        self.store.add(customer_id, f"Customer previously asked: {message}")

    @traceable(name="copilot-turn", run_type="chain")
    def respond(self, customer_id: str, message: str) -> dict:
        cost = {"in": 0, "out": 0, "calls": 0, "jev": 0}
        memories = self._recall(customer_id, message)
        decision = route(message, self.client)
        cost["jev"] += decision["jev"]
        k = 5 if decision["complexity"] == "complex" else 3
        kb_docs = self._retrieve(message, k)
        draft = self._draft(message, kb_docs, memories, cost)
        reply, revised = self._reflect(message, draft, kb_docs, memories, cost)
        self._consolidate(customer_id, message)
        return {"reply": reply, "memories": memories, "route": decision,
                "revised": revised, "kb": [d["id"] for d in kb_docs],
                "cost": cost}
