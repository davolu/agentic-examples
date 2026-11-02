"""
Jev-routed support chatbot — cutting LLM token cost and latency with a
System One decision model.

A Streamlit support bot for a fictional electronics store (TechGear) that
splits every turn into DECIDING (route, scope, tone, escalation) and WRITING
(bespoke prose). Jev — TypeSafe's decision model ($0.042/M input tokens,
output free) — answers all the deciding questions in one fast call; the LLM
only runs when the route genuinely needs generated text.

The sidebar measures the effect in real money and real milliseconds:
cost per session vs. an all-LLM baseline, and response latency for
Jev-routed turns vs. LLM turns.

Observability: if LANGSMITH_API_KEY is set, the Claude client and the Jev
decision step are traced to LangSmith automatically.

Usage:
    pip install -r requirements.txt
    cp .env.example .env   # add TYPESAFE_API_KEY + ANTHROPIC_API_KEY
    streamlit run app.py
"""

import os
import re
import time

import anthropic
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-opus-4-8")
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
JEV_LIVE = bool(os.environ.get("TYPESAFE_API_KEY"))

# $ per million tokens: (input, output)
LLM_PRICES = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
LLM_PRICE_IN, LLM_PRICE_OUT = LLM_PRICES.get(CHAT_MODEL, (5.00, 25.00))
JEV_PRICE_IN = 0.042  # $/M input tokens; Jev output tokens are free

if JEV_LIVE:
    from typesafe_sdk import Choice, Noul, TypeSafeClient

# --- Optional LangSmith observability -------------------------------------
LANGSMITH = bool(os.environ.get("LANGSMITH_API_KEY"))
if LANGSMITH:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "jev-agentic")
    from langsmith import traceable
    from langsmith.wrappers import wrap_anthropic

    claude = wrap_anthropic(anthropic.Anthropic())
else:
    def traceable(**_kwargs):  # no-op decorator when LangSmith is off
        return lambda fn: fn

    claude = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# The decision layer
# ---------------------------------------------------------------------------

ROUTES = {
    "order_status": "Asking where an order is, tracking, or delivery updates",
    "shipping_faq": "General questions about shipping costs, methods, or times",
    "returns_faq": "Questions about return policy, refunds, or exchanges",
    "product_question": "Questions about a product's specs, compatibility, or availability",
    "complaint": "Something went wrong: broken item, double charge, bad experience",
    "chitchat": "Greetings, thanks, small talk — no support request",
    "out_of_scope": "Unrelated to the store, its products, or orders",
}


@traceable(name="jev_decide", run_type="chain")
def jev_decide(message: str, history: list[dict]) -> dict:
    """One Jev call answering every routine decision for this turn.

    The state includes recent conversation context — a follow-up like "it
    still hasn't arrived!" contains no store words on its own, and a decision
    model needs the same context a human triager would get.
    """
    recent = " | ".join(f"{m['role']}: {m['content'][:200]}" for m in history[-4:])
    start = time.perf_counter()
    with TypeSafeClient(model=JEV_MODEL) as client:
        response = client.system_one(
            state={"recent_conversation": recent, "latest_customer_message": message},
            questions={
                "route": Choice(
                    instructions="What kind of customer-support message is this?",
                    criteria=ROUTES,
                ),
                "sentiment": Choice(
                    instructions="What is the customer's emotional tone?",
                    criteria={"calm": None, "frustrated": None, "angry": None},
                ),
                "needs_human": Noul(
                    instructions="The customer explicitly asks for a human agent, "
                    "or has a problem (legal threat, repeated failure, account "
                    "security) that software should not handle alone",
                ),
                "in_scope": Noul(
                    instructions="The message relates to shopping, orders, "
                    "products, or customer support for an electronics store",
                ),
            },
        )
    latency_ms = (time.perf_counter() - start) * 1000
    route = response.choices["route"]
    usage = getattr(response, "usage", None)
    return {
        "route": route.choice,
        "route_confidence": route.confidence,
        "sentiment": response.choices["sentiment"].choice,
        "needs_human": response.nouls["needs_human"].noul,
        "in_scope": response.nouls["in_scope"].noul,
        "latency_ms": latency_ms,
        "jev_input_tokens": usage.input_tokens if usage else 0,
        "simulated": False,
    }


def simulated_decide(message: str) -> dict:
    """Keyword stand-in used when TYPESAFE_API_KEY is missing."""
    text = message.lower()
    route = "product_question"
    if re.search(r"\b(hi|hello|hey|thanks|thank you)\b", text):
        route = "chitchat"
    if re.search(r"\b(ship|shipping|deliver)\b", text):
        route = "shipping_faq"
    if re.search(r"\b(return|refund|exchange)\b", text):
        route = "returns_faq"
    if re.search(r"\b(order|track|package|#?\d{5,})\b", text):
        route = "order_status"
    if re.search(r"\b(broken|charged twice|terrible|angry|wrong item)\b", text):
        route = "complaint"
    if re.search(r"\b(weather|bitcoin|homework|recipe)\b", text):
        route = "out_of_scope"
    return {
        "route": route,
        "route_confidence": 0.5,
        "sentiment": "frustrated" if route == "complaint" else "calm",
        "needs_human": 0.8 if "human" in text or "agent" in text else 0.1,
        "in_scope": 0.1 if route == "out_of_scope" else 0.9,
        "latency_ms": 0.1,
        "jev_input_tokens": 0,
        "simulated": True,
    }


def decide(message: str, history: list[dict]) -> dict:
    return jev_decide(message, history) if JEV_LIVE else simulated_decide(message)


# ---------------------------------------------------------------------------
# Zero-LLM handlers: templates and a fake order-lookup tool
# ---------------------------------------------------------------------------

CANNED = {
    "shipping_faq": "We ship within 2–4 business days. Standard shipping is "
    "free over $50 (otherwise $4.99); express is $12.99 and arrives in 1–2 "
    "days. You'll get a tracking link by email the moment your order ships. 📦",
    "returns_faq": "You can return any item within 30 days of delivery for a "
    "full refund — no questions asked. Start a return from *Your Orders → "
    "Return item*, print the prepaid label, and your refund lands 3–5 days "
    "after we receive it. 🔄",
    "chitchat": "Hi! 👋 I'm the TechGear assistant. I can check your order "
    "status, answer shipping and returns questions, or help you pick a "
    "product. What can I do for you?",
    "out_of_scope": "I'm only able to help with TechGear orders, products, "
    "shipping and returns. Is there anything store-related I can help with?",
    "escalate": "I've flagged this for a human agent — someone from our team "
    "will reach out within the hour. I'm sorry for the trouble; if there's "
    "anything I can help with in the meantime, I'm here.",
}


def lookup_order(message: str) -> str:
    """Pretend order-lookup tool (deterministic fake data)."""
    match = re.search(r"#?(\d{5,})", message)
    if not match:
        return (
            "Happy to check that! Could you share your order number? "
            "It's the 5+ digit number in your confirmation email (e.g. #48291)."
        )
    order_id = match.group(1)
    statuses = ["packed and awaiting carrier pickup", "in transit", "out for delivery"]
    status = statuses[int(order_id) % len(statuses)]
    eta = ["Thursday", "Friday", "Monday"][int(order_id) % 3]
    return (
        f"Order **#{order_id}** is currently **{status}** and on track to "
        f"arrive **{eta}**. You can follow the live tracking link in your "
        f"shipping-confirmation email. Anything else I can check?"
    )


# ---------------------------------------------------------------------------
# The LLM handler — only runs when the route needs bespoke prose
# ---------------------------------------------------------------------------

TONE_GUIDES = {
    "calm": "Be friendly and efficient.",
    "frustrated": "The customer is frustrated: acknowledge it briefly, be extra "
    "clear and concrete, and don't be chirpy.",
    "angry": "The customer is angry: lead with a sincere apology, take "
    "ownership, keep it short, and offer a concrete next step.",
}


def llm_reply(history: list[dict], sentiment: str):
    """Stream a Claude reply; caller reads text_stream + get_final_message."""
    return claude.messages.stream(
        model=CHAT_MODEL,
        max_tokens=1024,
        system="You are the support assistant for TechGear, an online "
        "electronics store (chargers, cables, keyboards, headphones, smart "
        "home). Answer helpfully and concretely in under 120 words. If you "
        "don't know an order-specific fact, say what the customer should "
        "check rather than inventing it. "
        + TONE_GUIDES.get(sentiment, ""),
        messages=history[-8:],
    )


def llm_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens / 1e6 * LLM_PRICE_IN + output_tokens / 1e6 * LLM_PRICE_OUT


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="TechGear Support — Jev + Claude", page_icon="🛠️")
st.title("🛠️ TechGear Support")
st.caption(
    "An agentic chatbot where **Jev decides** (route, tone, scope, escalation "
    "— one fast, $0.042/M-token call) and **Claude writes only when writing "
    "is needed**."
)

if "messages" not in st.session_state:
    st.session_state.messages = []       # chat history for display + LLM
    st.session_state.turns = []          # per-turn cost/latency metadata

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about orders, shipping, returns, products..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    turn_start = time.perf_counter()
    decision = decide(prompt, st.session_state.messages[:-1])
    route = decision["route"]

    # Routing logic — plain code acting on Jev's answers. Jev answers the
    # atomic questions; PRECEDENCE between them is our job. Escalation beats
    # everything (an angry "get me a human" has no store words, so the scope
    # guardrail would otherwise eat it); the scope gate only fires when the
    # route itself is out_of_scope or the router is unsure.
    used_llm = False
    llm_in = llm_out = 0
    with st.chat_message("assistant"):
        if decision["needs_human"] > 0.7:
            reply = CANNED["escalate"]
            st.markdown(reply)
        elif route == "out_of_scope" or (
            decision["in_scope"] < 0.5 and decision["route_confidence"] < 0.6
        ):
            reply = CANNED["out_of_scope"]
            st.markdown(reply)
        elif route in ("shipping_faq", "returns_faq", "chitchat"):
            reply = CANNED[route]
            st.markdown(reply)
        elif route == "order_status":
            reply = lookup_order(prompt)
            st.markdown(reply)
        else:  # complaint, product_question → real generation
            used_llm = True
            with llm_reply(st.session_state.messages, decision["sentiment"]) as stream:
                reply = st.write_stream(stream.text_stream)
                final = stream.get_final_message()
                llm_in = final.usage.input_tokens
                llm_out = final.usage.output_tokens

        turn_ms = (time.perf_counter() - turn_start) * 1000
        cost = llm_cost(llm_in, llm_out)
        st.caption(
            f"route: **{route}** ({decision['route_confidence']:.0%}) · "
            f"tone: {decision['sentiment']} · "
            f"decision: {decision['latency_ms']:.0f}ms · "
            f"turn: {turn_ms/1000:.1f}s"
            f"{' · simulated' if decision['simulated'] else ''} · "
            f"LLM: {'yes — $' + format(cost, '.4f') if used_llm else 'skipped'}"
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.turns.append(
        {
            **decision,
            "used_llm": used_llm,
            "llm_in": llm_in,
            "llm_out": llm_out,
            "llm_cost": cost,
            "turn_ms": turn_ms,
        }
    )

# ---------------------------------------------------------------------------
# Sidebar: cost and latency, measured
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("💰 Cost & latency, measured")
    if not JEV_LIVE:
        st.warning(
            "TYPESAFE_API_KEY not set — using a keyword **simulator** instead "
            "of Jev. Get a key at console.typesafe.ai."
        )

    turns = st.session_state.get("turns", [])
    if turns:
        llm_turns = [t for t in turns if t["used_llm"]]
        routed = [t for t in turns if not t["used_llm"]]

        # --- Cost ---
        llm_spend = sum(t["llm_cost"] for t in llm_turns)
        jev_spend = sum(t["jev_input_tokens"] for t in turns) / 1e6 * JEV_PRICE_IN
        spend = llm_spend + jev_spend
        # Baseline: every turn handled by the LLM. Avoided turns estimated at
        # the average observed LLM turn (fallback: 450 in / 150 out).
        avg_turn_cost = (llm_spend / len(llm_turns)) if llm_turns else llm_cost(450, 150)
        baseline = llm_spend + len(routed) * avg_turn_cost
        saved = baseline - spend
        pct = (saved / baseline * 100) if baseline > 0 else 0.0

        st.subheader("Cost")
        col1, col2 = st.columns(2)
        col1.metric("This session", f"${spend:.4f}")
        col2.metric("All-LLM baseline", f"${baseline:.4f}")
        st.metric("Saved", f"${saved:.4f}", f"-{pct:.0f}% cost")
        st.caption(
            f"LLM: ${llm_spend:.4f} ({len(llm_turns)} turn(s)) · "
            f"Jev: ${jev_spend:.6f} ({sum(t['jev_input_tokens'] for t in turns):,} tokens)"
        )

        # --- Latency ---
        st.subheader("Latency")
        col3, col4 = st.columns(2)
        if routed:
            routed_avg = sum(t["turn_ms"] for t in routed) / len(routed) / 1000
            col3.metric("Jev-routed turns", f"{routed_avg:.1f}s avg")
        if llm_turns:
            llm_avg = sum(t["turn_ms"] for t in llm_turns) / len(llm_turns) / 1000
            col4.metric("LLM turns", f"{llm_avg:.1f}s avg")
        if routed and llm_turns:
            st.metric(
                "Routine-turn speedup", f"{llm_avg / routed_avg:.1f}x faster"
            )

        st.divider()
        st.subheader("Decision log")
        for t in reversed(turns[-10:]):
            st.markdown(
                f"`{t['route']}` ({t['route_confidence']:.0%}) · {t['sentiment']} · "
                f"{t['turn_ms']/1000:.1f}s · "
                f"{'LLM $' + format(t['llm_cost'], '.4f') if t['used_llm'] else 'no LLM'}"
            )
    else:
        st.caption("Send a message to start measuring.")

    st.divider()
    st.caption(
        f"Decisions: **{'Jev (' + JEV_MODEL + ')' if JEV_LIVE else 'simulator'}** · "
        f"Generation: **{CHAT_MODEL}** · "
        f"Tracing: **{'LangSmith on' if LANGSMITH else 'off'}**"
    )
