"""
Jev-routed support chatbot — an agent that doesn't overthink.

A Streamlit support bot for a fictional electronics store (TechGear) that
splits every turn into two kinds of work:

  DECIDING  — what is this message? is it in scope? does it need a human?
              what tone is the customer in? does it need custom prose at all?
  WRITING   — actually composing a bespoke reply.

Most chatbots burn a full LLM call on both. Here, Jev (TypeSafe's System One
decision model) answers all the deciding questions in ONE fast call, and the
LLM only runs when the route genuinely needs generated text. FAQs, order
lookups, greetings, and out-of-scope messages are handled with templates —
zero LLM tokens.

Usage:
    pip install -r requirements.txt
    # .env file (or exported env vars):
    #   TYPESAFE_API_KEY=...   from https://console.typesafe.ai/keys
    #   ANTHROPIC_API_KEY=...  from https://platform.claude.com
    streamlit run app.py

Without TYPESAFE_API_KEY the app falls back to a clearly-labeled keyword
simulator so the architecture is still demoable.
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

if JEV_LIVE:
    from typesafe_sdk import Choice, Noul, TypeSafeClient

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
        "jev_tokens": (usage.input_tokens + usage.output_tokens) if usage else 0,
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
        "jev_tokens": 0,
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
    """Stream a Claude reply; returns (generator, get_final_message)."""
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


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="TechGear Support — Jev + Claude", page_icon="🛠️")
st.title("🛠️ TechGear Support")
st.caption(
    "An agentic chatbot where **Jev decides** (route, tone, scope, escalation "
    "— one ~fast call) and **Claude writes only when writing is needed**."
)

if "messages" not in st.session_state:
    st.session_state.messages = []       # chat history for display + LLM
    st.session_state.turns = []          # per-turn decision metadata

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about orders, shipping, returns, products..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    decision = decide(prompt, st.session_state.messages[:-1])
    route = decision["route"]

    # Routing logic — plain code acting on Jev's answers. Jev answers the
    # atomic questions; PRECEDENCE between them is our job. Escalation beats
    # everything (an angry "get me a human" has no store words, so the scope
    # guardrail would otherwise eat it); the scope gate only fires when the
    # route itself is out_of_scope or the router is unsure.
    used_llm = False
    llm_tokens = 0
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
                llm_tokens = final.usage.input_tokens + final.usage.output_tokens

        st.caption(
            f"route: **{route}** ({decision['route_confidence']:.0%}) · "
            f"tone: {decision['sentiment']} · "
            f"decision: {decision['latency_ms']:.0f}ms"
            f"{' · simulated' if decision['simulated'] else ''} · "
            f"LLM: {'yes — ' + str(llm_tokens) + ' tokens' if used_llm else 'skipped'}"
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.turns.append(
        {**decision, "used_llm": used_llm, "llm_tokens": llm_tokens}
    )

# ---------------------------------------------------------------------------
# Sidebar: the cost story
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("💰 Decision-layer savings")
    if not JEV_LIVE:
        st.warning(
            "TYPESAFE_API_KEY not set — using a keyword **simulator** instead "
            "of Jev. Get a key at console.typesafe.ai."
        )

    turns = st.session_state.get("turns", [])
    if turns:
        llm_turns = [t for t in turns if t["used_llm"]]
        skipped = len(turns) - len(llm_turns)
        spent = sum(t["llm_tokens"] for t in llm_turns)
        # Estimate: a skipped turn would have cost about the average observed
        # LLM turn (fallback 900 tokens: history + system prompt + reply).
        avg = (sum(t["llm_tokens"] for t in llm_turns) / len(llm_turns)) if llm_turns else 900
        saved = int(skipped * avg)
        jev_ms = [t["latency_ms"] for t in turns if not t["simulated"]]

        col1, col2 = st.columns(2)
        col1.metric("Turns", len(turns))
        col2.metric("LLM calls avoided", f"{skipped}/{len(turns)}")
        col1.metric("LLM tokens spent", f"{spent:,}")
        col2.metric("LLM tokens saved (est.)", f"~{saved:,}")
        if jev_ms:
            st.metric("Avg Jev decision latency", f"{sum(jev_ms)/len(jev_ms):.0f} ms")

        st.divider()
        st.subheader("Decision log")
        for i, t in enumerate(reversed(turns[-10:]), 1):
            st.markdown(
                f"`{t['route']}` ({t['route_confidence']:.0%}) · {t['sentiment']} · "
                f"{'LLM ' + format(t['llm_tokens'], ',') + ' tok' if t['used_llm'] else 'no LLM'}"
            )
    else:
        st.caption("Send a message to start measuring.")

    st.divider()
    st.caption(
        f"Decisions: **{'Jev (' + JEV_MODEL + ')' if JEV_LIVE else 'simulator'}** · "
        f"Generation: **{CHAT_MODEL}**"
    )
