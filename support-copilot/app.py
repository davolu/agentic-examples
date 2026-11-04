"""
Streamlit chat UI for the support copilot.

    streamlit run app.py

Pick a customer in the sidebar and chat. Because memory is per-customer and
persistent, switch customers (or restart the app) and a returning customer is
still remembered. The sidebar shows the live cost dashboard (LLM calls, tokens,
Jev tokens) and — per message — the route decision, what memory was recalled,
which KB docs grounded the answer, and whether reflection revised the draft.
Full traces appear in LangSmith when LANGSMITH_API_KEY is set.
"""

import streamlit as st

from copilot import Copilot

st.set_page_config(page_title="NovaTech Support Copilot", page_icon="🎧")


@st.cache_resource
def get_copilot():
    return Copilot()


cop = get_copilot()

# ---- sidebar: customer + cost dashboard ----
with st.sidebar:
    st.header("NovaTech Support Copilot")
    customer = st.selectbox("Customer", ["alice", "bob", "carol"],
                            help="Memory is scoped per customer and persists.")
    st.caption(f"Memory backend: **{cop.store.backend}**")

    st.divider()
    st.subheader("Cost dashboard")
    totals = st.session_state.setdefault("totals", {"calls": 0, "tok": 0, "jev": 0})
    c1, c2, c3 = st.columns(3)
    c1.metric("LLM calls", totals["calls"])
    c2.metric("Tokens", f'{totals["tok"]:,}')
    c3.metric("Jev tok", totals["jev"])
    if st.button("Reset conversation view"):
        st.session_state.pop("chats", None)
        st.rerun()

# ---- chat state, per customer ----
chats = st.session_state.setdefault("chats", {})
history = chats.setdefault(customer, [])

st.title("🎧 Support chat")
st.caption(f"Chatting as **{customer}**")

for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            m = msg["meta"]
            with st.expander("trace"):
                st.write(f"**route:** {m['route']['complexity']}, "
                         f"account={m['route']['account']}")
                st.write(f"**memory recalled:** "
                         + ("; ".join(m["memories"]) if m["memories"] else "(none)"))
                st.write(f"**KB docs:** {', '.join(m['kb'])}")
                st.write(f"**reflection revised the draft:** {m['revised']}")
                st.write(f"**cost:** {m['cost']['calls']} calls, "
                         f"{m['cost']['in'] + m['cost']['out']:,} tokens, "
                         f"{m['cost']['jev']} Jev tokens")

if prompt := st.chat_input("Ask NovaTech support…"):
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("thinking…"):
            r = cop.respond(customer, prompt)
        st.markdown(r["reply"])
    history.append({"role": "assistant", "content": r["reply"], "meta": r})
    totals["calls"] += r["cost"]["calls"]
    totals["tok"] += r["cost"]["in"] + r["cost"]["out"]
    totals["jev"] += r["cost"]["jev"]
    st.rerun()
