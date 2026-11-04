"""
Agentic Lead Gen — Streamlit UI.  streamlit run app.py

Type one targeting prompt, watch the pipeline run, and browse the result: the
funnel (sourced → no-site → verified), the honest false-positive rate, the
ranked leads, each generated demo site rendered inline, and the draft pitches.
Pitches are DRAFTS — nothing is sent.
"""

import streamlit as st
import streamlit.components.v1 as components

from pipeline import run_campaign

st.set_page_config(page_title="Agentic Lead Gen", page_icon="🎯",
                   layout="wide")
st.title("🎯 Agentic Lead Gen")
st.caption("One prompt in, a campaign out: plan → source (OpenStreetMap) → score "
           "(Jev) → filter → verify → build demo sites → draft pitches.")

st.warning("Pitches are **drafts for human review** — Agentic Lead Gen never sends anything. "
           "Lead data is from OpenStreetMap.", icon="✋")

prompt = st.text_input("What leads do you want?",
                       value="Find dentists in Austin without a website")

if st.button("Run campaign", type="primary"):
    log = st.empty()
    lines = []

    def progress(m):
        lines.append(f"· {m}")
        log.code("\n".join(lines))

    with st.spinner("running pipeline…"):
        r = run_campaign(prompt, progress=progress)
    st.session_state["result"] = r

r = st.session_state.get("result")
if r:
    st.subheader("Funnel")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sourced (OSM)", r["sourced"])
    c2.metric("No-site tag", r["no_site_candidates"])
    c3.metric("Verified checked", r["verify_n"])
    c4.metric("Already had a site", r["false_positives"],
              f'{r["fp_rate"]:.0%} false-positive', delta_color="inverse")
    c5.metric("Real no-site leads", len(r["verified"]))
    tok = r["cost"]["in"] + r["cost"]["out"]
    st.caption(f'{r["cost"]["calls"]} LLM calls · {tok:,} tokens · the "no website" '
               f'OSM signal was {r["fp_rate"]:.0%} wrong — verification is what '
               f"makes the list trustworthy.")

    st.subheader(f'Top {len(r["top"])} leads — demo site built + pitch drafted')
    for l in r["top"]:
        with st.expander(f'🦷 {l["name"]}  ·  score {l["score"]:.2f}  ·  '
                         f'{l["phone"] or "no phone"} · {l["address"] or "no address"}',
                         expanded=True):
            left, right = st.columns([3, 2])
            with left:
                st.markdown("**Generated demo site**")
                try:
                    with open(l["site"]) as f:
                        components.html(f.read(), height=480, scrolling=True)
                except OSError:
                    st.info("site file not found")
            with right:
                st.markdown("**Draft pitch** _(review before sending)_")
                st.text_area("pitch", l.get("pitch", ""), height=420,
                             label_visibility="collapsed", key=l["name"])
