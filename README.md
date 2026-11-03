# Agentic Examples

A collection of small, runnable examples of agentic AI patterns.

## Examples

| Folder | Pattern | Description | Article |
|--------|---------|-------------|---------|
| [`reflectio-agent/`](reflectio-agent/) | Reflection | An agent that critiques and revises its own output in a generate → critique → revise loop until a strict critic approves or an iteration cap is hit. | [Medium](https://medium.com/@oluyaled/the-agent-that-grades-its-own-homework-building-a-reflection-loop-db34e5d70913) |
| [`reflexion-agent/`](reflexion-agent/) | Reflexion | An agent that attempts a task, gets judged by real unit tests, writes a verbal self-reflection on each failure, and retries from scratch with its accumulated lessons in memory (Shinn et al., 2023). | [Medium](https://medium.com/@oluyaled/agent-that-learns-from-its-mistakes-building-a-reflexion-loop-with-claude-8e97a537da71) |
| [`deep-agent/`](deep-agent/) | Deep Agent | A long-horizon research agent with the four deep-agent ingredients — planning todos, a virtual file system, parallel research sub-agents with web search, and a detailed operating prompt — built twice: from scratch on the raw SDK, and with LangChain's `deepagents`. | [Medium](https://medium.com/@oluyaled/going-deep-building-a-deep-agent-twice-from-scratch-then-with-langchain-71cf43a63be5) |
| [`jev-agentic/`](jev-agentic/) | Decision-routed agent | A Streamlit support chatbot where Jev (TypeSafe's System One decision model) answers every routine decision — route, tone, scope, escalation — in one fast call, and the LLM only writes when bespoke prose is needed. Live cost dashboard shows LLM calls avoided and tokens saved. | [Medium](https://medium.com/@oluyaled/cutting-llm-token-costs-up-to-40-80-and-making-replies-11x-faster-with-jev-typesafes-system-824dec44d7f6) |
| [`agentic-recorder/`](agentic-recorder/) | Capstone: agentic recording | A desktop app where an agent films product demo videos by itself: you write the brief, it explores the page (sub-agent on Playwright MCP), scripts and critiques a walkthrough (reflection), dry-runs every step (Reflexion), films with an animated cursor, and Jev QA-grades every frame — with cross-session site memory making repeat videos ~10x cheaper. | [Medium](https://medium.com/@oluyaled/self-filming-agents-an-end-to-end-pipeline-for-automated-demo-videos-with-deep-agents-mcp-a217cf687c96) |
| [`injection-arena/`](injection-arena/) | Security: prompt injection | An email-assistant agent run through a gauntlet of 10 prompt-injection attacks (authority impersonation, markdown-image exfiltration, compound attacks, system-prompt leaks…), with programmatic success detection. A naive agent vs. a hardened one (spotlighting, programmatic tool guards, output validation, a Jev injection pre-filter) — scorecard: 8/10 → 10/10 blocked, zero utility lost. | — |
| [`hitl-agent/`](hitl-agent/) | Human-in-the-loop | A refund-processing agent that pauses before high-stakes actions for a human to approve / edit / reject, then resumes mid-task — built twice: from scratch (interrupt = persist state + exit; resume = reload a checkpoint) and with LangGraph's `interrupt()` + checkpointer. A deterministic gate (plus a Jev second-opinion Noul) decides which refunds need a human. | — |
| [`agent-evals/`](agent-evals/) | Evals: your agent needs CI | An eval harness for a triage agent — exact-match on structured fields, a calibrated LLM-as-judge for free-text replies, a near-free Jev judge, cost/latency, and a baseline diff. A "more thorough" v2 prompt looks better but silently drops spam detection 25%; the per-category regression check catches it and blocks the ship. | — |
| [`agent-debate/`](agent-debate/) | Multi-agent debate | Does multi-agent debate actually beat a single pass? N agents answer reasoning-trap questions independently, then see each other and revise over rounds (Du et al., 2023). Measured on a current model: single-pass already scores ~94–100%, and debate's effect is within noise (±1 question) at ~12× the cost — and can *amplify* a shared bias via majority vote. An honest, run-the-numbers-yourself result. | — |

More examples coming.

## Running an example

Each folder is self-contained with its own `requirements.txt`:

```bash
cd <example-folder>
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python <main script>.py
```
