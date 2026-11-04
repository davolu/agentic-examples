# Agentic Examples

A collection of small, runnable examples of agentic AI patterns — each self-contained, each producing a real measured result or working artifact, and each paired with a hands-on article.

## The series

Eleven examples that build on one another, grouped into an arc:

- **Core patterns** — [reflection](reflectio-agent/), [Reflexion](reflexion-agent/), [deep agents](deep-agent/), and [decision models](jev-agentic/): the building blocks.
- **Capstone** — [agentic-recorder](agentic-recorder/): an agent that films its own demo videos, composing every core pattern into one pipeline.
- **Trustworthy agents** — [injection defense](injection-arena/), [human-in-the-loop](hitl-agent/), and [evals](agent-evals/): making agents safe, controllable, and measurable.
- **Reality checks** — [multi-agent debate](agent-debate/) (a hyped technique measured honestly) and [durable/ambient agents](durable-agent/) (surviving production), plus [building an MCP server](mcp-server/).

Three themes recur across the series: **enforce safety and policy structurally** rather than trusting model goodwill; **measure techniques on your own model** instead of cargo-culting results from older papers; and **build each pattern from scratch once** so the frameworks stop looking like magic.

## Examples

| Folder | Pattern | Description | Article |
|--------|---------|-------------|---------|
| [`reflectio-agent/`](reflectio-agent/) | Reflection | An agent that critiques and revises its own output in a generate → critique → revise loop until a strict critic approves or an iteration cap is hit. | [Medium](https://medium.com/@oluyaled/the-agent-that-grades-its-own-homework-building-a-reflection-loop-db34e5d70913) |
| [`reflexion-agent/`](reflexion-agent/) | Reflexion | An agent that attempts a task, gets judged by real unit tests, writes a verbal self-reflection on each failure, and retries from scratch with its accumulated lessons in memory (Shinn et al., 2023). | [Medium](https://medium.com/@oluyaled/agent-that-learns-from-its-mistakes-building-a-reflexion-loop-with-claude-8e97a537da71) |
| [`deep-agent/`](deep-agent/) | Deep Agent | A long-horizon research agent with the four deep-agent ingredients — planning todos, a virtual file system, parallel research sub-agents with web search, and a detailed operating prompt — built twice: from scratch on the raw SDK, and with LangChain's `deepagents`. | [Medium](https://medium.com/@oluyaled/going-deep-building-a-deep-agent-twice-from-scratch-then-with-langchain-71cf43a63be5) |
| [`jev-agentic/`](jev-agentic/) | Decision-routed agent | A Streamlit support chatbot where Jev (TypeSafe's System One decision model) answers every routine decision — route, tone, scope, escalation — in one fast call, and the LLM only writes when bespoke prose is needed. Live cost dashboard shows LLM calls avoided and tokens saved. | [Medium](https://medium.com/@oluyaled/cutting-llm-token-costs-up-to-40-80-and-making-replies-11x-faster-with-jev-typesafes-system-824dec44d7f6) |
| [`agentic-recorder/`](agentic-recorder/) | Capstone: agentic recording | A desktop app where an agent films product demo videos by itself: you write the brief, it explores the page (sub-agent on Playwright MCP), scripts and critiques a walkthrough (reflection), dry-runs every step (Reflexion), films with an animated cursor, and Jev QA-grades every frame — with cross-session site memory making repeat videos ~10x cheaper. | [Medium](https://medium.com/@oluyaled/self-filming-agents-an-end-to-end-pipeline-for-automated-demo-videos-with-deep-agents-mcp-a217cf687c96) |
| [`injection-arena/`](injection-arena/) | Security: prompt injection | An email-assistant agent run through a gauntlet of 10 prompt-injection attacks (authority impersonation, markdown-image exfiltration, compound attacks, system-prompt leaks…), with programmatic success detection. A naive agent vs. a hardened one (spotlighting, programmatic tool guards, output validation, a Jev injection pre-filter) — scorecard: 8/10 → 10/10 blocked, zero utility lost. | [Medium](https://medium.com/@oluyaled/prompt-injecting-an-ai-agent-then-fixing-it-b73da48eba00) |
| [`hitl-agent/`](hitl-agent/) | Human-in-the-loop | A refund-processing agent that pauses before high-stakes actions for a human to approve / edit / reject, then resumes mid-task — built twice: from scratch (interrupt = persist state + exit; resume = reload a checkpoint) and with LangGraph's `interrupt()` + checkpointer. A deterministic gate (plus a Jev second-opinion Noul) decides which refunds need a human. | [Medium](https://medium.com/@oluyaled/human-in-the-loop-agents-demystified-build-the-pause-resume-twice-from-scratch-then-with-577a109edbc9) |
| [`agent-evals/`](agent-evals/) | Evals: your agent needs CI | An eval harness for a triage agent — exact-match on structured fields, a calibrated LLM-as-judge for free-text replies, a near-free Jev judge, cost/latency, and a baseline diff. A "more thorough" v2 prompt looks better but silently drops spam detection 25%; the per-category regression check catches it and blocks the ship. | [Medium](https://medium.com/@oluyaled/building-an-eval-harness-that-catches-a-regression-you-cant-see-7b3894c648f5) |
| [`agent-debate/`](agent-debate/) | Multi-agent debate | Does multi-agent debate actually beat a single pass? N agents answer reasoning-trap questions independently, then see each other and revise over rounds (Du et al., 2023). Measured on a current model: single-pass already scores ~94–100%, and debate's effect is within noise (±1 question) at ~12× the cost — and can *amplify* a shared bias via majority vote. An honest, run-the-numbers-yourself result. | [Medium](https://medium.com/@oluyaled/does-multi-agent-debate-actually-make-models-smarter-i-measured-it-on-a-2026-frontier-model-5004d5a12658) |
| [`durable-agent/`](durable-agent/) | Durability / ambient | A crash-proof batch agent that checkpoints after every item (atomic writes) and resumes exactly where it died — demonstrated by crashing it on purpose mid-batch and watching it pick up at item 9. Idempotent resume makes "just run it again" safe, which also turns it into a scheduled/ambient agent via one cron line. Points to LangGraph checkpointers / Temporal for production. | [Medium](https://medium.com/@oluyaled/crash-proof-agents-building-a-durable-batch-runner-that-resumes-where-it-died-576173a97443) |
| [`mcp-server/`](mcp-server/) | Building an MCP server | The other half of MCP — *serving* tools instead of consuming them. A task & notes MCP server (official SDK) exposing all three primitives (tools, resources, prompts) plus progress reporting, with a stdio test client that proves it end to end and a Claude Desktop/Code connector config. | [Medium](https://medium.com/@oluyaled/building-an-mcp-server-give-claude-your-own-tools-resources-and-prompts-da33966f8bf0) |

More examples coming.

## Running an example

Each folder is self-contained with its own `requirements.txt`:

```bash
cd <example-folder>
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python <main script>.py
```

Some examples use additional keys (e.g. `TYPESAFE_API_KEY` for the Jev decision-model layer, `LANGSMITH_API_KEY` for tracing) — see each folder's `.env.example`.

## Credits

Written and maintained by **[David Oluyalegbenga](https://medium.com/@oluyaled)** — full write-ups for every example are on [Medium](https://medium.com/@oluyaled).

**Built with:**

- [Anthropic Claude](https://www.anthropic.com/) (Opus, Sonnet, Haiku) — the models behind every agent
- [Jev / TypeSafe](https://docs.typesafe.ai) — the System One decision model used for cheap, calibrated routing, gating, and QA
- [LangChain `deepagents`](https://github.com/langchain-ai/deepagents) & [LangGraph](https://github.com/langchain-ai/langgraph) — the framework halves of the deep-agent and human-in-the-loop examples
- [Model Context Protocol](https://modelcontextprotocol.io/) — consumed (Playwright MCP) and served (the MCP server example)
- [Playwright](https://playwright.dev/), [Streamlit](https://streamlit.io/), [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) — browser automation and UIs

**Papers referenced:** Reflexion (Shinn et al., 2023) · Multi-Agent Debate (Du et al., 2023).

## License

MIT — use the code freely. Attribution appreciated but not required.
