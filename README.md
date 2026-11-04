# Agentic Examples

A collection of small, runnable examples of agentic AI patterns — each self-contained, each producing a real measured result or working artifact, and each paired with a hands-on article.

## The series

Nineteen examples that build on one another, grouped into an arc:

- **Core patterns** — [reflection](reflectio-agent/), [Reflexion](reflexion-agent/), [deep agents](deep-agent/), [decision models](jev-agentic/), and [chain-of-verification](cove-agent/): the building blocks.
- **Capstones** — [agentic-recorder](agentic-recorder/): an agent that films its own demo videos, composing every core pattern into one pipeline; and [support-copilot](support-copilot/): a customer-support product that composes routing, adaptive RAG, reflection, and a decision model into one turn, adding **long-term per-customer memory** (Pinecone-pluggable) and LangSmith observability; and [agentic-lead-gen](agentic-lead-gen/): an agentic lead-gen system where one prompt drives plan → source → score → filter → verify → build demo sites → draft pitches — and verification reveals the raw "no website" filter is ~90% wrong.
- **Trustworthy agents** — [injection defense](injection-arena/), [human-in-the-loop](hitl-agent/), and [evals](agent-evals/): making agents safe, controllable, and measurable.
- **Orchestration & production** — [DAG workflows](DAG-agents/) (parallelism from declared dependencies), [durable/ambient agents](durable-agent/) (surviving production), and [building an MCP server](mcp-server/).
- **Reality checks** — [multi-agent debate](agent-debate/), [agentic RAG](agentic-rag/), [corrective RAG](crag-agent/), [adaptive RAG](adaptive-rag/), and [GraphRAG](graphrag-agent/): hyped techniques measured honestly on a current model — sometimes the gains shrink, sometimes they're real (CRAG recovers corpus-misses via the web; adaptive routing matches the best strategy at far less retrieval; GraphRAG answers global questions top-k can't — but only earns its cost at scale).

Three themes recur across the series: **enforce safety and policy structurally** rather than trusting model goodwill; **measure techniques on your own model** instead of cargo-culting results from older papers; and **build each pattern from scratch once** so the frameworks stop looking like magic.

## Examples

| Folder | Pattern | Description | Article |
|--------|---------|-------------|---------|
| [`reflectio-agent/`](reflectio-agent/) | Reflection | An agent that critiques and revises its own output in a generate → critique → revise loop until a strict critic approves or an iteration cap is hit. | [Medium](https://medium.com/@oluyaled/the-agent-that-grades-its-own-homework-building-a-reflection-loop-db34e5d70913) |
| [`reflexion-agent/`](reflexion-agent/) | Reflexion | An agent that attempts a task, gets judged by real unit tests, writes a verbal self-reflection on each failure, and retries from scratch with its accumulated lessons in memory (Shinn et al., 2023). | [Medium](https://medium.com/@oluyaled/agent-that-learns-from-its-mistakes-building-a-reflexion-loop-with-claude-8e97a537da71) |
| [`deep-agent/`](deep-agent/) | Deep Agent | A long-horizon research agent with the four deep-agent ingredients — planning todos, a virtual file system, parallel research sub-agents with web search, and a detailed operating prompt — built twice: from scratch on the raw SDK, and with LangChain's `deepagents`. | [Medium](https://medium.com/@oluyaled/going-deep-building-a-deep-agent-twice-from-scratch-then-with-langchain-71cf43a63be5) |
| [`jev-agentic/`](jev-agentic/) | Decision-routed agent | A Streamlit support chatbot where Jev (TypeSafe's System One decision model) answers every routine decision — route, tone, scope, escalation — in one fast call, and the LLM only writes when bespoke prose is needed. Live cost dashboard shows LLM calls avoided and tokens saved. | [Medium](https://medium.com/@oluyaled/cutting-llm-token-costs-up-to-40-80-and-making-replies-11x-faster-with-jev-typesafes-system-824dec44d7f6) |
| [`agentic-recorder/`](agentic-recorder/) | Capstone: agentic recording | A desktop app where an agent films product demo videos by itself: you write the brief, it explores the page (sub-agent on Playwright MCP), scripts and critiques a walkthrough (reflection), dry-runs every step (Reflexion), films with an animated cursor, and Jev QA-grades every frame — with cross-session site memory making repeat videos ~10x cheaper. | [Medium](https://medium.com/@oluyaled/self-filming-agents-an-end-to-end-pipeline-for-automated-demo-videos-with-deep-agents-mcp-a217cf687c96) |
| [`support-copilot/`](support-copilot/) | Capstone: product + long-term memory | A customer-support copilot (Streamlit) that composes the patterns into one turn: recall → Jev route → adaptive KB retrieval → draft → reflect → respond → **write memory**. Adds **long-term per-customer memory** behind a pluggable store (local Chroma default, **Pinecone** backend via one env var) and **LangSmith** tracing of every step. Measured: on a vague follow-up, memory lets it resolve what a memoryless bot can't — and the write-up keeps a real bug where the reflection step silently deleted memory's contribution until told it was legitimate context. | — |
| [`agentic-lead-gen/`](agentic-lead-gen/) | Capstone: agentic lead generation | One prompt — *"find dentists in Austin without a website"* — drives a whole campaign: plan → source from **OpenStreetMap** (free, no key) → score with **Jev** → filter → **web-verify** → build a real demo site per lead → draft a pitch. Streamlit + headless, LangSmith-traced. Honest headline: verification found **~90% of the raw "no-website" leads already had sites** (stale OSM tags), so the agentic verify step is what separates a trustworthy list from a spam cannon. Pitches are **drafted, never sent** (human gate). | — |
| [`injection-arena/`](injection-arena/) | Security: prompt injection | An email-assistant agent run through a gauntlet of 10 prompt-injection attacks (authority impersonation, markdown-image exfiltration, compound attacks, system-prompt leaks…), with programmatic success detection. A naive agent vs. a hardened one (spotlighting, programmatic tool guards, output validation, a Jev injection pre-filter) — scorecard: 8/10 → 10/10 blocked, zero utility lost. | [Medium](https://medium.com/@oluyaled/prompt-injecting-an-ai-agent-then-fixing-it-b73da48eba00) |
| [`hitl-agent/`](hitl-agent/) | Human-in-the-loop | A refund-processing agent that pauses before high-stakes actions for a human to approve / edit / reject, then resumes mid-task — built twice: from scratch (interrupt = persist state + exit; resume = reload a checkpoint) and with LangGraph's `interrupt()` + checkpointer. A deterministic gate (plus a Jev second-opinion Noul) decides which refunds need a human. | [Medium](https://medium.com/@oluyaled/human-in-the-loop-agents-demystified-build-the-pause-resume-twice-from-scratch-then-with-577a109edbc9) |
| [`agent-evals/`](agent-evals/) | Evals: your agent needs CI | An eval harness for a triage agent — exact-match on structured fields, a calibrated LLM-as-judge for free-text replies, a near-free Jev judge, cost/latency, and a baseline diff. A "more thorough" v2 prompt looks better but silently drops spam detection 25%; the per-category regression check catches it and blocks the ship. | [Medium](https://medium.com/@oluyaled/building-an-eval-harness-that-catches-a-regression-you-cant-see-7b3894c648f5) |
| [`agent-debate/`](agent-debate/) | Multi-agent debate | Does multi-agent debate actually beat a single pass? N agents answer reasoning-trap questions independently, then see each other and revise over rounds (Du et al., 2023). Measured on a current model: single-pass already scores ~94–100%, and debate's effect is within noise (±1 question) at ~12× the cost — and can *amplify* a shared bias via majority vote. An honest, run-the-numbers-yourself result. | [Medium](https://medium.com/@oluyaled/does-multi-agent-debate-actually-make-models-smarter-i-measured-it-on-a-2026-frontier-model-5004d5a12658) |
| [`durable-agent/`](durable-agent/) | Durability / ambient | A crash-proof batch agent that checkpoints after every item (atomic writes) and resumes exactly where it died — demonstrated by crashing it on purpose mid-batch and watching it pick up at item 9. Idempotent resume makes "just run it again" safe, which also turns it into a scheduled/ambient agent via one cron line. Points to LangGraph checkpointers / Temporal for production. | [Medium](https://medium.com/@oluyaled/crash-proof-agents-building-a-durable-batch-runner-that-resumes-where-it-died-576173a97443) |
| [`mcp-server/`](mcp-server/) | Building an MCP server | The other half of MCP — *serving* tools instead of consuming them. A task & notes MCP server (official SDK) exposing all three primitives (tools, resources, prompts) plus progress reporting, with a stdio test client that proves it end to end and a Claude Desktop/Code connector config. | [Medium](https://medium.com/@oluyaled/building-an-mcp-server-give-claude-your-own-tools-resources-and-prompts-da33966f8bf0) |
| [`DAG-agents/`](DAG-agents/) | DAG orchestration | A résumé-review workflow as a diamond DAG (parse → 3 analyses in parallel → synthesize), built twice: a ~50-line from-scratch scheduler (topological levels + `asyncio.gather` + cycle detection) and with LangGraph. Measured sequential→parallel speedup (1.7×, bounded by nodes/critical-path), a cycle-rejection demo, and an honest note on why DAGs can't do feedback loops. Anchored to LLMCompiler (Kim et al., 2024). | — |
| [`cove-agent/`](cove-agent/) | Chain-of-Verification | An agent that fact-checks itself (Dhuliawala et al., 2023): baseline → plan verification questions → answer them *independently* → revise. Measured on hallucination-prone questions: CoVe lifts 86% → 93%, but the key finding is the independence ablation — *contaminated* (joint) verification scored **79%, below baseline**, so factored beats joint by +14 points. "Add a self-check" is only good if the check is independent. | — |
| [`agentic-rag/`](agentic-rag/) | Agentic RAG | Naive RAG vs. an agentic retrieval loop (rewrite → grade relevance → re-retrieve → abstain), over a **real vector DB** (ChromaDB). Honest measured result: on a strong model + good retriever + modest corpus, naive RAG already abstains on out-of-scope and handles multi-hop — matching agentic RAG at +0% and 1.2× cost. Shows exactly *which conditions* (lossy retrieval, weak models, big corpora) make the loop earn its keep. | — |
| [`crag-agent/`](crag-agent/) | Corrective RAG (CRAG) | Plain RAG vs. CRAG (Yan et al., 2024): a cheap retrieval evaluator (a Jev `Score`) routes each query three ways — *correct* (refine docs), *incorrect* (discard → **web search**), *ambiguous* (both). Measured on questions half in a fictional corpus, half real-world facts it can't contain: CRAG **doubles accuracy, 50% → 100%**, recovering every corpus-miss via web fallback, at 6.4× tokens. The contrast with agentic RAG: CRAG wins because its corrective action *adds a capability the model lacks* (fetch external knowledge), not just gates one it already has (abstain). | — |
| [`adaptive-rag/`](adaptive-rag/) | Adaptive RAG | Route each query by *complexity* (Jeong et al., 2024) to one of three pipelines — no-retrieval / single-step / multi-step iterative — and **measure the router itself**: Jev vs. an LLM classifier vs. an oracle. Result: adaptive matches always-multi-step accuracy (93%) at **42% less retrieval**, over-retrieving *lowers* accuracy, plain single-step RAG is the worst sensible default (53%), and the cheap classifier (Jev, 13/15) routes as well as the LLM (12/15). The win is structural, so it survives on a strong model. | — |
| [`graphrag-agent/`](graphrag-agent/) | GraphRAG | For **global/sensemaking** questions top-k retrieval structurally can't answer (Edge et al., 2024). Built from scratch: LLM entity extraction → entity resolution → `networkx` graph → Louvain communities → community summaries → map-reduce global search (Jev filters partial answers). On "what are customers' recurring problems?" over 25 themed tickets: GraphRAG **7/7 theme coverage** vs top-k vector RAG's 3/7 — but the honest kicker is that *stuff-all-into-context also gets 7/7, cheaper*, so GraphRAG only earns its ~14k-token index at **scale** (when the corpus can't fit). Ties vector RAG on local lookups. | — |

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
- [ChromaDB](https://www.trychroma.com/) & [Pinecone](https://www.pinecone.io/) — vector retrieval and long-term memory (Chroma local by default; Pinecone as the production memory backend)
- [LangSmith](https://smith.langchain.com/) — tracing/observability for the support-copilot and agentic-lead-gen pipelines
- [OpenStreetMap](https://www.openstreetmap.org/) (Nominatim + Overpass) — free, keyless business data for the agentic-lead-gen lead-gen example
- [networkx](https://networkx.org/) — the knowledge graph + Louvain communities behind the GraphRAG example

**Papers & references** (the technique each example is anchored to):

- **Reflexion** — *Reflexion: Language Agents with Verbal Reinforcement Learning*, Shinn et al., 2023 — [`reflexion-agent/`](reflexion-agent/)
- **Self-Refine** — *Self-Refine: Iterative Refinement with Self-Feedback*, Madaan et al., 2023 — [`reflectio-agent/`](reflectio-agent/)
- **Chain-of-Verification** — *Chain-of-Verification Reduces Hallucination in Large Language Models*, Dhuliawala et al., 2023 — [`cove-agent/`](cove-agent/)
- **Multi-Agent Debate** — *Improving Factuality and Reasoning in Language Models through Multiagent Debate*, Du et al., 2023 — [`agent-debate/`](agent-debate/)
- **LLMCompiler** — *An LLM Compiler for Parallel Function Calling*, Kim et al., 2024 — [`DAG-agents/`](DAG-agents/)
- **Self-RAG** — *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*, Asai et al., 2023 — [`agentic-rag/`](agentic-rag/)
- **Corrective RAG (CRAG)** — *Corrective Retrieval Augmented Generation*, Yan et al., 2024 — [`crag-agent/`](crag-agent/)
- **Adaptive-RAG** — *Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity*, Jeong et al., 2024 — [`adaptive-rag/`](adaptive-rag/)
- **GraphRAG** — *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, Edge et al., 2024 — [`graphrag-agent/`](graphrag-agent/)
- **OWASP Top 10 for LLM Applications** (prompt injection, LLM01) — [`injection-arena/`](injection-arena/)

## License

MIT — use the code freely. Attribution appreciated but not required.
