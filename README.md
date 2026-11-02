# Agentic Examples

A collection of small, runnable examples of agentic AI patterns.

## Examples

| Folder | Pattern | Description | Article |
|--------|---------|-------------|---------|
| [`reflectio-agent/`](reflectio-agent/) | Reflection | An agent that critiques and revises its own output in a generate → critique → revise loop until a strict critic approves or an iteration cap is hit. | [Medium](https://medium.com/@oluyaled/the-agent-that-grades-its-own-homework-building-a-reflection-loop-db34e5d70913) |
| [`reflexion-agent/`](reflexion-agent/) | Reflexion | An agent that attempts a task, gets judged by real unit tests, writes a verbal self-reflection on each failure, and retries from scratch with its accumulated lessons in memory (Shinn et al., 2023). | [Medium](https://medium.com/@oluyaled/agent-that-learns-from-its-mistakes-building-a-reflexion-loop-with-claude-8e97a537da71) |
| [`deep-agent/`](deep-agent/) | Deep Agent | A long-horizon research agent with the four deep-agent ingredients — planning todos, a virtual file system, parallel research sub-agents with web search, and a detailed operating prompt — built twice: from scratch on the raw SDK, and with LangChain's `deepagents`. | [Medium](https://medium.com/@oluyaled/going-deep-building-a-deep-agent-twice-from-scratch-then-with-langchain-71cf43a63be5) |

| [`jev-agentic/`](jev-agentic/) | Decision-routed agent | A Streamlit support chatbot where Jev (TypeSafe's System One decision model) answers every routine decision — route, tone, scope, escalation — in one fast call, and the LLM only writes when bespoke prose is needed. Live cost dashboard shows LLM calls avoided and tokens saved. | — |

More examples coming.

## Running an example

Each folder is self-contained with its own `requirements.txt`:

```bash
cd <example-folder>
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python <main script>.py
```
