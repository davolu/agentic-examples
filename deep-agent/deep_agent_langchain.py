"""
Deep Agent — LangChain edition.

The same deep-agent pattern as deep_agent.py (planning todos, a virtual file
system, research sub-agents, detailed instructions), but built with the
`deepagents` library from the LangChain ecosystem instead of from scratch.

`create_deep_agent` ships the deep-agent ingredients as built-ins:
  - write_file / read_file / edit_file / ls / glob / grep (virtual filesystem)
  - task (spawn sub-agents with isolated context)
  - write_todos via TodoListMiddleware (planning)
  - a deep-agent system prompt wrapping your instructions

We add Anthropic's server-side web_search tool so sub-agents can research.

Usage:
    pip install deepagents langchain-anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python deep_agent_langchain.py "Compare FastAPI, Django and Flask for building APIs in 2026"

    # Use a different model (default: claude-opus-4-8)
    DEEP_AGENT_MODEL=claude-sonnet-5 python deep_agent_langchain.py "..."
"""

import os
import sys

from deepagents import create_deep_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_anthropic import ChatAnthropic

MODEL = os.environ.get("DEEP_AGENT_MODEL", "claude-opus-4-8")

# Anthropic's server-side web search — executed on Anthropic's infrastructure,
# so there is no search API key or client-side handler to write.
# NOTE: the basic 20250305 variant, not web_search_20260209 — the newer
# variant runs code execution under the hood (dynamic filtering), and
# LangChain's message conversion drops the paired code_execution_tool_result
# block, producing a 400 on history replay. The raw-SDK version
# (deep_agent.py) can use the newer variant because it echoes response
# blocks back verbatim.
WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}

SYSTEM_PROMPT = """\
You are a deep research agent that produces well-sourced reports.

Operating procedure:
1. Plan with write_todos: one research subtask per independent topic, then a \
final "write report" task.
2. Delegate each research subtask to a researcher sub-agent via the task \
tool. Instruct each sub-agent to research with web search and write its \
findings to a distinct file (e.g. notes/<topic>.md).
3. Keep the todo list current as you work.
4. When all research is done, read the notes files and synthesize the final \
deliverable into report.md.
5. Finish with a 3-5 sentence summary. The report lives in report.md — do not \
paste it into the conversation."""

RESEARCHER = {
    "name": "researcher",
    "description": "Researches one focused topic on the web and writes its "
    "findings to a workspace file.",
    "system_prompt": "You are a focused research sub-agent. Use web search to "
    "research exactly what you were asked — nothing more. Write your findings "
    "to the workspace file you were told to use (markdown, factual, cite "
    "source URLs). Reply with a summary under 150 words.",
    "tools": [WEB_SEARCH],
}


def text_of(content) -> str:
    """LangChain message content can be a string or a list of blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


def run(task: str) -> None:
    agent = create_deep_agent(
        # streaming=True: long turns (thinking + several web searches) would
        # otherwise hit the HTTP read timeout on non-streaming requests.
        model=ChatAnthropic(model=MODEL, max_tokens=16000, streaming=True),
        tools=[WEB_SEARCH],
        system_prompt=SYSTEM_PROMPT,
        subagents=[RESEARCHER],
        middleware=[TodoListMiddleware()],
    )

    print(f"Task: {task}\n")

    # Stream state values so the run is observable, like deep_agent.py's log.
    final_state = None
    seen = set()
    for state in agent.stream(
        {"messages": [{"role": "user", "content": task}]},
        stream_mode="values",
        config={"recursion_limit": 1000},
    ):
        final_state = state
        message = state["messages"][-1]
        for call in getattr(message, "tool_calls", []) or []:
            if call["id"] in seen:
                continue
            seen.add(call["id"])
            args = call.get("args", {})
            if call["name"] == "task":
                print(f"[lead] ⤷ sub-agent: {args.get('description', '')[:80]}", flush=True)
            elif call["name"] == "write_file":
                print(f"[agent] wrote {args.get('file_path', args.get('path', '?'))}", flush=True)
            else:
                print(f"[agent] tool: {call['name']}", flush=True)

    print("\n" + "=" * 60)
    print("LEAD AGENT SUMMARY")
    print("=" * 60)
    print(text_of(final_state["messages"][-1].content))

    files = final_state.get("files", {})
    print("\n" + "=" * 60)
    print(f"WORKSPACE FILES: {list(files)}")
    print("=" * 60)
    # deepagents roots workspace paths (e.g. /tmp/outputs/report.md),
    # so match the report by suffix rather than exact key.
    report = next((v for k, v in files.items() if k.endswith("report.md")), None)
    if report:
        print(report if isinstance(report, str) else text_of(report))


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or (
        "Compare FastAPI, Django REST Framework and Flask for building "
        "production APIs in 2026, and write a recommendation report."
    )
    run(task)
