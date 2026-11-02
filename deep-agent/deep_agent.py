"""
Deep Agent — planning, a file system, and sub-agents for long-horizon tasks.

A "shallow" agent is a model in a tool loop. A deep agent adds the four
ingredients that make long-horizon work tractable:

  1. Planning     — a `write_todos` tool; the agent keeps an explicit, visible
                    plan and updates it as work progresses.
  2. File system  — `write_file` / `read_file` / `ls` over a virtual workspace;
                    findings are offloaded to files instead of bloating context.
  3. Sub-agents   — a `task` tool that spawns a fresh-context research worker
                    (with web search) for each subtask; only its final summary
                    returns to the orchestrator (context isolation).
  4. Detailed system prompt — operating instructions, not just a persona.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python deep_agent.py "Compare FastAPI, Django and Flask for building APIs in 2026"

    # Use a different model (default: claude-opus-4-8)
    DEEP_AGENT_MODEL=claude-sonnet-5 python deep_agent.py "..."
"""

import json
import os
import sys

import anthropic

MODEL = os.environ.get("DEEP_AGENT_MODEL", "claude-opus-4-8")
MAX_TURNS = 30          # orchestrator loop cap
MAX_SUBAGENT_TURNS = 15  # sub-agent loop cap

client = anthropic.Anthropic()

# The virtual file system — shared by the orchestrator and all sub-agents.
FILES: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

WRITE_TODOS = {
    "name": "write_todos",
    "description": "Create or update your task plan. Call this before starting "
    "work, and again whenever a task's status changes. Always send the FULL "
    "list, not a delta.",
    "input_schema": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    },
}

WRITE_FILE = {
    "name": "write_file",
    "description": "Write (or overwrite) a file in the workspace. Use files to "
    "store research notes and the final deliverable instead of keeping "
    "everything in the conversation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "e.g. notes/fastapi.md"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
}

READ_FILE = {
    "name": "read_file",
    "description": "Read a file from the workspace.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}

LS = {
    "name": "ls",
    "description": "List all files in the workspace.",
    "input_schema": {"type": "object", "properties": {}},
}

TASK = {
    "name": "task",
    "description": "Delegate a focused research subtask to a sub-agent. The "
    "sub-agent starts with a FRESH context (it cannot see this conversation), "
    "has web search and the shared workspace files, and returns a summary. "
    "Tell it exactly what to research and which file to write its findings to. "
    "Delegate independent subtasks rather than researching everything yourself.",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short label, e.g. 'research FastAPI'"},
            "prompt": {"type": "string", "description": "Full instructions for the sub-agent"},
        },
        "required": ["description", "prompt"],
    },
}

WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}

ORCHESTRATOR_SYSTEM = """\
You are a deep research agent that produces well-sourced reports.

Operating procedure:
1. Start by calling write_todos with a plan: one research subtask per \
independent topic, then a final "write report" task.
2. Delegate each research subtask to a sub-agent via the task tool. Instruct \
each sub-agent to write its findings to a distinct file (e.g. notes/<topic>.md).
3. Keep your todo list current: mark tasks in_progress / completed as you go.
4. When all research is done, read the notes files and write the final \
deliverable to report.md. Synthesize — do not copy-paste the notes.
5. Finish by replying with a 3-5 sentence summary of the report. Do not paste \
the report into the conversation; it lives in report.md.

Rules: delegate research, don't do it yourself. One sub-agent per independent \
topic. Keep notes files factual and include sources."""

SUBAGENT_SYSTEM = """\
You are a focused research sub-agent. Use web search to research exactly what \
you were asked — nothing more. Write your findings to the workspace file you \
were told to use (markdown, factual, cite source URLs). Then reply with a \
short summary of what you found and wrote. Your reply goes to the lead agent, \
so keep it under 150 words."""

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def log(depth: int, message: str) -> None:
    print(f"{'    ' * depth}{message}", flush=True)


def execute_tool(name: str, args: dict, depth: int) -> str:
    if name == "write_todos":
        marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
        for todo in args["todos"]:
            log(depth, f"  {marks[todo['status']]} {todo['content']}")
        return "Todo list updated."

    if name == "write_file":
        FILES[args["path"]] = args["content"]
        log(depth, f"  wrote {args['path']} ({len(args['content'])} chars)")
        return f"Wrote {args['path']}."

    if name == "read_file":
        if args["path"] not in FILES:
            return f"Error: {args['path']} does not exist. Files: {list(FILES)}"
        return FILES[args["path"]]

    if name == "ls":
        return json.dumps(list(FILES)) if FILES else "(workspace is empty)"

    if name == "task":
        log(depth, f"  ⤷ sub-agent: {args['description']}")
        result = run_agent(
            system=SUBAGENT_SYSTEM,
            prompt=args["prompt"],
            tools=[WRITE_FILE, READ_FILE, LS, WEB_SEARCH],
            label=args["description"],
            depth=depth + 1,
            max_turns=MAX_SUBAGENT_TURNS,
        )
        return result

    return f"Error: unknown tool {name}"


# ---------------------------------------------------------------------------
# The agent loop (used for both the orchestrator and sub-agents)
# ---------------------------------------------------------------------------


def run_agent(system, prompt, tools, label, depth, max_turns) -> str:
    messages = [{"role": "user", "content": prompt}]

    for _ in range(max_turns):
        # Stream and collect: a single turn can run for minutes when the model
        # thinks + runs several server-side web searches, and a non-streaming
        # request would hit the SDK's HTTP read timeout.
        with client.messages.stream(
            model=MODEL,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=system,
            tools=tools,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        # Server-side tool (web search) hit its iteration limit: resume.
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        if response.stop_reason != "tool_use":
            return next(
                (b.text for b in response.content if b.type == "text"),
                "(no text in final response)",
            )

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "server_tool_use":
                log(depth, f"  🔎 web_search: {block.input.get('query', '')}")
            if block.type != "tool_use":
                continue
            log(depth, f"[{label}] tool: {block.name}")
            result = execute_tool(block.name, block.input, depth)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )

        messages.append({"role": "user", "content": tool_results})

    return f"(stopped: hit {max_turns}-turn cap)"


def run(task: str) -> None:
    print(f"Task: {task}\n")
    summary = run_agent(
        system=ORCHESTRATOR_SYSTEM,
        prompt=task,
        tools=[WRITE_TODOS, WRITE_FILE, READ_FILE, LS, TASK],
        label="lead",
        depth=0,
        max_turns=MAX_TURNS,
    )

    print("\n" + "=" * 60)
    print("LEAD AGENT SUMMARY")
    print("=" * 60)
    print(summary)

    print("\n" + "=" * 60)
    print(f"WORKSPACE FILES: {list(FILES)}")
    print("=" * 60)
    if "report.md" in FILES:
        print(FILES["report.md"])


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or (
        "Compare FastAPI, Django REST Framework and Flask for building "
        "production APIs in 2026, and write a recommendation report."
    )
    run(task)
