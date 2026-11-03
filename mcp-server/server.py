"""
A Task & Notes MCP server — the flip side of consuming MCP.

In the agentic-recorder example an agent *consumed* tools from the Playwright
MCP server. Here we *serve* tools: this is an MCP server you plug into Claude
Desktop, Claude Code, or any MCP client, after which Claude can manage your
tasks and notes directly.

It demonstrates all three MCP primitives plus progress reporting:

  TOOLS     — actions the model can take (add/list/complete tasks, notes,
              search, and a bulk import that streams progress).
  RESOURCES — read-only context the client can load on demand
              (tasks://open, note://{id}).
  PROMPTS   — reusable prompt templates the user can invoke (standup).

Run it directly over stdio (how desktop clients launch it):
    python server.py

Built on the official MCP Python SDK (mcp 2.x: FastMCP is now MCPServer).
"""

from mcp.server.mcpserver import Context, MCPServer

import db

db.init()
mcp = MCPServer("tasks-and-notes")


# ---------------------------------------------------------------------------
# TOOLS — verbs the model can call. Clear names + docstrings are the whole
# interface the model sees, so write them like API docs for an LLM.
# ---------------------------------------------------------------------------

@mcp.tool()
def add_task(title: str, priority: str = "medium", tags: str = "") -> str:
    """Create a task. priority is 'high' | 'medium' | 'low'. tags is a
    comma-separated string. Returns the new task id."""
    tid = db.add_task(title, priority, tags)
    return f"Created task #{tid}: {title} [{priority}]"


@mcp.tool()
def list_tasks(status: str = "open", tag: str = "") -> list[dict]:
    """List tasks, highest priority first. status is 'open' | 'done'.
    Optionally filter by a single tag."""
    return db.list_tasks(status, tag or None)


@mcp.tool()
def complete_task(task_id: int) -> str:
    """Mark a task done by its id."""
    ok = db.complete_task(task_id)
    return f"Completed task #{task_id}" if ok else f"No task #{task_id} found"


@mcp.tool()
def add_note(title: str, body: str, tags: str = "") -> str:
    """Save a note (title + body, optional comma-separated tags). Returns its id."""
    nid = db.add_note(title, body, tags)
    return f"Saved note #{nid}: {title}"


@mcp.tool()
def search(query: str) -> dict:
    """Search tasks and notes by keyword (title, tags, and note bodies).
    Returns {'tasks': [...], 'notes': [...]}."""
    return db.search(query)


@mcp.tool()
async def bulk_add_tasks(lines: str, ctx: Context) -> str:
    """Add many tasks at once from newline-separated titles. Streams progress
    to the client — use for long imports."""
    titles = [l.strip() for l in lines.splitlines() if l.strip()]
    created = []
    for i, title in enumerate(titles, 1):
        created.append(db.add_task(title))
        await ctx.report_progress(i, len(titles), f"added {i}/{len(titles)}")
    return f"Imported {len(created)} tasks."


# ---------------------------------------------------------------------------
# RESOURCES — read-only context the client can pull in. A static resource and
# a templated one (the {note_id} is filled from the requested URI).
# ---------------------------------------------------------------------------

@mcp.resource("tasks://open")
def open_tasks() -> str:
    """The current open tasks, highest priority first — handy as context."""
    tasks = db.list_tasks("open")
    if not tasks:
        return "No open tasks."
    return "\n".join(
        f"#{t['id']} [{t['priority']}] {t['title']}"
        f"{' (' + t['tags'] + ')' if t['tags'] else ''}"
        for t in tasks)


@mcp.resource("note://{note_id}")
def note_resource(note_id: str) -> str:
    """A single note's full content, addressed by id."""
    note = db.get_note(int(note_id))
    if not note:
        return f"Note #{note_id} not found."
    return f"# {note['title']}\n({note['created_at']}, tags: {note['tags']})\n\n{note['body']}"


# ---------------------------------------------------------------------------
# PROMPTS — reusable templates the user picks in the client UI.
# ---------------------------------------------------------------------------

@mcp.prompt()
def standup() -> str:
    """Draft a daily standup from the open tasks."""
    return ("Here are my open tasks:\n\n"
            + open_tasks()
            + "\n\nWrite a short daily standup: what I'll focus on today "
            "(top priorities), and flag anything that looks blocked or stale.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
