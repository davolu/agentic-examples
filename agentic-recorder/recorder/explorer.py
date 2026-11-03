"""
Explorer — a sub-agent whose browser tools are mounted from the official
Playwright MCP server (@playwright/mcp over stdio).

Two of the "standard stack" concepts live here:

  MCP: we don't hand-roll navigate/click/snapshot tools — we mount them from
  the standard server using the Anthropic SDK's MCP helpers (anthropic[mcp])
  and the beta tool_runner, which drives the agentic loop automatically.

  CONTEXT ISOLATION (deep-agent sub-agent): page snapshots are huge (tens of
  thousands of tokens). They stay inside THIS agent's context; only the final
  notes.md — a few hundred tokens of verified selectors and structure — goes
  back to the orchestrator.

The notes must contain CSS selectors the scriptwriter can trust, so the
explorer is instructed to *verify* each one with browser_evaluate before
writing it down.
"""

import asyncio

from anthropic import AsyncAnthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .telemetry import CHAT_MODEL, LANGSMITH, traceable

# Read/probe tools only — the explorer looks, it doesn't fill forms or buy things.
TOOL_WHITELIST = {
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_hover",
    "browser_evaluate",
    "browser_wait_for",
}

EXPLORER_SYSTEM = """You are a site explorer for an agentic screen recorder.
Your job: explore the target page and produce NOTES that a scriptwriter will
use to write a timed video walkthrough with CSS selectors.

Method:
1. browser_navigate to the URL, then browser_snapshot to see the structure.
2. Identify the 6-10 most tour-worthy elements for the stated brief
   (navigation, hero, standout sections, search, anything distinctive).
3. For EACH, derive a CSS selector and VERIFY it with browser_evaluate, e.g.:
   () => { const el = document.querySelector('li#downloads a');
           return el ? el.textContent.trim().slice(0,60) : 'MISSING'; }
   Only selectors you verified this way may appear in your notes.
4. Note any cookie/consent banner and the verified selector to dismiss it.
5. Explore below the fold (scroll or snapshot again) — tours need depth.

Budget: at most 12 tool calls. Then STOP and output the notes.

Output format (this final message becomes notes.md):
# Site notes: <url>
## Banner to dismiss (if any)
- <selector> — <what it is>
## Tour-worthy elements (verified selectors)
- <selector> — <what it is, why it matters for the brief>
## Page narrative
<2-4 sentences: what story a first-time visitor tour should tell>"""


@traceable(name="explore", run_type="chain")
def explore(url: str, brief: str, ledger=None) -> str:
    """Run the MCP-mounted explorer sub-agent; returns notes markdown."""
    return asyncio.run(_explore_async(url, brief, ledger))


async def _explore_async(url: str, brief: str, ledger=None) -> str:
    client = AsyncAnthropic()
    server = StdioServerParameters(
        command="npx", args=["-y", "@playwright/mcp@latest", "--headless"]
    )
    notes = ""
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            tools = [
                async_mcp_tool(t, session)
                for t in tools_result.tools
                if t.name in TOOL_WHITELIST
            ]
            runner = client.beta.messages.tool_runner(
                model=CHAT_MODEL,
                max_tokens=8000,
                system=EXPLORER_SYSTEM,
                tools=tools,
                messages=[
                    {
                        "role": "user",
                        "content": f"Explore {url} for this brief: {brief}",
                    }
                ],
            )
            iterations = 0
            async for message in runner:
                iterations += 1
                if ledger:
                    ledger.add_llm("explore", message.usage)
                text = "".join(b.text for b in message.content if b.type == "text")
                if text.strip():
                    notes = text
                if iterations >= 16:  # hard cap on loop length
                    break
    return notes
