"""
A test client — proves the server works end to end without Claude Desktop.

It launches server.py over stdio exactly as a desktop client would, then
discovers and exercises every primitive: lists tools, calls them, reads a
templated resource, and fetches the prompt. This is also how you'd smoke-test
an MCP server in CI.

    python test_client.py
"""

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(command="python3", args=["server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()

            print("== TOOLS ==")
            tools = await s.list_tools()
            for t in tools.tools:
                print(f"  {t.name}: {t.description.splitlines()[0]}")

            print("\n== CALL add_task x3 ==")
            for title, pri in [("Ship MCP server example", "high"),
                               ("Review injection-arena PR", "medium"),
                               ("Renew TLS cert", "high")]:
                r = await s.call_tool("add_task", {"title": title, "priority": pri})
                print(" ", r.content[0].text)

            print("\n== CALL bulk_add_tasks (with progress) ==")
            r = await s.call_tool("bulk_add_tasks",
                                  {"lines": "Write Q3 report\nUpdate docs\nTriage inbox"})
            print(" ", r.content[0].text)

            print("\n== CALL list_tasks ==")
            r = await s.call_tool("list_tasks", {"status": "open"})
            print(" ", r.content[0].text[:300])

            print("\n== CALL complete_task #1, then search ==")
            print(" ", (await s.call_tool("complete_task", {"task_id": 1})).content[0].text)
            r = await s.call_tool("search", {"query": "cert"})
            print("  search 'cert':", r.content[0].text[:200])

            print("\n== RESOURCES ==")
            res = await s.list_resources()
            for r in res.resources:
                print(f"  {r.uri}: {r.name}")
            content = await s.read_resource("tasks://open")
            print("  read tasks://open:")
            for line in content.contents[0].text.splitlines():
                print("    ", line)

            print("\n== add_note + read note://1 ==")
            print(" ", (await s.call_tool("add_note",
                  {"title": "Deploy steps", "body": "1. tag\n2. push\n3. verify",
                   "tags": "ops"})).content[0].text)
            note = await s.read_resource("note://1")
            print("  read note://1:")
            for line in note.contents[0].text.splitlines():
                print("    ", line)

            print("\n== PROMPTS ==")
            prompts = await s.list_prompts()
            for p in prompts.prompts:
                print(f"  {p.name}: {p.description}")
            got = await s.get_prompt("standup", {})
            print("  standup prompt renders:")
            print("   ", got.messages[0].content.text[:160].replace("\n", " "), "...")

            print("\nAll primitives exercised ✓")


if __name__ == "__main__":
    asyncio.run(main())
