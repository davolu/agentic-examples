"""
Verifier — the Reflexion evaluator. Objective, not an opinion.

Dry-runs the shot list in a real (non-recording) browser at speed: navigations
happen, selectors must resolve to visible elements, clicks execute so later
steps are validated in the page state they'll actually run in.

On failure, a short Claude call writes a verbal LESSON ("the pricing cards
lazy-load; scroll before clicking") — not an edit. The orchestrator feeds
lessons back to the scriptwriter, which retries FROM SCRATCH. Lessons also
persist into the site's memory profile for future runs.
"""

from playwright.sync_api import sync_playwright

from .scriptwriter import ShotList
from .telemetry import CHAT_MODEL, make_client, traceable

client = make_client()


@traceable(name="dry_run", run_type="chain")
def dry_run(shotlist: ShotList) -> list[dict]:
    """Execute every step without recording. Returns a list of failures."""
    failures = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context(viewport={"width": 1280, "height": 720}).new_page()
        for i, step in enumerate(shotlist.steps):
            try:
                if step.action == "goto":
                    page.goto(step.url, wait_until="domcontentloaded", timeout=20000)
                elif step.action in ("click", "hover"):
                    locator = page.locator(step.selector).first
                    locator.wait_for(state="visible", timeout=6000)
                    box = locator.bounding_box()
                    if not box:
                        raise RuntimeError("element has no bounding box")
                    if step.action == "click":
                        page.mouse.click(box["x"] + box["width"] / 2,
                                         box["y"] + box["height"] / 2)
                        page.wait_for_timeout(700)
                elif step.action == "scroll_to":
                    locator = page.locator(step.selector).first
                    locator.wait_for(state="attached", timeout=6000)
                elif step.action == "pause":
                    pass
            except Exception as exc:
                failures.append(
                    {"step_index": i, "action": step.action,
                     "selector": step.selector, "url": step.url,
                     "error": str(exc).splitlines()[0][:300]}
                )
        browser.close()
    return failures


@traceable(name="reflect_on_failures", run_type="chain")
def reflect(brief: str, shotlist: ShotList, failures: list[dict], ledger=None) -> str:
    """Write a 1-3 sentence lesson about WHY verification failed."""
    failure_text = "\n".join(
        f"- step {f['step_index']} ({f['action']} {f['selector'] or f['url']}): {f['error']}"
        for f in failures
    )
    response = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=2000,
        system="You are an agent reflecting on a failed browser dry-run. In 1-3 "
        "sentences, state WHY the steps failed and WHAT the next script should "
        "do differently (different selector strategy, scroll first, dismiss a "
        "banner, avoid that element). Be specific. Do not write the script.",
        messages=[
            {
                "role": "user",
                "content": f"<brief>{brief}</brief>\n\n"
                f"<shotlist>\n{shotlist.model_dump_json(indent=1)}\n</shotlist>\n\n"
                f"<failures>\n{failure_text}\n</failures>\n\nWrite the lesson.",
            }
        ],
    )
    if ledger:
        ledger.add_llm("verify", response.usage)
    return next(b.text for b in response.content if b.type == "text").strip()
