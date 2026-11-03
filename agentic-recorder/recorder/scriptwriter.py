"""
Scriptwriter — turns explorer notes + the brief into a timed shot list.

Structured output end to end: the shot list is a Pydantic schema and the API
guarantees the response matches it (messages.parse), so the filmer never
parses prose. Lessons from failed verification trials (Reflexion) and the
critic's notes (reflection) are injected into the prompt on retries.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .telemetry import CHAT_MODEL, make_client, traceable

client = make_client()

# Per-step overhead the filmer adds on top of dwell (movement, ripples, loads)
OVERHEAD = {"goto": 2.0, "click": 2.2, "hover": 1.0, "scroll_to": 1.5, "pause": 0.2}


class Step(BaseModel):
    action: Literal["goto", "click", "hover", "scroll_to", "pause"]
    url: Optional[str] = Field(None, description="Required for goto")
    selector: Optional[str] = Field(
        None, description="CSS selector, required for click/hover/scroll_to. "
        "Use selectors confirmed to exist in the exploration notes."
    )
    caption: str = Field(description="On-screen caption for this step, <= 90 chars")
    dwell: float = Field(description="Seconds to hold after the action, 1.0-4.0")


class ShotList(BaseModel):
    title: str
    steps: list[Step] = Field(description="6-10 steps for a ~30s video")
    estimated_seconds: float = Field(
        description="Sum over steps of (dwell + action overhead)"
    )


SYSTEM = """You are a demo-video scriptwriter for an agentic screen recorder.
You write shot lists that a deterministic filmer replays in a browser with an
animated cursor and caption overlays.

Rules:
- Every selector must be COPIED CHARACTER-FOR-CHARACTER from the exploration
  notes. Do not construct, paraphrase, or "improve" selectors — if the notes
  don't list a selector for something, you cannot visit it. When in doubt,
  use scroll_to/hover on elements the notes DO list.
- The ONLY goto is the first step, to the target URL. Never goto a subpage
  unless its full URL appears verbatim in the notes. Prefer hover over click
  for navigation items so the tour stays on one page.
- If the notes mention a cookie/consent banner, dismiss it with a click step
  immediately after the goto.
- Captions tell the story for the stated audience; short, concrete, no fluff.
- TIME BUDGET: per-step cost is dwell + overhead (goto 2.0s, click 2.2s,
  hover 1.0s, scroll_to 1.5s, pause 0.2s). The total must land within ±15%
  of the requested duration. Compute estimated_seconds honestly.
- Prefer hover/scroll_to over click for a tour; every click must stay on the
  same site. End with a pause step that closes the story."""


@traceable(name="write_script", run_type="chain")
def write_script(
    brief: str,
    url: str,
    duration_s: int,
    notes: str,
    lessons: list[str],
    critique: str = "",
    ledger=None,
) -> ShotList:
    extras = ""
    if lessons:
        extras += "\n\nLESSONS from failed verification attempts (do not repeat these mistakes):\n" + "\n".join(
            f"- {l}" for l in lessons
        )
    if critique:
        extras += f"\n\nEDITOR'S CRITIQUE of your previous draft (address every point):\n{critique}"

    response = client.messages.parse(
        model=CHAT_MODEL,
        max_tokens=8000,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"<brief>\n{brief}\n</brief>\n"
                f"<target_url>{url}</target_url>\n"
                f"<target_duration_seconds>{duration_s}</target_duration_seconds>\n\n"
                f"<exploration_notes>\n{notes}\n</exploration_notes>{extras}\n\n"
                "Write the shot list.",
            }
        ],
        output_format=ShotList,
    )
    if ledger:
        ledger.add_llm("script", response.usage)
    return response.parsed_output


def estimate_seconds(shotlist: ShotList) -> float:
    return sum(s.dwell + OVERHEAD[s.action] for s in shotlist.steps)
