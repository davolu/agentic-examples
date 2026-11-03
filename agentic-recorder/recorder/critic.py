"""
Critic — the REFLECTION pass. Judges the shot list BEFORE any browser time.

Same pattern as the reflection-agent example: a structured critique with an
approval gate drives the loop. The critic is an opinion (unlike the verifier,
which is objective) — it catches story problems a dry-run never will: wrong
pacing for the duration, captions that don't serve the audience, a tour with
no narrative arc.
"""

from pydantic import BaseModel, Field

from .scriptwriter import ShotList, estimate_seconds
from .telemetry import CHAT_MODEL, make_client, traceable

client = make_client()


class Critique(BaseModel):
    approved: bool = Field(description="True only if the script needs no revision")
    score: int = Field(description="1 (unusable) to 10 (ship it)")
    issues: list[str] = Field(description="Concrete problems, empty if approved")
    suggestions: list[str] = Field(description="A specific fix per issue")


SYSTEM = """You are a demanding video editor reviewing a shot list for an
agent-made product demo. Judge strictly:
- PACING: computed duration must be within ±15% of the target; no step should
  feel rushed (<1s dwell on content) or dead (>4s on nothing).
- NARRATIVE: does the sequence tell a story for the stated audience —
  orientation, highlights, close? Hovering five nav items in a row is a list,
  not a story.
- CAPTIONS: concrete and informative, no marketing fluff, <= 90 chars.
- SELECTOR HYGIENE: every selector must appear in the exploration notes.
Set approved=true only for scripts scoring 8+."""


@traceable(name="critique_script", run_type="chain")
def critique_script(
    brief: str, duration_s: int, shotlist: ShotList, notes: str, ledger=None
) -> Critique:
    response = client.messages.parse(
        model=CHAT_MODEL,
        max_tokens=4000,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"<brief>{brief}</brief>\n"
                f"<target_seconds>{duration_s}</target_seconds>\n"
                f"<computed_duration>{estimate_seconds(shotlist):.1f}s</computed_duration>\n\n"
                f"<shotlist>\n{shotlist.model_dump_json(indent=1)}\n</shotlist>\n\n"
                f"<exploration_notes>\n{notes}\n</exploration_notes>\n\n"
                "Critique this shot list.",
            }
        ],
        output_format=Critique,
    )
    if ledger:
        ledger.add_llm("critique", response.usage)
    return response.parsed_output
