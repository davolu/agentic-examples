"""
The agent nodes for a résumé-review DAG.

The dependency structure is a diamond:

    parse_resume ──┬─▶ check_ats ──────┐
                   ├─▶ skill_gaps ─────┼─▶ synthesize
                   └─▶ market_compare ─┘

parse_resume is the root; the three analyses depend only on it (so they're
INDEPENDENT of each other and can run in parallel); synthesize fans them in.

Nodes are async and use AsyncAnthropic, so running the three middle nodes
concurrently is real I/O parallelism — the wall-clock win is genuine, not
simulated. Each node takes a dict of its dependencies' outputs and returns
its own output string.
"""

import os

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
client = AsyncAnthropic()

TARGET_ROLE = "Senior Backend Engineer"

RESUME = """\
Jordan Avery
Software Engineer · jordan.avery@email.com

EXPERIENCE
- Software Engineer, Midsize SaaS Co (2021–present)
  Built and maintained REST APIs in Python/Flask. Moved a monolith's billing
  module to a separate service. Wrote integration tests, reduced flaky tests.
  On-call rotation; handled incidents.
- Junior Developer, Agency (2019–2021)
  Built client websites in Django and some React. Managed a small MySQL DB.

SKILLS
Python, Flask, Django, REST, MySQL, Git, Docker (basic), some React, pytest

EDUCATION
B.S. Computer Science, State University, 2019
"""


async def _ask(system: str, user: str, max_tokens: int = 600) -> str:
    r = await client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    return next(b.text for b in r.content if b.type == "text")


async def parse_resume(deps: dict) -> str:
    """Root node: extract a structured summary from the raw résumé."""
    return await _ask(
        "Extract a crisp structured summary of this résumé: current level, "
        "years of experience, core skills, and notable achievements. Bullet points.",
        RESUME)


async def check_ats(deps: dict) -> str:
    """Independent analysis: ATS / keyword readiness for the target role."""
    return await _ask(
        f"You are an ATS-optimization reviewer for a '{TARGET_ROLE}' role. "
        "Given this résumé summary, list the top keyword/format gaps that would "
        "hurt it in applicant-tracking systems, and quick fixes. Be specific.",
        deps["parse"])


async def skill_gaps(deps: dict) -> str:
    """Independent analysis: skill gaps vs the target role."""
    return await _ask(
        f"You are a technical career coach. For a '{TARGET_ROLE}' role, identify "
        "the most important skill gaps in this candidate's profile and what to "
        "learn or demonstrate to close them.",
        deps["parse"])


async def market_compare(deps: dict) -> str:
    """Independent analysis: market positioning vs the target role."""
    return await _ask(
        f"You are a tech recruiter. Compare this candidate against the typical "
        f"'{TARGET_ROLE}' bar: where do they stand (seniority, scope, impact), "
        "and what would make them a competitive applicant?",
        deps["parse"])


async def synthesize(deps: dict) -> str:
    """Fan-in node: combine the three analyses into one prioritized review."""
    return await _ask(
        f"You are a senior career advisor. Combine these three analyses into one "
        f"concise, prioritized résumé-review for a candidate targeting "
        f"'{TARGET_ROLE}'. Lead with the 3 highest-impact actions. No fluff.",
        f"<ats_review>\n{deps['ats']}\n</ats_review>\n\n"
        f"<skill_gaps>\n{deps['skills']}\n</skill_gaps>\n\n"
        f"<market>\n{deps['market']}\n</market>",
        max_tokens=900)


# The DAG: node name -> (async function, list of dependency node names).
# This declaration IS the workflow — the scheduler derives everything
# (execution order, parallelism, termination) from it.
GRAPH = {
    "parse":  (parse_resume, []),
    "ats":    (check_ats, ["parse"]),
    "skills": (skill_gaps, ["parse"]),
    "market": (market_compare, ["parse"]),
    "synth":  (synthesize, ["ats", "skills", "market"]),
}
