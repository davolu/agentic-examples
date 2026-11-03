"""
Memory — cross-session site profiles, with a Jev gate in front of the most
expensive phase.

After every successful recording, the site's verified exploration notes and
any hard-won verification lessons are persisted to memory/<domain>.json.
On the next job for that domain, we don't re-explore blindly: one Jev Noul
asks whether the stored profile is sufficient for the new brief. High-
confidence yes → the explore phase (the biggest token line-item in the
pipeline) is skipped entirely. A cheap, calibrated decision standing in
front of an expensive operation — the decision-model pattern.
"""

import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from .telemetry import traceable

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"
JEV_LIVE = bool(os.environ.get("TYPESAFE_API_KEY"))
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")

if JEV_LIVE:
    from typesafe_sdk import Noul, TypeSafeClient


def _domain(url: str) -> str:
    return urlparse(url).netloc.replace(":", "_") or "unknown"


def load_profile(url: str) -> dict | None:
    path = MEMORY_DIR / f"{_domain(url)}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_profile(url: str, notes: str, lessons: list[str]):
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DIR / f"{_domain(url)}.json"
    profile = load_profile(url) or {"domain": _domain(url), "runs": 0, "lessons": []}
    profile["notes"] = notes
    profile["lessons"] = list(dict.fromkeys(profile["lessons"] + lessons))[-20:]
    profile["runs"] += 1
    profile["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(profile, indent=1))


@traceable(name="jev_memory_gate", run_type="chain")
def profile_sufficient(profile: dict, brief: str, ledger=None) -> tuple[bool, float]:
    """Jev Noul: can we skip re-exploration? Returns (skip, probability)."""
    if not JEV_LIVE:
        return False, 0.0
    with TypeSafeClient(model=JEV_MODEL) as client:
        response = client.system_one(
            state={
                "new_video_brief": brief,
                "stored_site_profile_notes": profile["notes"][:6000],
                "profile_age": profile.get("updated_at", "unknown"),
                "past_lessons": profile.get("lessons", []),
            },
            questions={
                "sufficient": Noul(
                    instructions="The stored site profile notes contain enough "
                    "verified selectors and page structure to script this new "
                    "video brief well without re-exploring the site",
                ),
            },
        )
    probability = response.nouls["sufficient"].noul
    usage = getattr(response, "usage", None)
    if ledger and usage:
        ledger.add_jev("memory_gate", usage.input_tokens)
    return probability > 0.75, probability
