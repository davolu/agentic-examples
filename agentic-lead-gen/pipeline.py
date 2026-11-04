"""
The Agentic Lead Gen pipeline — one prompt in, a campaign out.

    plan ─▶ source ─▶ score ─▶ filter ─▶ verify ─▶ build sites ─▶ draft pitches
   (LLM)    (OSM)     (Jev)   (no-site)  (web)     (codegen)      (human-gated)

You configured nothing: the planner wrote the campaign, OSM supplied the leads,
Jev ranked them, verification pruned the false "no website" ones, and the top
few got a real demo site and a draft pitch. Every step is a LangSmith span.
Nothing is sent — pitches are drafts for a human to review.
"""

import os

from dotenv import load_dotenv
from langsmith import traceable

from demo_site import build_site
from pitch import draft_pitch
from planner import plan
from score import score
from sources import find_businesses, geocode
from verify import has_real_website

load_dotenv()


@traceable(name="agentic-lead-gen-campaign", run_type="chain")
def run_campaign(prompt: str, progress=None) -> dict:
    cost = {"in": 0, "out": 0, "calls": 0}

    def say(msg):
        if progress:
            progress(msg)

    spec = plan(prompt)
    say(f"planned: {spec['category']} in {spec['location']} "
        f"(no-website={spec['require_no_website']})")

    bbox = geocode(spec["location"])
    leads = find_businesses(bbox, spec["osm_filters"], spec["max_leads"])
    say(f"sourced {len(leads)} businesses from OpenStreetMap")

    # Filter to the no-website candidates (per OSM tags).
    cands = [l for l in leads if not l["has_website"]] if spec["require_no_website"] else leads
    say(f"{len(cands)} have no website tag in OSM")

    # Score, rank.
    for l in cands:
        l["score"] = score(l)
    cands.sort(key=lambda l: l["score"], reverse=True)

    # Verify down the ranked list, pruning false positives, until we have top_n
    # genuine no-site leads (or hit the check cap). Lazy so we don't web-search
    # the whole list when the real leads are near the top.
    max_verify = min(len(cands), spec.get("max_verify", 12))
    verified, checked, false_pos = [], 0, 0
    for l in cands:
        if checked >= max_verify or len(verified) >= spec["top_n"]:
            break
        checked += 1
        if has_real_website(l, spec["location"], cost):
            l["false_positive"] = True
            false_pos += 1
        else:
            l["false_positive"] = False
            verified.append(l)
    fp_rate = false_pos / checked if checked else 0.0
    say(f"verified {checked}: {false_pos} already had a site "
        f"({fp_rate:.0%} false-positive) → {len(verified)} real no-site leads")

    # Build demo sites + draft pitches for the top verified leads.
    top = verified[:spec["top_n"]]
    for l in top:
        l["site"] = build_site(l, spec["category"], spec["location"], cost)
        l["pitch"] = draft_pitch(l, spec["category"], spec["location"], l["site"], cost)
        say(f"built demo + draft for {l['name']}")

    return {
        "spec": spec,
        "sourced": len(leads),
        "no_site_candidates": len(cands),
        "verify_n": checked,
        "false_positives": false_pos,
        "fp_rate": fp_rate,
        "verified": verified,
        "top": top,
        "cost": cost,
    }
