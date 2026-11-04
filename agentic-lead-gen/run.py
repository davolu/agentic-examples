"""
Agentic Lead Gen headless — one prompt in, a campaign out.

    python run.py "Find dentists in Austin without a website"

Prints the funnel (sourced → no-site → verified) with the honest false-positive
rate, and writes real artifacts: demo sites under output/sites/<biz>/index.html
and draft pitches under output/pitches/<biz>.txt. Nothing is sent.
"""

import json
import os
import sys

from pipeline import run_campaign

DEFAULT = "Find dentists in Austin without a website"


def main():
    prompt = " ".join(sys.argv[1:]) or DEFAULT
    print(f'> {prompt}\n')
    r = run_campaign(prompt, progress=lambda m: print(f"  · {m}"))

    print("\n" + "=" * 68)
    print(f'FUNNEL   sourced {r["sourced"]}  →  no-site tag {r["no_site_candidates"]}'
          f'  →  verified real no-site {len(r["verified"])}')
    print(f'VERIFY   checked top {r["verify_n"]}, {r["false_positives"]} actually '
          f'had a site  ({r["fp_rate"]:.0%} false-positive rate in OSM "no website")')
    tok = r["cost"]["in"] + r["cost"]["out"]
    print(f'COST     {r["cost"]["calls"]} LLM calls, {tok:,} tokens\n')

    print(f'TOP {len(r["top"])} LEADS — demo site built + pitch drafted (NOT sent):')
    for l in r["top"]:
        print(f'  • {l["name"]}  (score {l["score"]:.2f})')
        print(f'      phone: {l["phone"] or "—"}   addr: {l["address"] or "—"}')
        print(f'      site:  {os.path.relpath(l["site"])}')
    # Persist the full campaign (minus the big HTML) for inspection/export.
    out = os.path.join(os.path.dirname(__file__), "output", "campaign.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({k: v for k, v in r.items() if k != "cost"} | {"cost": r["cost"]},
                  f, indent=2, default=str)
    print(f'\nwrote campaign.json + {len(r["top"])} demo sites + draft pitches '
          f'under output/')


if __name__ == "__main__":
    main()
