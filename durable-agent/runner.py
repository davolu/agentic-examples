"""
A durable batch runner — survives crashes, resumes where it died.

The entire durability pattern, with nothing hidden:

  1. CHECKPOINT AFTER EVERY ITEM, not at the end. A result that isn't persisted
     doesn't exist — a crash at item 20 of 25 must not lose items 1-19.
  2. ATOMIC WRITES. Write to a temp file, then os.replace() it over the real
     one. os.replace is atomic on POSIX, so a crash mid-write can never leave a
     half-written, corrupt checkpoint — you have either the old file or the new
     one, never garbage.
  3. IDEMPOTENT RESUME. On startup, load the checkpoint and SKIP items already
     done (keyed by id). Re-running the whole batch is always safe; completed
     work is never redone.
  4. AT-LEAST-ONCE semantics. We mark an item done only after its result is
     persisted. So a crash *during* an item simply redoes that one item on
     resume — safe precisely because the task is idempotent by id.

`CRASH_AFTER=N` deterministically simulates a crash after N items this run, so
you can watch resume work without pulling the power cord.

    python runner.py                 # run / resume until the batch is done
    CRASH_AFTER=8 python runner.py   # process 8, then 'crash'
    python runner.py --status        # show progress without running
"""

import json
import os
import sys
import time
from pathlib import Path

from work import COMPANIES, enrich

CHECKPOINT = Path(__file__).resolve().parent / "checkpoint.json"
CRASH_AFTER = int(os.environ.get("CRASH_AFTER", "0"))


def load() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {"done": {}, "in_tokens": 0, "out_tokens": 0, "started_at": None}


def save_atomic(state: dict):
    """Write to a temp file, then atomically replace — never a half-written file."""
    tmp = CHECKPOINT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, CHECKPOINT)   # atomic on POSIX


def status(state: dict):
    done = len(state["done"])
    print(f"Progress: {done}/{len(COMPANIES)} items done "
          f"({done/len(COMPANIES):.0%})")
    if done:
        print(f"Tokens so far: {state['in_tokens']}in / {state['out_tokens']}out")
        remaining = [c for c in COMPANIES if c not in state["done"]]
        if remaining:
            print(f"Next up: {', '.join(remaining[:5])}"
                  f"{' ...' if len(remaining) > 5 else ''}")


def run():
    state = load()
    state["started_at"] = state["started_at"] or time.strftime("%Y-%m-%dT%H:%M:%S")

    already = len(state["done"])
    if already:
        print(f"↻ Resuming — {already}/{len(COMPANIES)} already done, "
              f"skipping them.\n")
    else:
        print(f"▶ Starting fresh batch of {len(COMPANIES)} companies.\n")

    processed_this_run = 0
    for company in COMPANIES:
        if company in state["done"]:          # idempotent: never redo
            continue

        profile, ti, to = enrich(company)     # the actual work
        state["done"][company] = profile
        state["in_tokens"] += ti
        state["out_tokens"] += to
        save_atomic(state)                    # persist BEFORE moving on
        processed_this_run += 1
        print(f"  ✓ [{len(state['done']):2}/{len(COMPANIES)}] {company:12} "
              f"— {profile['industry']}")

        if CRASH_AFTER and processed_this_run >= CRASH_AFTER:
            print(f"\n💥 Simulated crash after {processed_this_run} items this run.")
            print(f"   {len(state['done'])}/{len(COMPANIES)} are safely checkpointed.")
            sys.exit(1)

    print(f"\n🏁 Batch complete — {len(state['done'])}/{len(COMPANIES)} done, "
          f"{state['in_tokens']}in/{state['out_tokens']}out tokens.")
    print(f"   Results in {CHECKPOINT.name}")


if __name__ == "__main__":
    if "--status" in sys.argv:
        status(load())
    elif "--reset" in sys.argv:
        CHECKPOINT.unlink(missing_ok=True)
        print("Checkpoint cleared.")
    else:
        run()
