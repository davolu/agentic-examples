"""
From durable to ambient — running the batch on a schedule.

A durable runner is already most of the way to an ambient (always-on,
scheduled) agent: because resume is idempotent and cheap, you can simply fire
the SAME runner on a timer. Each tick processes whatever is unfinished and
exits; a tick that finds nothing to do is a no-op. Nothing to orchestrate.

Two ways to schedule it:

1. OS cron (production) — one line, survives reboots, no long-lived process:

     */15 * * * *  cd /path/to/durable-agent && /usr/bin/python3 runner.py >> cron.log 2>&1

   Every 15 minutes the runner picks up any new/unfinished work. When the
   batch is complete, each run is an instant no-op until you add more items.

2. This in-process loop (demos / environments without cron):

     python schedule.py --every 900     # run the batch every 900s

Either way the durability guarantees from runner.py carry over unchanged: a
crash mid-tick just resumes on the next tick.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tick():
    subprocess.run([sys.executable, str(HERE / "runner.py")], check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=900, help="seconds between ticks")
    ap.add_argument("--max-ticks", type=int, default=0, help="0 = forever")
    args = ap.parse_args()

    ticks = 0
    while True:
        print(f"\n— tick {ticks + 1} @ {time.strftime('%H:%M:%S')} —")
        tick()
        ticks += 1
        if args.max_ticks and ticks >= args.max_ticks:
            break
        time.sleep(args.every)
