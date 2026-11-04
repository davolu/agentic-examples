"""
Headless scenario — shows the one thing long-term memory buys you.

Alice reports an export problem (session 1). Later, in a SEPARATE session, she
sends a vague follow-up ("it's still not working"). With per-customer memory the
copilot recalls the export context and answers in context. Bob sends the exact
same vague message with NO history — the control — and the copilot has nothing
to personalize from.

Run it twice: the second time Alice's session-1 memory persists on disk, which
is the whole point of a long-term store.
"""

import os
import shutil

# Fresh memory for a reproducible demo (comment out to see persistence across runs).
shutil.rmtree(os.path.join(os.path.dirname(__file__), ".chroma_memory"),
              ignore_errors=True)

from copilot import Copilot  # noqa: E402

cop = Copilot()


def turn(title, cid, msg):
    r = cop.respond(cid, msg)
    c = r["cost"]
    print(f"\n{'='*72}\n{title}  [{cid}]")
    print(f"  customer: {msg}")
    print(f"  route: {r['route']['complexity']}, account={r['route']['account']} "
          f"| kb: {r['kb']} | reflected-revise: {r['revised']}")
    print(f"  memory recalled: "
          + (f"{r['memories']}" if r["memories"] else "(none)"))
    print(f"  copilot: {r['reply']}")
    print(f"  cost: {c['calls']} LLM calls, {c['in'] + c['out']:,} tokens, "
          f"{c['jev']} Jev tokens")


def main():
    print("memory backend:", cop.store.backend)
    turn("Alice · session 1 — reports the problem", "alice",
         "How do I export my data? The download link keeps expiring before the "
         "download finishes.")
    turn("Alice · session 2 (separate, later) — vague follow-up", "alice",
         "Hey, it's still not working for me. Any fix?")
    turn("Bob · no history (control) — same vague message", "bob",
         "Hey, it's still not working for me. Any fix?")
    print(f"\n{'='*72}")
    print("The point: Alice's session-2 reply is grounded in the recalled export "
          "context; Bob's can't be. Memory persists on disk across runs.")


if __name__ == "__main__":
    main()
