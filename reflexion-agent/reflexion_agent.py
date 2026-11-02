"""
Reflexion Agent — attempt, evaluate, reflect, retry.

Implements the Reflexion pattern (Shinn et al., 2023): the agent attempts a
whole task, an EXTERNAL evaluator scores the attempt (here: real unit tests
executed in a subprocess), and on failure the agent writes a verbal
self-reflection — a lesson, not an edit. Each new trial starts from scratch,
but with every past reflection in memory.

How this differs from a reflection agent:
  - Feedback is objective (tests pass or they don't), not an LLM critic's opinion.
  - The agent retries the whole task; it does not revise the previous draft.
  - What carries forward between trials is a memory of lessons ("I failed
    because..."), i.e. verbal reinforcement learning.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python reflexion_agent.py

    # Use a different model (default: claude-opus-4-8)
    REFLEXION_MODEL=claude-sonnet-5 python reflexion_agent.py
"""

import os
import re
import subprocess
import sys
import tempfile

import anthropic

MODEL = os.environ.get("REFLEXION_MODEL", "claude-opus-4-8")
MAX_TRIALS = 4

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# The task: a deliberately under-specified spec. The hidden tests encode
# conventions the first attempt is unlikely to fully guess — the point of
# Reflexion is to learn them from objective failures.
# ---------------------------------------------------------------------------

SPEC = """\
Write a Python function `parse_duration(s: str) -> int` that parses a
human-readable duration string and returns the total number of seconds.

Examples:
    parse_duration("1h30m") == 5400
    parse_duration("45s") == 45
"""

HIDDEN_TESTS = """\
assert parse_duration("1h30m") == 5400
assert parse_duration("45s") == 45
assert parse_duration("2h") == 7200
assert parse_duration("90") == 90                 # bare number = seconds
assert parse_duration("1.5h") == 5400             # decimals allowed
assert parse_duration("2h 30m 15s") == 9015       # spaces allowed
assert parse_duration("1d2h") == 93600            # days supported
assert parse_duration("  10m  ") == 600           # surrounding whitespace
try:
    parse_duration("")
    raise SystemExit("FAIL: empty string must raise ValueError")
except ValueError:
    pass
try:
    parse_duration("abc")
    raise SystemExit("FAIL: garbage input must raise ValueError")
except ValueError:
    pass
try:
    parse_duration("-5m")
    raise SystemExit("FAIL: negative durations must raise ValueError")
except ValueError:
    pass
print("ALL TESTS PASSED")
"""


def extract_code(text: str) -> str:
    """Pull the code out of a markdown fence if the model used one."""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text


def act(spec: str, reflections: list[str]) -> str:
    """Attempt the task from scratch, conditioned on past reflections."""
    memory = ""
    if reflections:
        lessons = "\n".join(f"- {r}" for r in reflections)
        memory = (
            f"\n\nLessons from your previous failed attempts "
            f"(do not repeat these mistakes):\n{lessons}"
        )
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system="You are an expert Python programmer. Output only the code for "
        "the requested function — no explanation, no usage examples.",
        messages=[{"role": "user", "content": spec + memory}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return extract_code(text)


def evaluate(code: str) -> tuple[bool, str]:
    """Run the hidden tests against the attempt. Returns (passed, output)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + HIDDEN_TESTS)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=10
        )
    except subprocess.TimeoutExpired:
        return False, "Execution timed out after 10 seconds."
    finally:
        os.unlink(path)
    output = (result.stdout + result.stderr).strip()
    return "ALL TESTS PASSED" in output, output


def reflect(spec: str, code: str, test_output: str) -> str:
    """Write a short verbal lesson about why the attempt failed."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system="You are an agent reflecting on a failed attempt. In 1-3 "
        "sentences, state WHY the attempt failed and WHAT to do differently "
        "next time. Be specific about the failing behavior. Do not write code.",
        messages=[
            {
                "role": "user",
                "content": f"<task>\n{spec}\n</task>\n\n"
                f"<attempt>\n{code}\n</attempt>\n\n"
                f"<test_output>\n{test_output}\n</test_output>\n\n"
                "Write your self-reflection.",
            }
        ],
    )
    return next(b.text for b in response.content if b.type == "text").strip()


def run() -> str | None:
    """Run the Reflexion trial loop. Returns passing code, or None."""
    reflections: list[str] = []

    for trial in range(1, MAX_TRIALS + 1):
        print(f"\n=== Trial {trial}/{MAX_TRIALS} ===")
        print(f"[act] Attempting task ({len(reflections)} lesson(s) in memory)...")
        code = act(SPEC, reflections)

        print("[evaluate] Running hidden tests...")
        passed, output = evaluate(code)
        if passed:
            print("[evaluate] ALL TESTS PASSED")
            return code

        first_line = output.splitlines()[-1] if output else "(no output)"
        print(f"[evaluate] FAILED: {first_line}")

        print("[reflect] Writing self-reflection...")
        lesson = reflect(SPEC, code, output)
        reflections.append(lesson)
        print(f"[reflect] Lesson: {lesson}")

    print(f"\nNo passing attempt after {MAX_TRIALS} trials.")
    return None


if __name__ == "__main__":
    final_code = run()
    if final_code:
        print("\n" + "=" * 60)
        print("PASSING SOLUTION")
        print("=" * 60)
        print(final_code)
