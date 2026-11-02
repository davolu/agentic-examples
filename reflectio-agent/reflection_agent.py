"""
Reflection Agent — generate, critique, revise.

The agent runs three roles in a loop:
  1. Generator  — produces a first draft of the answer.
  2. Critic     — evaluates the draft against explicit quality criteria
                  and returns a structured critique (approved or not).
  3. Reviser    — rewrites the draft using the critique as guidance.

The loop ends when the critic approves the draft or MAX_ITERATIONS is hit.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python reflection_agent.py "Write a short blog post about why unit tests matter"

    # Use a different model (default: claude-opus-4-8)
    REFLECTION_MODEL=claude-sonnet-5 python reflection_agent.py "..."
"""

import os
import sys

import anthropic
from pydantic import BaseModel, Field

# Override with e.g. REFLECTION_MODEL=claude-fable-5 or claude-sonnet-5
MODEL = os.environ.get("REFLECTION_MODEL", "claude-opus-4-8")
MAX_ITERATIONS = 3

client = anthropic.Anthropic()


class Critique(BaseModel):
    """Structured output the critic must return."""

    approved: bool = Field(description="True only if the draft needs no further revision")
    score: int = Field(description="Overall quality score from 1 (poor) to 10 (excellent)")
    strengths: list[str] = Field(description="What the draft does well")
    issues: list[str] = Field(description="Concrete problems found in the draft")
    suggestions: list[str] = Field(description="Specific, actionable fixes for each issue")


def generate(task: str) -> str:
    """Produce the first draft."""
    with client.messages.stream(
        model=MODEL,
        max_tokens=64000,
        thinking={"type": "adaptive"},
        system="You are a skilled writer. Produce the best first draft you can. "
        "Output only the draft itself — no preamble or meta-commentary.",
        messages=[{"role": "user", "content": task}],
    ) as stream:
        response = stream.get_final_message()
    return next(b.text for b in response.content if b.type == "text")


def critique(task: str, draft: str) -> Critique:
    """Evaluate the draft and return a structured critique."""
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system="You are an exacting editor with very high standards. Score 1-10, "
        "where 10 means you cannot find a single meaningful improvement. Most "
        "competent drafts score 6-8. Set approved=true only for drafts scoring 9 "
        "or higher, and never approve without genuinely hunting for weaknesses "
        "first: accuracy, clarity, structure, completeness, constraint compliance, "
        "and tone. When you reject, every issue must be concrete and paired with "
        "a specific, actionable fix.",
        messages=[
            {
                "role": "user",
                "content": f"<task>\n{task}\n</task>\n\n<draft>\n{draft}\n</draft>\n\n"
                "Critique this draft.",
            }
        ],
        output_format=Critique,
    )
    return response.parsed_output


def revise(task: str, draft: str, review: Critique) -> str:
    """Rewrite the draft using the critic's feedback."""
    feedback = "\n".join(
        f"- Issue: {issue}\n  Fix: {fix}"
        for issue, fix in zip(review.issues, review.suggestions)
    )
    with client.messages.stream(
        model=MODEL,
        max_tokens=64000,
        thinking={"type": "adaptive"},
        system="You are a skilled writer revising your own work. Apply every piece "
        "of feedback while preserving what already works. Output only the revised "
        "draft — no preamble or change log.",
        messages=[
            {
                "role": "user",
                "content": f"<task>\n{task}\n</task>\n\n<draft>\n{draft}\n</draft>\n\n"
                f"<feedback>\n{feedback}\n</feedback>\n\nRevise the draft.",
            }
        ],
    ) as stream:
        response = stream.get_final_message()
    return next(b.text for b in response.content if b.type == "text")


def run(task: str) -> str:
    """Run the full reflection loop and return the final draft."""
    print(f"Task: {task}\n")

    print("[1/3] Generating first draft...")
    draft = generate(task)

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"[2/3] Critiquing draft (iteration {iteration}/{MAX_ITERATIONS})...")
        review = critique(task, draft)
        print(f"      Score: {review.score}/10 — {'approved' if review.approved else 'needs revision'}")

        if review.approved:
            break

        for issue in review.issues:
            print(f"      Issue: {issue}")

        print("[3/3] Revising draft...")
        draft = revise(task, draft, review)
    else:
        print(f"      Max iterations ({MAX_ITERATIONS}) reached — returning latest draft.")

    return draft


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "Write a short blog post about why unit tests matter."
    final = run(task)
    print("\n" + "=" * 60)
    print("FINAL DRAFT")
    print("=" * 60)
    print(final)
