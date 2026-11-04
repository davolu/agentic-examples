"""
Chain-of-Verification (CoVe) — Dhuliawala et al., 2023.

The bet: a model is better at verifying individual facts in isolation than at
producing a fully correct answer in one shot. So instead of trusting the first
draft, the model fact-checks itself in four steps:

  1. baseline              — answer the question normally (may hallucinate)
  2. plan_verifications    — write targeted fact-check questions about the draft
  3. execute_verifications — answer each verification question INDEPENDENTLY
  4. revise                — rewrite the answer, fixing whatever didn't hold up

Step 3 is the crux. The `mode` controls independence — the paper's key variable:
  - 'factored' : each verification answered in its OWN prompt, with NO sight of
                 the original draft. Maximum independence → best de-hallucination.
  - 'joint'    : all verifications answered together, WITH the draft in view.
                 Contaminated — the model tends to rubber-stamp its own mistakes.

We expose both so the benchmark can show that independence is what makes CoVe work.
"""

import asyncio
import os

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
client = AsyncAnthropic()


async def _ask(system: str, user: str, max_tokens: int = 500) -> str:
    r = await client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    return next((b.text for b in r.content if b.type == "text"), "")


async def baseline(question: str) -> str:
    """Step 1 — the ordinary single-pass answer."""
    return await _ask("Answer the question concisely and directly.", question)


async def plan_verifications(question: str, draft: str) -> list[str]:
    """Step 2 — generate independent fact-check questions about the draft."""
    out = await _ask(
        "You are fact-checking a draft answer. Write 2-4 short, specific "
        "verification questions that would independently check the factual "
        "claims in the draft. Each question must stand alone (no pronouns "
        "referring to the draft). Output one question per line, nothing else.",
        f"<question>{question}</question>\n<draft_answer>{draft}</draft_answer>")
    return [l.strip(" -•\t") for l in out.splitlines() if l.strip()][:4]


async def _answer_one(vq: str) -> str:
    # Independent: this prompt never sees the draft or the other questions.
    return await _ask("Answer this factual question briefly and accurately. "
                      "If it rests on a false premise, say so.", vq)


async def execute_verifications(vqs: list[str], draft: str, mode: str) -> list[dict]:
    """Step 3 — answer the verification questions. mode = factored | joint."""
    if mode == "factored":
        answers = await asyncio.gather(*(_answer_one(vq) for vq in vqs))
        return [{"q": q, "a": a} for q, a in zip(vqs, answers)]

    # joint (contaminated): one prompt, sees the draft and all questions at once.
    joined = "\n".join(f"{i+1}. {q}" for i, q in enumerate(vqs))
    out = await _ask(
        "Answer each verification question. Here is the draft answer you are "
        "checking for context.",
        f"<draft_answer>{draft}</draft_answer>\n<questions>\n{joined}\n</questions>")
    return [{"q": q, "a": out} for q in vqs]  # shared blob — intentionally weaker


async def revise(question: str, draft: str, checks: list[dict]) -> str:
    """Step 4 — produce the final verified answer."""
    ev = "\n".join(f"Q: {c['q']}\nA: {c['a']}" for c in checks)
    return await _ask(
        "You wrote a draft answer, then independently fact-checked it. Using the "
        "verification results, produce a corrected final answer. If a check "
        "contradicts the draft, trust the check. If the question had a false "
        "premise, reject it. Be concise.",
        f"<question>{question}</question>\n<draft>{draft}</draft>\n"
        f"<verifications>\n{ev}\n</verifications>")


async def cove(question: str, mode: str = "factored") -> dict:
    """Full CoVe. Returns the baseline, the verification trace, and the final."""
    draft = await baseline(question)
    vqs = await plan_verifications(question, draft)
    checks = await execute_verifications(vqs, draft, mode)
    final = await revise(question, draft, checks)
    return {"baseline": draft, "verifications": checks, "final": final, "mode": mode}
