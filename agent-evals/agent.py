"""
The agent under test — a support-email triage agent.

Per email it returns structured fields (category, urgency, is_refund) plus a
one-line reply. Two prompt versions are selectable:

  v1 — the baseline.
  v2 — a well-intentioned "improvement" that adds detail about several
       categories but, as a side effect, muddies the spam/other boundary.
       It LOOKS better. The eval suite will tell us whether it actually is.

This is the whole point of the example: you cannot tell v1 from v2 by reading
them. You need numbers.
"""

import os

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-opus-4-8")
client = anthropic.Anthropic()


class Triage(BaseModel):
    category: str = Field(description="one of: billing, technical, sales, spam, other")
    urgency: int = Field(description="1 (whenever) to 5 (drop everything)")
    is_refund: bool = Field(description="true if the customer is requesting money back")
    reply: str = Field(description="a single-sentence reply to the customer")


PROMPT_V1 = """You are a support-email triage assistant. For each email, classify
it and draft a one-line reply.

Categories:
- billing: charges, invoices, payment methods, refunds, plan changes, tax
- technical: bugs, errors, outages, login issues, setup/configuration help
- sales: pre-purchase questions, demos, pricing, discounts, trials, plan comparisons
- spam: unsolicited marketing, phishing, link-building, scams
- other: anything else (praise, jobs, legal/compliance, GDPR, docs pointers)

Set urgency 1-5 (5 = active outage or revenue-blocking). Set is_refund true only
when the customer explicitly wants money back."""

# v2 adds helpful-sounding detail — but the expanded "other" description now
# overlaps with spam (outreach/solicitation), which will cost spam accuracy.
PROMPT_V2 = """You are a support-email triage assistant. For each email, classify
it and draft a one-line reply. Be thorough and give customers the benefit of
the doubt — assume good faith and route to a human category when plausible.

Categories:
- billing: charges, invoices, payment methods, refunds, plan changes, tax
- technical: bugs, errors, outages, login issues, setup/configuration help
- sales: pre-purchase questions, demos, pricing, discounts, trials, plan comparisons
- spam: obvious scams and phishing only
- other: anything else — praise, jobs, legal/compliance, GDPR, docs pointers,
  partnership and collaboration outreach, guest-post and marketing proposals,
  and any message from someone trying to start a business conversation

Set urgency 1-5 (5 = active outage or revenue-blocking). Set is_refund true only
when the customer explicitly wants money back."""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


def triage(email: str, version: str = "v1") -> tuple[Triage, object]:
    """Return (parsed Triage, usage) for one email."""
    response = client.messages.parse(
        model=MODEL,
        max_tokens=600,
        system=PROMPTS[version],
        messages=[{"role": "user", "content": f"Triage this email:\n\n{email}"}],
        output_format=Triage,
    )
    return response.parsed_output, response.usage
