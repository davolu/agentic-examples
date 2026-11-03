"""
The work queue and the per-item task.

A realistic batch job: enrich a list of companies with a structured profile
(industry, one-line description, HQ country best-guess). Each item is one LLM
call. The task is deliberately ordinary — the interesting part is surviving a
crash partway through 25 of them.
"""

import os

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5")
client = anthropic.Anthropic()

COMPANIES = [
    "Stripe", "Shopify", "Notion", "Figma", "Databricks", "Snowflake",
    "Airtable", "Vercel", "Linear", "Retool", "Ramp", "Brex", "Plaid",
    "Twilio", "Datadog", "HashiCorp", "Confluent", "MongoDB", "Elastic",
    "Cloudflare", "Fastly", "Segment", "Amplitude", "Mixpanel", "PostHog",
]


class Profile(BaseModel):
    industry: str = Field(description="primary industry / category")
    description: str = Field(description="one concise sentence on what they do")
    hq_country: str = Field(description="best-guess HQ country")


def enrich(company: str) -> tuple[dict, int, int]:
    """Enrich one company. Returns (profile_dict, in_tokens, out_tokens)."""
    r = client.messages.parse(
        model=MODEL, max_tokens=400,
        system="You are a business-data analyst. Give a crisp, factual profile.",
        messages=[{"role": "user", "content": f"Profile the company: {company}"}],
        output_format=Profile)
    return r.parsed_output.model_dump(), r.usage.input_tokens, r.usage.output_tokens
