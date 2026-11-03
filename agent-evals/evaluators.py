"""
Evaluators — three kinds, because one size doesn't fit an agent's output.

  1. EXACT-MATCH for the structured fields that have a ground truth:
     category (==), is_refund (==), urgency (within ±1 — humans disagree by a
     notch, so a tolerance is honest).
  2. LLM-AS-JUDGE for the free-text reply, which has no single right answer:
     Claude scores it against the labeled reply gist. We also CALIBRATE the
     judge against a few obvious pass/fail cases so we trust its scores.
  3. JEV JUDGE (optional): the same reply pass/fail as a Jev Noul — an eval
     dimension made nearly free with a decision model.
"""

import os

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-opus-4-8")
JEV_LIVE = bool(os.environ.get("TYPESAFE_API_KEY"))
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
if JEV_LIVE:
    from typesafe_sdk import Noul, TypeSafeClient

client = anthropic.Anthropic()


# ---- 1. exact-match --------------------------------------------------------

def exact_match(pred, case) -> dict:
    return {
        "category_ok": pred.category == case["category"],
        "is_refund_ok": pred.is_refund == case["is_refund"],
        "urgency_ok": abs(pred.urgency - case["urgency"]) <= 1,
    }


# ---- 2. LLM-as-judge -------------------------------------------------------

class Judgment(BaseModel):
    acceptable: bool = Field(description="true if the reply is helpful and on-point")
    reason: str = Field(description="one short sentence")


JUDGE_SYSTEM = """You are grading a support agent's one-line reply to a customer
email. You are given the email, a gist of what an acceptable reply should do,
and the agent's actual reply. Mark acceptable=true if the reply addresses the
customer's need in the spirit of the gist (wording may differ). Mark false if
it misunderstands the request, acts on the wrong intent, or is unhelpful."""


def llm_judge(email, reply, gist, ledger=None) -> Judgment:
    response = client.messages.parse(
        model=MODEL, max_tokens=400, system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content":
                   f"<email>{email}</email>\n<acceptable_reply_gist>{gist}"
                   f"</acceptable_reply_gist>\n<agent_reply>{reply}</agent_reply>\n\n"
                   "Grade the reply."}],
        output_format=Judgment)
    if ledger is not None:
        ledger["llm_in"] += response.usage.input_tokens
        ledger["llm_out"] += response.usage.output_tokens
    return response.parsed_output


# A tiny calibration set: cases where the correct verdict is obvious. If the
# judge disagrees with these, don't trust its scores on the real data.
CALIBRATION = [
    {"email": "I was charged twice, refund me.",
     "gist": "confirm the duplicate charge will be refunded",
     "reply": "I've confirmed the duplicate charge and a refund is on its way.",
     "expected": True},
    {"email": "I was charged twice, refund me.",
     "gist": "confirm the duplicate charge will be refunded",
     "reply": "Our Enterprise plan starts at $99/month — want a demo?",
     "expected": False},
    {"email": "The API is down, checkout is broken.",
     "gist": "treat as urgent outage and investigate",
     "reply": "Thanks for the kind words!",
     "expected": False},
]


def calibrate_judge() -> dict:
    agree = sum(1 for c in CALIBRATION
                if llm_judge(c["email"], c["reply"], c["gist"]).acceptable == c["expected"])
    return {"agreement": agree, "total": len(CALIBRATION),
            "trustworthy": agree == len(CALIBRATION)}


# ---- 3. Jev judge (optional) -----------------------------------------------

def jev_judge(email, reply, gist, ledger=None):
    if not JEV_LIVE:
        return None
    with TypeSafeClient(model=JEV_MODEL) as c:
        r = c.system_one(
            state={"customer_email": email, "acceptable_reply_gist": gist,
                   "agent_reply": reply},
            questions={"acceptable": Noul(
                instructions="The `agent_reply` helpfully addresses the "
                "`customer_email` in the spirit of `acceptable_reply_gist`")})
    p = r.nouls["acceptable"].noul
    usage = getattr(r, "usage", None)
    if ledger is not None and usage:
        ledger["jev_in"] += usage.input_tokens
    return p > 0.5
