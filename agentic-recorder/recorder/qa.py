"""
QA — Jev grades EVERY frame of the finished video; Claude vision judges only
the flagged ones.

The economics are the point: at $0.042/M input tokens, grading one frame per
second of video costs a fraction of a cent, so exhaustive QA is affordable.
Frames Jev flags (or is unsure about) escalate to Claude vision — the same
confidence-routing pattern as the jev-agentic example, applied to video.
One refilm maximum.
"""

import base64
import os
from pathlib import Path

from pydantic import BaseModel, Field

from .telemetry import CHAT_MODEL, make_client, traceable

JEV_LIVE = bool(os.environ.get("TYPESAFE_API_KEY"))
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")

if JEV_LIVE:
    from typesafe_sdk import Image, Noul, Score, TypeSafeClient

claude = make_client()


class VisionVerdict(BaseModel):
    refilm: bool = Field(description="True only if the video is genuinely defective")
    reason: str = Field(description="One-sentence justification")

POLISH_RUBRIC = [
    "Blank, broken, or error page",
    "Messy: overlapping elements or half-loaded content",
    "Acceptable demo frame",
    "Clean, professional demo frame",
]


@traceable(name="jev_grade_frames", run_type="chain")
def grade_frames(frames_dir: Path, brief: str, ledger=None) -> dict:
    """Grade every extracted frame with Jev. Returns a QA report."""
    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    report = {"frames_graded": 0, "flagged": [], "grades": [], "jev_available": JEV_LIVE}
    if not JEV_LIVE or not frames:
        return report

    with TypeSafeClient(model=JEV_MODEL) as client:
        for frame in frames:
            response = client.system_one(
                state={"frame": Image(frame), "video_brief": brief},
                questions={
                    "broken": Noul(
                        instructions="`frame` looks broken: blank page, browser "
                        "error, or severely garbled/overlapping layout",
                    ),
                    "on_brief": Noul(
                        instructions="`frame` plausibly belongs in a demo video "
                        "for the stated `video_brief`",
                    ),
                    "polish": Score(
                        instructions="Visual quality of `frame` as one frame of "
                        "a product demo video",
                        criteria=POLISH_RUBRIC,
                    ),
                },
                extra_headers={"TypeSafe-Danger-Zone-Images": "true"},
            )
            broken = response.nouls["broken"].noul
            on_brief = response.nouls["on_brief"].noul
            polish = response.scores["polish"].score
            grade = {
                "frame": frame.name, "broken": round(broken, 3),
                "on_brief": round(on_brief, 3), "polish": round(polish, 2),
            }
            report["grades"].append(grade)
            report["frames_graded"] += 1
            if broken > 0.5 or on_brief < 0.35 or polish < 1.2:
                report["flagged"].append(grade)
            usage = getattr(response, "usage", None)
            if ledger and usage:
                ledger.add_jev("qa", usage.input_tokens)
    return report


@traceable(name="vision_review", run_type="chain")
def vision_review(flagged: list[dict], frames_dir: Path, brief: str, ledger=None) -> dict:
    """Claude vision judges Jev-flagged frames: refilm or acceptable?"""
    if not flagged:
        return {"verdict": "pass", "reason": "No frames flagged by Jev."}

    content = [
        {
            "type": "text",
            "text": f"These frames from an agent-made demo video (brief: {brief!r}) "
            f"were flagged by a cheap QA grader: {flagged}. Judge whether the "
            "video needs refilming or the flags are false alarms. IMPORTANT: "
            "you are looking at STILLS from a MOVING video — frames captured "
            "mid-scroll, mid-navigation, or mid-animation are normal. In "
            "particular, carousel/slideshow cross-fades produce double-exposed "
            "overlapping text in stills that looks perfectly normal in motion; "
            "that is a false alarm, not a defect. Order a refilm only for "
            "defects that would persist in motion: blank pages, error pages, "
            "broken layout, missing content.",
        }
    ]
    for grade in flagged[:5]:
        data = base64.standard_b64encode(
            (Path(frames_dir) / grade["frame"]).read_bytes()
        ).decode()
        content.append(
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/png", "data": data}}
        )

    response = claude.messages.parse(
        model=CHAT_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": content}],
        output_format=VisionVerdict,
    )
    if ledger:
        ledger.add_llm("qa", response.usage)
    verdict = response.parsed_output
    return {"verdict": "refilm" if verdict.refilm else "pass", "reason": verdict.reason}

