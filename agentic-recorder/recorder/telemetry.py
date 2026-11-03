"""
Telemetry — observability for the whole pipeline.

Two layers:
  1. LangSmith tracing (if LANGSMITH_API_KEY is set): the Claude client is
     wrapped, and each phase function is decorated with @traceable, so one trace
     shows the entire explore→script→verify→film→QA pipeline.
  2. A local cost/latency ledger that every phase reports into; the app shows
     it live and the orchestrator writes it to metrics.json for the article.
"""

import os
import time
from dataclasses import asdict, dataclass, field

import anthropic
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-opus-4-8")

# $ per million tokens: (input, output)
LLM_PRICES = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
JEV_PRICE_IN = 0.042  # $/M input tokens; Jev output is free

LANGSMITH = bool(os.environ.get("LANGSMITH_API_KEY"))
if LANGSMITH:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "agentic-recorder")
    from langsmith import traceable
    from langsmith.wrappers import wrap_anthropic

    def make_client() -> anthropic.Anthropic:
        return wrap_anthropic(anthropic.Anthropic())
else:
    def traceable(**_kwargs):  # no-op decorator
        return lambda fn: fn

    def make_client() -> anthropic.Anthropic:
        return anthropic.Anthropic()


def llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = LLM_PRICES.get(model, (5.00, 25.00))
    return input_tokens / 1e6 * p_in + output_tokens / 1e6 * p_out


@dataclass
class PhaseRecord:
    phase: str
    latency_s: float = 0.0
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cost: float = 0.0
    jev_calls: int = 0
    jev_input_tokens: int = 0
    jev_cost: float = 0.0
    notes: str = ""


class Ledger:
    """Collects per-phase spend; thread-safe enough for our one-worker app."""

    def __init__(self, on_update=None):
        self.records: dict[str, PhaseRecord] = {}
        self.on_update = on_update or (lambda ledger: None)

    def phase(self, name: str) -> PhaseRecord:
        if name not in self.records:
            self.records[name] = PhaseRecord(phase=name)
        return self.records[name]

    def add_llm(self, phase: str, usage, model: str = CHAT_MODEL):
        r = self.phase(phase)
        r.llm_calls += 1
        r.llm_input_tokens += usage.input_tokens
        r.llm_output_tokens += usage.output_tokens
        r.llm_cost += llm_cost(model, usage.input_tokens, usage.output_tokens)
        self.on_update(self)

    def add_jev(self, phase: str, input_tokens: int):
        r = self.phase(phase)
        r.jev_calls += 1
        r.jev_input_tokens += input_tokens
        r.jev_cost += input_tokens / 1e6 * JEV_PRICE_IN
        self.on_update(self)

    def add_latency(self, phase: str, seconds: float):
        self.phase(phase).latency_s += seconds
        self.on_update(self)

    def totals(self) -> dict:
        return {
            "llm_cost": sum(r.llm_cost for r in self.records.values()),
            "jev_cost": sum(r.jev_cost for r in self.records.values()),
            "llm_calls": sum(r.llm_calls for r in self.records.values()),
            "jev_calls": sum(r.jev_calls for r in self.records.values()),
            "wall_clock_s": sum(r.latency_s for r in self.records.values()),
        }

    def to_dict(self) -> dict:
        return {"phases": [asdict(r) for r in self.records.values()],
                "totals": self.totals()}


class timed:
    """Context manager: adds elapsed wall-clock to a ledger phase."""

    def __init__(self, ledger: Ledger, phase: str):
        self.ledger, self.phase = ledger, phase

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ledger.add_latency(self.phase, time.perf_counter() - self.start)
        return False
