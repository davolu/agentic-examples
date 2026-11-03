"""
Orchestrator — the deep-agent core.

Maintains the explicit plan (todos the UI renders live), the job workspace
(phases communicate through files, not one giant context), and the phase
sequence:

  memory gate → [explore] → script ⇄ critique (reflection, ≤2 revisions)
      → verify ⇄ rescript (Reflexion, ≤3 trials) → film → Jev frame QA
      → [vision review] → [one refilm] → save memory + metrics.json
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import critic, explorer, memory, qa, scriptwriter, verifier
from .filmer import Filmer
from .telemetry import Ledger, timed, traceable

JOBS_DIR = Path(__file__).resolve().parent.parent / "jobs"

PLAN = [
    ("memory", "Check site memory (Jev gate)"),
    ("explore", "Explore the site (MCP sub-agent)"),
    ("script", "Write the shot list"),
    ("critique", "Reflection: critic reviews the script"),
    ("verify", "Reflexion: dry-run every step"),
    ("film", "Film with animated cursor"),
    ("qa", "Jev grades every frame"),
    ("done", "Save memory + metrics"),
]


@dataclass
class Job:
    url: str
    brief: str
    duration_s: int = 30
    workspace: Path = None
    status: dict = field(default_factory=dict)   # phase -> pending|active|done|skipped
    log: list = field(default_factory=list)


class Orchestrator:
    def __init__(self, on_event=None):
        """on_event(kind, payload) — UI callback: 'todo', 'log', 'ledger'."""
        self.on_event = on_event or (lambda kind, payload: None)
        self.ledger = Ledger(on_update=lambda l: self.on_event("ledger", l.to_dict()))

    def _log(self, message: str):
        self.on_event("log", message)

    def _todo(self, job: Job, phase: str, state: str):
        job.status[phase] = state
        self.on_event("todo", dict(job.status))

    @traceable(name="record_demo", run_type="chain")
    def run(self, url: str, brief: str, duration_s: int = 30) -> dict:
        job = Job(url=url, brief=brief, duration_s=duration_s)
        job.workspace = JOBS_DIR / time.strftime("%Y%m%d_%H%M%S")
        job.workspace.mkdir(parents=True, exist_ok=True)
        for phase, _ in PLAN:
            self._todo(job, phase, "pending")

        # ---- 1. memory gate (Jev) -----------------------------------------
        self._todo(job, "memory", "active")
        notes, skipped_explore, gate_p = None, False, 0.0
        profile = memory.load_profile(url)
        if profile:
            with timed(self.ledger, "memory_gate"):
                skipped_explore, gate_p = memory.profile_sufficient(
                    profile, brief, self.ledger
                )
            if skipped_explore:
                notes = profile["notes"]
                self._log(f"🧠 Memory gate: profile sufficient (p={gate_p:.2f}) — "
                          "skipping exploration")
            else:
                self._log(f"🧠 Memory gate: profile insufficient (p={gate_p:.2f}) — "
                          "re-exploring")
        else:
            self._log("🧠 No site memory for this domain — exploring fresh")
        self._todo(job, "memory", "done")

        # ---- 2. explore (MCP sub-agent) -----------------------------------
        if notes is None:
            self._todo(job, "explore", "active")
            self._log("🔍 Explorer sub-agent launching (Playwright MCP)...")
            with timed(self.ledger, "explore"):
                notes = explorer.explore(url, brief, self.ledger)
            (job.workspace / "notes.md").write_text(notes)
            self._log(f"🔍 Exploration done — notes.md ({len(notes)} chars)")
            self._todo(job, "explore", "done")
        else:
            self._todo(job, "explore", "skipped")
        (job.workspace / "notes.md").write_text(notes)

        lessons = list(profile["lessons"]) if profile else []

        # ---- 3+4. script ⇄ critique (reflection) ---------------------------
        self._todo(job, "script", "active")
        critique_text = ""
        shotlist = None
        for revision in range(3):
            with timed(self.ledger, "script"):
                shotlist = scriptwriter.write_script(
                    brief, url, duration_s, notes, lessons, critique_text, self.ledger
                )
            est = scriptwriter.estimate_seconds(shotlist)
            self._log(f"✍️ Script draft {revision + 1}: {len(shotlist.steps)} steps, "
                      f"~{est:.0f}s")
            self._todo(job, "script", "done")
            self._todo(job, "critique", "active")
            with timed(self.ledger, "critique"):
                review = critic.critique_script(
                    brief, duration_s, shotlist, notes, self.ledger
                )
            if review.approved:
                self._log(f"🎭 Critic: approved ({review.score}/10)")
                break
            critique_text = "\n".join(
                f"- {issue} FIX: {fix}"
                for issue, fix in zip(review.issues, review.suggestions)
            )
            self._log(f"🎭 Critic: {review.score}/10 — revising "
                      f"({len(review.issues)} issue(s))")
        self._todo(job, "critique", "done")

        # ---- 5. verify ⇄ rescript (Reflexion) ------------------------------
        # Invariant: whatever reaches the filmer has passed a dry-run. If the
        # trial cap is hit, we PRUNE the failing steps and verify the remainder
        # rather than gamble the filming run on unverified selectors.
        self._todo(job, "verify", "active")
        trials = 0
        for trial in range(4):
            trials = trial + 1
            with timed(self.ledger, "verify"):
                failures = verifier.dry_run(shotlist)
            if not failures:
                self._log(f"✅ Verification passed (trial {trials})")
                break
            self._log(f"❌ Verification trial {trials}: {len(failures)} step(s) failed")
            if trial < 3:
                with timed(self.ledger, "verify"):
                    lesson = verifier.reflect(brief, shotlist, failures, self.ledger)
                lessons.append(lesson)
                self._log(f"📝 Lesson: {lesson[:140]}")
                with timed(self.ledger, "script"):
                    shotlist = scriptwriter.write_script(
                        brief, url, duration_s, notes, lessons, "", self.ledger
                    )
            else:
                bad = {f["step_index"] for f in failures}
                shotlist.steps = [
                    s for i, s in enumerate(shotlist.steps) if i not in bad
                ]
                self._log(f"✂️ Trial cap hit — pruned {len(bad)} unverifiable step(s)")
        (job.workspace / "shotlist.json").write_text(shotlist.model_dump_json(indent=1))

        self._todo(job, "verify", "done")

        # ---- 6. film + 7. QA (with one refilm allowed) ----------------------
        film_result, report, vision = None, None, {"verdict": "pass", "reason": ""}
        for attempt in range(2):
            self._todo(job, "film", "active")
            self._log("🎬 Filming..." if attempt == 0 else "🎬 Refilming...")
            with timed(self.ledger, "film"):
                film_result = Filmer(job.workspace).film(
                    [s.model_dump() for s in shotlist.steps]
                )
            self._todo(job, "film", "done")

            self._todo(job, "qa", "active")
            with timed(self.ledger, "qa"):
                report = qa.grade_frames(
                    Path(film_result["frames_dir"]), brief, self.ledger
                )
            self._log(f"🔬 Jev graded {report['frames_graded']} frames — "
                      f"{len(report['flagged'])} flagged")
            if report["flagged"]:
                with timed(self.ledger, "qa"):
                    vision = qa.vision_review(
                        report["flagged"], Path(film_result["frames_dir"]),
                        brief, self.ledger,
                    )
                self._log(f"👁️ Vision review: {vision['verdict']} — {vision['reason']}")
                if vision["verdict"] == "refilm" and attempt == 0:
                    continue
            break
        self._todo(job, "qa", "done")

        # ---- 8. persist memory + metrics ------------------------------------
        self._todo(job, "done", "active")
        memory.save_profile(url, notes, lessons)
        metrics = {
            "url": url, "brief": brief, "duration_target_s": duration_s,
            "explore_skipped_by_memory_gate": skipped_explore,
            "memory_gate_probability": gate_p,
            "verify_trials": trials,
            "frames_graded": report["frames_graded"],
            "frames_flagged": len(report["flagged"]),
            "vision_verdict": vision["verdict"],
            "video": film_result["mp4"] or film_result["webm"],
            "ledger": self.ledger.to_dict(),
        }
        (job.workspace / "metrics.json").write_text(json.dumps(metrics, indent=1))
        self._todo(job, "done", "done")
        totals = self.ledger.totals()
        try:  # log a repo-relative path — full home paths don't belong in UIs
            video_display = str(Path(metrics["video"]).relative_to(JOBS_DIR.parent))
        except ValueError:
            video_display = Path(metrics["video"]).name
        self._log(f"🏁 Done — {video_display}")
        self._log(f"💰 LLM ${totals['llm_cost']:.4f} ({totals['llm_calls']} calls) · "
                  f"Jev ${totals['jev_cost']:.6f} ({totals['jev_calls']} calls)")
        return metrics
