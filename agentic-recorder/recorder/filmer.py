"""
Filmer — the deterministic artifact layer. No LLM in here.

Replays a verified shot list in a recording browser context, with the two
tricks that make agent-made videos look human:

  1. A FAKE CURSOR. Headless recordings show no OS pointer, so we inject a
     DOM overlay dot that glides between targets with eased motion and emits
     a ripple on click — smoother and more deliberate than a real mouse.
  2. CAPTION OVERLAYS. A bottom bar that fades in per step, so the video
     narrates itself without audio.

Both overlays are installed via an init script, so they survive navigations.
Output: .webm from Playwright, converted to .mp4 + per-second frames via ffmpeg.
"""

import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

VIEWPORT = {"width": 1280, "height": 720}

OVERLAY_JS = """
(() => {
  function install() {
    if (!document.body || document.getElementById('__agent_cursor')) return;
    const c = document.createElement('div');
    c.id = '__agent_cursor';
    c.style.cssText = `
      position: fixed; left: 80px; top: 80px; width: 22px; height: 22px;
      border-radius: 50%; background: rgba(255, 64, 64, 0.85);
      border: 3px solid white; box-shadow: 0 0 12px rgba(0,0,0,0.45);
      z-index: 2147483647; pointer-events: none;
      transform: translate(-50%, -50%);
    `;
    const cap = document.createElement('div');
    cap.id = '__agent_caption';
    cap.style.cssText = `
      position: fixed; bottom: 42px; left: 50%; transform: translateX(-50%);
      max-width: 72%; padding: 14px 26px; border-radius: 12px;
      background: rgba(17, 17, 17, 0.88); color: #fff;
      font: 600 22px/1.35 -apple-system, 'Segoe UI', Roboto, sans-serif;
      text-align: center; opacity: 0; transition: opacity 0.4s ease;
      z-index: 2147483647; pointer-events: none;
      box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    `;
    document.body.appendChild(c);
    document.body.appendChild(cap);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', install);
  } else { install(); }
})();
"""


class Filmer:
    def __init__(self, workspace: Path, headless: bool = True):
        self.workspace = Path(workspace)
        self.headless = headless
        self.skipped: list[dict] = []

    # ---- overlay helpers (all pure JS evaluation) -------------------------

    def _ensure_overlay(self, page):
        page.evaluate(OVERLAY_JS)

    def _caption(self, page, text: str):
        self._ensure_overlay(page)
        page.evaluate(
            """(text) => {
                const cap = document.getElementById('__agent_caption');
                if (!cap) return;
                if (!text) { cap.style.opacity = '0'; return; }
                cap.textContent = text;
                cap.style.opacity = '1';
            }""",
            text,
        )

    def _move_cursor(self, page, x: float, y: float, duration_ms: int = 700):
        self._ensure_overlay(page)
        page.evaluate(
            """([x, y, ms]) => {
                const c = document.getElementById('__agent_cursor');
                if (!c) return;
                c.style.transition = `left ${ms}ms cubic-bezier(.4,0,.2,1), top ${ms}ms cubic-bezier(.4,0,.2,1)`;
                c.style.left = x + 'px';
                c.style.top = y + 'px';
            }""",
            [x, y, duration_ms],
        )
        time.sleep(duration_ms / 1000 + 0.1)

    def _ripple(self, page, x: float, y: float):
        page.evaluate(
            """([x, y]) => {
                const r = document.createElement('div');
                r.style.cssText = `
                  position: fixed; left: ${x}px; top: ${y}px; width: 14px; height: 14px;
                  border-radius: 50%; border: 3px solid rgba(255,64,64,0.9);
                  transform: translate(-50%,-50%) scale(1); opacity: 1;
                  z-index: 2147483646; pointer-events: none;
                  transition: transform 0.45s ease-out, opacity 0.45s ease-out;
                `;
                document.body.appendChild(r);
                requestAnimationFrame(() => {
                  r.style.transform = 'translate(-50%,-50%) scale(4.5)';
                  r.style.opacity = '0';
                });
                setTimeout(() => r.remove(), 600);
            }""",
            [x, y],
        )
        time.sleep(0.5)

    def _target_point(self, page, selector: str):
        locator = page.locator(selector).first
        locator.wait_for(state="visible", timeout=8000)
        locator.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        box = locator.bounding_box()
        if not box:
            raise RuntimeError(f"No bounding box for selector: {selector}")
        return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    # ---- step execution ---------------------------------------------------

    def _run_step(self, page, step: dict):
        action = step["action"]
        caption = step.get("caption", "")
        dwell = float(step.get("dwell", 2.0))

        if action == "goto":
            page.goto(step["url"], wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(800)
            self._caption(page, caption)
            time.sleep(dwell)

        elif action in ("click", "hover"):
            x, y = self._target_point(page, step["selector"])
            self._caption(page, caption)
            self._move_cursor(page, x, y)
            if action == "click":
                self._ripple(page, x, y)
                page.mouse.click(x, y)
                page.wait_for_timeout(900)
            time.sleep(dwell)

        elif action == "scroll_to":
            locator = page.locator(step["selector"]).first
            locator.wait_for(state="attached", timeout=8000)
            self._caption(page, caption)
            locator.evaluate(
                "el => el.scrollIntoView({behavior: 'smooth', block: 'center'})"
            )
            page.wait_for_timeout(1200)
            time.sleep(dwell)

        elif action == "pause":
            self._caption(page, caption)
            time.sleep(dwell)

        else:
            raise ValueError(f"Unknown action: {action}")

    # ---- public API ---------------------------------------------------------

    def film(self, shotlist: list[dict]) -> dict:
        """Record the shot list. Returns paths: webm, mp4, frames dir."""
        video_dir = self.workspace / "video"
        frames_dir = self.workspace / "frames"
        video_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                viewport=VIEWPORT, record_video_dir=str(video_dir),
                record_video_size=VIEWPORT,
            )
            context.add_init_script(OVERLAY_JS)
            page = context.new_page()
            for step in shotlist:
                try:
                    self._run_step(page, step)
                except Exception as exc:
                    # Resilience: a missing beat beats a crashed film. Clear
                    # any stale caption and move on.
                    self.skipped.append({"step": step, "error": str(exc)[:200]})
                    try:
                        self._caption(page, "")
                    except Exception:
                        pass
            self._caption(page, "")
            time.sleep(0.6)
            video = page.video
            context.close()           # flushes the recording
            webm_path = Path(video.path())
            browser.close()

        result = {"webm": str(webm_path), "mp4": None, "frames_dir": str(frames_dir)}

        if shutil.which("ffmpeg"):
            mp4_path = self.workspace / "demo.mp4"
            subprocess.run(
                # -ss 0.9 trims the blank pre-load second — a defect Jev's
                # frame QA caught in the first smoke test.
                ["ffmpeg", "-y", "-ss", "0.9", "-i", str(webm_path), "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4_path)],
                capture_output=True, check=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp4_path), "-vf", "fps=1",
                 str(frames_dir / "frame_%03d.png")],
                capture_output=True, check=True,
            )
            result["mp4"] = str(mp4_path)

        return result


if __name__ == "__main__":
    # Smoke test: hardcoded mini-tour of python.org
    demo = [
        {"action": "goto", "url": "https://www.python.org",
         "caption": "This is python.org — home of the Python language", "dwell": 2.5},
        {"action": "hover", "selector": "#about a",
         "caption": "The About section covers what Python is used for", "dwell": 1.5},
        {"action": "click", "selector": "#downloads a",
         "caption": "Downloads: grab the latest release for your OS", "dwell": 2.5},
        {"action": "scroll_to", "selector": "footer",
         "caption": "Docs, community, and success stories live further down", "dwell": 2.0},
        {"action": "pause", "caption": "That's the 15-second tour!", "dwell": 2.0},
    ]
    filmer = Filmer(Path("jobs/smoke_test"))
    out = filmer.film(demo)
    print(out)
