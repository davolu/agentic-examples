"""
Agentic Recorder — desktop app.

Give it a URL, say what the video is for, pick a duration, press Record.
The agent explores the page, writes and critiques a walkthrough, verifies
every step, films it with an animated cursor, QA-grades every frame, and
hands you an .mp4.

    python app.py
"""

import subprocess
import sys
import threading

import customtkinter as ctk

from recorder.orchestrator import PLAN, Orchestrator

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

STATE_COLORS = {
    "pending": ("#6b7280", "○"),
    "active": ("#fbbf24", "◐"),
    "done": ("#34d399", "●"),
    "skipped": ("#60a5fa", "⊘"),
}
ACCENT = "#ef4444"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Agentic Recorder")
        self.geometry("1060x680")
        self.video_path = None
        self._build()

    def _build(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # ---- header ---------------------------------------------------------
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="we", padx=18, pady=(16, 6))
        ctk.CTkLabel(header, text="🎬 Agentic Recorder",
                     font=("SF Pro Display", 24, "bold")).pack(side="left")
        ctk.CTkLabel(header, text="  you write the brief — the agent films it",
                     font=("SF Pro Text", 14), text_color="#9ca3af").pack(
            side="left", pady=(6, 0))

        # ---- inputs ---------------------------------------------------------
        form = ctk.CTkFrame(self, corner_radius=14)
        form.grid(row=1, column=0, columnspan=2, sticky="we", padx=18, pady=6)
        form.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(form, text="Page URL", font=("SF Pro Text", 13)).grid(
            row=0, column=0, sticky="w", padx=(16, 8), pady=(14, 4))
        self.url_var = ctk.StringVar(value="https://www.python.org")
        ctk.CTkEntry(form, textvariable=self.url_var, height=36,
                     font=("SF Pro Text", 13)).grid(
            row=0, column=1, sticky="we", pady=(14, 4))

        ctk.CTkLabel(form, text="What's the video for?", font=("SF Pro Text", 13)).grid(
            row=1, column=0, sticky="w", padx=(16, 8), pady=4)
        self.brief_var = ctk.StringVar(
            value="A 30-second tour of python.org for first-time visitors")
        ctk.CTkEntry(form, textvariable=self.brief_var, height=36,
                     font=("SF Pro Text", 13)).grid(row=1, column=1, sticky="we", pady=4)

        row3 = ctk.CTkFrame(form, fg_color="transparent")
        row3.grid(row=2, column=0, columnspan=2, sticky="we", padx=16, pady=(4, 14))
        ctk.CTkLabel(row3, text="Duration", font=("SF Pro Text", 13)).pack(side="left")
        self.duration_var = ctk.IntVar(value=30)
        ctk.CTkSlider(row3, from_=15, to=90, number_of_steps=15,
                      variable=self.duration_var, width=180).pack(side="left", padx=10)
        self.duration_label = ctk.CTkLabel(row3, text="30s", width=40,
                                           font=("SF Pro Text", 13, "bold"))
        self.duration_label.pack(side="left")
        self.duration_var.trace_add(
            "write",
            lambda *_: self.duration_label.configure(text=f"{self.duration_var.get()}s"),
        )

        self.record_btn = ctk.CTkButton(
            row3, text="●  Record", command=self.start, height=40, width=150,
            corner_radius=20, fg_color=ACCENT, hover_color="#dc2626",
            font=("SF Pro Text", 15, "bold"))
        self.record_btn.pack(side="right")
        self.open_btn = ctk.CTkButton(
            row3, text="▶  Open video", command=self.open_video, height=40,
            width=140, corner_radius=20, fg_color="#374151", hover_color="#4b5563",
            state="disabled", font=("SF Pro Text", 14))
        self.open_btn.pack(side="right", padx=10)

        # ---- plan panel -----------------------------------------------------
        plan = ctk.CTkFrame(self, corner_radius=14, width=320)
        plan.grid(row=2, column=0, sticky="nsw", padx=(18, 8), pady=(6, 16))
        ctk.CTkLabel(plan, text="AGENT PLAN", font=("SF Pro Text", 12, "bold"),
                     text_color="#9ca3af").pack(anchor="w", padx=16, pady=(14, 8))
        self.todo_labels = {}
        for phase, title in PLAN:
            label = ctk.CTkLabel(plan, text=f"○  {title}", font=("SF Pro Text", 13),
                                 text_color=STATE_COLORS["pending"][0])
            label.pack(anchor="w", padx=16, pady=3)
            self.todo_labels[phase] = (label, title)

        self.cost_label = ctk.CTkLabel(
            plan, text="LLM   $0.0000   (0 calls)\nJev   $0.000000 (0 calls)",
            font=("SF Mono", 12), text_color="#d1d5db", justify="left")
        self.cost_label.pack(anchor="w", padx=16, pady=(18, 16))

        # ---- log panel ------------------------------------------------------
        log_frame = ctk.CTkFrame(self, corner_radius=14)
        log_frame.grid(row=2, column=1, sticky="nsew", padx=(8, 18), pady=(6, 16))
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(log_frame, text="LIVE LOG", font=("SF Pro Text", 12, "bold"),
                     text_color="#9ca3af").grid(row=0, column=0, sticky="w",
                                                padx=16, pady=(14, 4))
        self.log_text = ctk.CTkTextbox(log_frame, font=("SF Mono", 12),
                                       fg_color="#111418", wrap="word")
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.log_text.configure(state="disabled")

    # ---- orchestrator events (arrive on worker thread) ----------------------

    def on_event(self, kind, payload):
        self.after(0, lambda: self._apply_event(kind, payload))

    def _apply_event(self, kind, payload):
        if kind == "log":
            self.log_text.configure(state="normal")
            self.log_text.insert("end", payload + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        elif kind == "todo":
            for phase, state in payload.items():
                label, title = self.todo_labels[phase]
                color, icon = STATE_COLORS.get(state, STATE_COLORS["pending"])
                label.configure(text=f"{icon}  {title}", text_color=color)
        elif kind == "ledger":
            totals = payload["totals"]
            self.cost_label.configure(
                text=f"LLM   ${totals['llm_cost']:.4f}   ({totals['llm_calls']} calls)"
                f"\nJev   ${totals['jev_cost']:.6f} ({totals['jev_calls']} calls)"
            )

    # ---- actions -------------------------------------------------------------

    def start(self):
        self.record_btn.configure(state="disabled", text="Recording…")
        self.open_btn.configure(state="disabled")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            orchestrator = Orchestrator(on_event=self.on_event)
            metrics = orchestrator.run(
                self.url_var.get().strip(),
                self.brief_var.get().strip(),
                int(self.duration_var.get()),
            )
            self.video_path = metrics["video"]
            self.after(0, lambda: self.open_btn.configure(
                state="normal", fg_color="#059669", hover_color="#047857"))
        except Exception as exc:  # surface errors into the log
            self.on_event("log", f"💥 ERROR: {exc}")
        finally:
            self.after(0, lambda: self.record_btn.configure(
                state="normal", text="●  Record"))

    def open_video(self):
        if self.video_path:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, self.video_path])


if __name__ == "__main__":
    app = App()
    if "--autorun" in sys.argv:  # used by the docs screenshot script
        app.geometry("1060x680+80+60")
        app.after(1500, app.start)
    app.mainloop()
