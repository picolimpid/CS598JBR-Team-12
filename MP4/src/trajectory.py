"""Trajectory recording for the agentic C-to-Rust pipeline.

Everything the agent sees or does is written under MP4/trajectories/project_<N>/:

  execution.log          human-readable log of the whole run (appended across resumes)
  trajectory.jsonl       one JSON event per line (LLM calls, tool calls, diffs, verifications)
  llm_messages/iter_XX.json   full chat transcript (prompts + responses + tool results) per iteration
  verification/iter_XX.json   structured compiler/test/clippy/safety results per iteration
  snapshots/iter_XX/     the Rust sources as they were at the end of each iteration
  best/                  the best-scoring Rust sources seen so far
  summary.json / summary.md   final status, per-iteration progression, budgets used

All text passes through `redact()` before hitting disk so API keys never land in the repo.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

_SECRET_PATTERNS = [
    re.compile(r"sk-or-v1-[A-Za-z0-9]{16,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}"),
]


class Trajectory:
    def __init__(self, traj_dir: Path, secrets: list[str] | None = None, echo: bool = True):
        self.dir = Path(traj_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for sub in ("llm_messages", "verification", "snapshots"):
            (self.dir / sub).mkdir(exist_ok=True)
        self.secrets = [s for s in (secrets or []) if s and len(s) >= 8]
        self.echo = echo
        self.iteration = 0
        self._log_path = self.dir / "execution.log"
        self._jsonl_path = self.dir / "trajectory.jsonl"

    # ------------------------------------------------------------------ redaction
    def add_secret(self, value: str | None) -> None:
        if value and len(value) >= 8 and value not in self.secrets:
            self.secrets.append(value)

    def redact(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        for s in self.secrets:
            text = text.replace(s, "[REDACTED]")
        for pat in _SECRET_PATTERNS:
            if pat.groups:
                text = pat.sub(lambda m: m.group(1) + "[REDACTED]", text)
            else:
                text = pat.sub("[REDACTED]", text)
        return text

    def _redact_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, dict):
            return {k: self._redact_obj(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._redact_obj(v) for v in obj]
        return obj

    # ------------------------------------------------------------------ writers
    @staticmethod
    def _now() -> str:
        return _dt.datetime.now().isoformat(timespec="seconds")

    def log(self, msg: str) -> None:
        line = f"[{self._now()}] [iter {self.iteration:02d}] {self.redact(msg)}"
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.echo:
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                print(line.encode("ascii", "replace").decode(), flush=True)

    def log_block(self, title: str, body: str, max_chars: int = 6000) -> None:
        """Log a multi-line block (diagnostics, test output, ...)."""
        body = body if len(body) <= max_chars else body[:max_chars] + f"\n... [{len(body) - max_chars} more chars in trajectory.jsonl]"
        self.log(f"{title}\n" + "\n".join("    " + ln for ln in body.splitlines()))

    def event(self, kind: str, **data: Any) -> None:
        rec = {"time": self._now(), "iteration": self.iteration, "event": kind}
        rec.update(data)
        with open(self._jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self._redact_obj(rec), ensure_ascii=False) + "\n")

    def save_json(self, rel_path: str, obj: Any) -> Path:
        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._redact_obj(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def save_text(self, rel_path: str, text: str) -> Path:
        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.redact(text), encoding="utf-8")
        return path

    def save_messages(self, iteration: int, messages: list[dict]) -> None:
        self.save_json(f"llm_messages/iter_{iteration:02d}.json", messages)

