"""Throttled, newline-based progress for batch-job logs and a live status file."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time


def _duration(seconds):
    if seconds is None:
        return "estimating"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class EvaluationProgress:
    """Report phase-local ETA without pretending to know later phases' cost.

    Each log update is a flushed line so ``tail -F`` works without a terminal.
    Updates never draw random numbers or touch model/sample state.
    """

    def __init__(self, output_dir, interval=15.0, enabled=True):
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.enabled = enabled
        self.started = time.monotonic()
        self.stage_started = self.started
        self.last_update = self.started
        self.stage = "Starting evaluation"
        self.total = None
        self.completed = 0
        self.unit = "items"
        self.status = "running"
        self.error = None

    def __enter__(self):
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.status = "completed"
            self.stage = "Evaluation complete"
            self.total = self.completed = 1
            self.unit = "evaluation"
        else:
            self.status = "failed"
            self.error = str(exc_value) or exc_type.__name__
        self._emit(force=True)
        return False

    def start(self, stage, total=None, unit="items"):
        self.stage, self.total, self.unit = stage, total, unit
        self.completed = 0
        self.stage_started = time.monotonic()
        self._emit(force=True)

    def update(self, completed):
        if self.total is None or not 0 <= completed <= self.total:
            raise ValueError("Progress must be within the current phase's total.")
        self.completed = completed
        self._emit(force=completed == self.total)

    def _emit(self, force=False):
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_update < self.interval:
            return
        elapsed = now - self.stage_started
        fraction = self.completed / self.total if self.total else None
        eta = elapsed * (self.total - self.completed) / self.completed if self.total and self.completed else None
        if self.status != "running":
            eta = 0.0 if self.status == "completed" else None
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = {
            "status": self.status, "stage": self.stage, "updated_at_utc": timestamp,
            "completed": self.completed, "total": self.total, "unit": self.unit,
            "stage_percent": 100 * fraction if fraction is not None else None,
            "elapsed_seconds": now - self.started, "stage_elapsed_seconds": elapsed,
            "stage_eta_seconds": eta, "error": self.error,
        }
        temporary = self.output_dir / "progress.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.output_dir / "progress.json")
        if fraction is not None:
            filled = int(20 * fraction)
            detail = f"[{'#' * filled}{'-' * (20 - filled)}] {100 * fraction:5.1f}% {self.completed}/{self.total} {self.unit}"
        else:
            detail = "working"
        timing = f"elapsed {_duration(elapsed)} | phase ETA {_duration(eta)}"
        if self.status == "completed":
            timing = f"total elapsed {_duration(now - self.started)}"
        print(f"[{timestamp}] {self.stage}: {detail} | {timing}"
              + (f" | FAILED: {self.error}" if self.error else ""), flush=True)
        self.last_update = now
