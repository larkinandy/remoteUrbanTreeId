"""Manage resumable Earth Engine export submissions.

The helpers isolate transient task throttling and JSONL bookkeeping from the
scientific definition of the exported data product.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any


ACTIVE_TASK_STATES = {"READY", "RUNNING"}


def sanitize_name(value: object, max_len: int = 90) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "")).strip("_")
    return cleaned[:max_len] or "unknown"


def load_submitted_keys(manifest_path: Path) -> set[str]:
    submitted: set[str] = set()
    if not manifest_path.exists():
        return submitted
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                key = json.loads(line).get("key")
                if key:
                    submitted.add(str(key))
    return submitted


def append_manifest(manifest_path: Path, record: dict[str, object]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def completed_output_exists(completed_dir: Path | None, export_prefix: str) -> bool:
    return bool(completed_dir and completed_dir.exists() and any(completed_dir.glob(f"{export_prefix}*.csv")))


def task_state(task: object) -> str:
    if isinstance(task, dict):
        metadata = task.get("metadata", {})
        return str(task.get("state") or (metadata.get("state") if isinstance(metadata, dict) else "") or "").upper()
    return str(task.status().get("state") or "").upper()  # type: ignore[attr-defined]


class TaskCapacityGate:
    """Throttle submissions to a configured number of active Earth Engine tasks."""

    def __init__(self, ee_module: Any, max_active_tasks: int, poll_seconds: int, task_list_limit: int):
        self.ee = ee_module
        self.max_active_tasks = int(max_active_tasks)
        self.poll_seconds = int(poll_seconds)
        self.task_list_limit = int(task_list_limit)
        self.cached_slots = 0

    def acquire(self) -> None:
        if self.cached_slots > 0:
            self.cached_slots -= 1
            return
        while True:
            available = self.refresh_available_slots()
            if available > 0:
                self.cached_slots = available - 1
                return
            print(f"Waiting {self.poll_seconds}s for an Earth Engine task slot...", flush=True)
            time.sleep(self.poll_seconds)

    def refresh_available_slots(self) -> int:
        try:
            tasks = self.ee.data.getTaskList()
        except Exception:
            tasks = self.ee.batch.Task.list()
        inspected = tasks if self.task_list_limit <= 0 else tasks[: self.task_list_limit]
        active = sum(task_state(task) in ACTIVE_TASK_STATES for task in inspected)
        available = max(0, self.max_active_tasks - active)
        print(f"Earth Engine capacity: {available} slot(s); {active}/{self.max_active_tasks} active", flush=True)
        return available
