"""Append-only JSONL audit trail, one file per pipeline run."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from core.config import AUDIT_DIR


def _log_path(run_id: str):
    return AUDIT_DIR / f"{run_id}.jsonl"


def log_event(run_id: str, agent: str, action: str, **kwargs) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "agent": agent,
        "action": action,
        **kwargs,
    }
    with open(_log_path(run_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def get_logs(run_id: str) -> list[dict]:
    path = _log_path(run_id)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
