"""Structured JSON-Lines request logging (Task 12), ELK-style: one JSON object per line.

Every text field is passed through the SAME `mask_pii` used on model input before it is written,
so a fixed-format phone number never reaches disk in the clear (defence in depth: callers already
pass masked text, and the logger masks again).
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import config
from app.guardrails import mask_pii

_lock = threading.Lock()
_log_path: Path = config.LOG_PATH


def set_log_path(path: Path) -> None:
    """Redirect logs (used by tests)."""
    global _log_path
    _log_path = Path(path)


def get_log_path() -> Path:
    return _log_path


def new_trace_id() -> str:
    return uuid.uuid4().hex


def _mask_value(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii(value)[0]
    if isinstance(value, list):
        return [_mask_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _mask_value(v) for k, v in value.items()}
    return value


def log_event(event: str, trace_id: str, **fields: Any) -> dict:
    """Append one JSON line: {timestamp, level, event, trace_id, ...fields} with PII masked."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "level": fields.pop("level", "INFO"),
        "service": "practo-support-agent",
        "event": event,
        "trace_id": trace_id,
        **_mask_value(fields),
    }
    line = json.dumps(entry, ensure_ascii=False, default=str)
    with _lock:
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        with _log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return entry
