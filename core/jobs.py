"""Admission control for training runs.

Every accepted POST to /process starts a thread that trains a model, and the
search and cross-validation paths ask for every core. Without a cap, one client
can create as many of those as it can send requests. This module holds the
count: one run at a time per browser session, and a fixed number across the
process. A request past either limit is refused before a thread exists.
"""

from __future__ import annotations

import os
import threading

# Two runs at once already saturate a small host, because each one fans out over
# every core inside scikit-learn.
MAX_CONCURRENT_TRAINING = int(os.environ.get("DIGITERRA_MAX_CONCURRENT_TRAINING", "2") or 2)

_lock = threading.Lock()
_active: dict[str, str] = {}  # storage session id -> training id


class SlotUnavailable(Exception):
    """No slot was free. `reason` is the message for the client."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def acquire(storage_session_id: str, training_id: str) -> None:
    """Claim the slot for this browser session, or raise SlotUnavailable."""
    with _lock:
        if storage_session_id in _active:
            raise SlotUnavailable("A model is already training for this session. Wait for it to finish or stop it.")
        if MAX_CONCURRENT_TRAINING and len(_active) >= MAX_CONCURRENT_TRAINING:
            raise SlotUnavailable("The server is training the maximum number of models. Try again shortly.")
        _active[storage_session_id] = training_id


def release(storage_session_id: str, training_id: str | None = None) -> None:
    """Free the slot. A stale release for a superseded run is ignored."""
    with _lock:
        current = _active.get(storage_session_id)
        if current is not None and (training_id is None or current == training_id):
            del _active[storage_session_id]


def active_sessions() -> set[str]:
    """Browser sessions with a run in flight."""
    with _lock:
        return set(_active)


def active_count() -> int:
    with _lock:
        return len(_active)
