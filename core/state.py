"""Session state and request-scoped helpers for DiGiTerra. Used by app and blueprints.

Storage is per process and contains whole DataFrames and fitted models, so it is
bounded and thread-safe: entries are evicted oldest-first once the cap is reached,
and every access takes a lock because training runs in a background thread while
the browser keeps polling.
"""

import os
import threading
from collections import OrderedDict

from flask import g, request

SESSION_COOKIE_NAME = "digiterra_session"

# One entry holds an uploaded frame, a fitted model and its scalers. Keep the cap
# small enough that a stream of cookie-less requests cannot exhaust memory.
MAX_SESSIONS = int(os.environ.get("DIGITERRA_MAX_SESSIONS", "32") or 32)

# The per-frame cap bounds one upload; this bounds all of them together. A client
# that rotates cookies could otherwise hold MAX_SESSIONS frames at the per-frame
# limit. 0 disables the budget. The desktop launcher runs with it disabled.
MAX_TOTAL_FRAME_MB = float(os.environ.get("DIGITERRA_MAX_TOTAL_FRAME_MB", "1024") or 0)
DATA_BYTES_KEY = "_data_bytes"

_STORE: "OrderedDict[str, dict]" = OrderedDict()
_LOCK = threading.RLock()

# Sessions with a run in flight are never evicted. Set by core.jobs at import so
# this module does not depend on it.
_active_sessions_provider = None


def set_active_sessions_provider(fn) -> None:
    global _active_sessions_provider
    _active_sessions_provider = fn


def _active() -> set:
    try:
        return set(_active_sessions_provider()) if _active_sessions_provider else set()
    except Exception:
        return set()


class MemoryBudgetExceeded(Exception):
    """The frame would push held data past MAX_TOTAL_FRAME_MB after every eviction."""


def frame_bytes(df) -> int:
    try:
        return int(df.memory_usage(deep=True).sum())
    except Exception:
        return 0


def total_data_bytes(exclude: str | None = None) -> int:
    with _LOCK:
        return sum(int(st.get(DATA_BYTES_KEY, 0) or 0) for sid, st in _STORE.items() if sid != exclude)


def _evict_one(protect: set) -> bool:
    """Drop the oldest session that is neither protected nor running a job."""
    for sid in list(_STORE):
        if sid in protect:
            continue
        del _STORE[sid]
        return True
    return False


def admit_frame(session_id: str, df) -> int:
    """Record df as session_id's data, evicting idle sessions to stay within the budget.

    Returns the byte size recorded. Raises MemoryBudgetExceeded when the frame
    alone, or the frame plus data held by sessions with running jobs, is over
    the budget.
    """
    nbytes = frame_bytes(df)
    budget = int(MAX_TOTAL_FRAME_MB * 1024 * 1024) if MAX_TOTAL_FRAME_MB > 0 else 0
    with _LOCK:
        store = get_session_storage(session_id)
        if budget:
            protect = _active() | {session_id}
            while total_data_bytes(exclude=session_id) + nbytes > budget:
                if not _evict_one(protect):
                    raise MemoryBudgetExceeded(
                        f"The server is holding its maximum of {MAX_TOTAL_FRAME_MB:.0f} MB of uploaded data. "
                        "Try again shortly."
                    )
        store["data"] = df
        store[DATA_BYTES_KEY] = nbytes
        return nbytes


def get_session_id() -> str:
    """Return the current session ID (from cookie or request context)."""
    sid = getattr(g, "session_id", None)
    if sid:
        return sid
    return request.cookies.get(SESSION_COOKIE_NAME) or "default"


def get_session_storage(session_id: str | None = None) -> dict:
    """Get or create session-scoped storage dict."""
    sid = session_id or get_session_id()
    with _LOCK:
        store = _STORE.get(sid)
        if store is None:
            store = {}
            _STORE[sid] = store
            protect = _active() | {sid}
            while len(_STORE) > MAX_SESSIONS:
                if not _evict_one(protect):
                    break
        else:
            _STORE.move_to_end(sid)
        return store


def clear_session_storage(session_id: str) -> None:
    """Drop one session's storage."""
    with _LOCK:
        _STORE.pop(session_id, None)


def session_count() -> int:
    """Number of sessions currently held."""
    with _LOCK:
        return len(_STORE)


class _StoreProxy:
    """Backwards-compatible stand-in for the old module-level SESSION_STORE dict."""

    def __len__(self):
        with _LOCK:
            return len(_STORE)

    def __contains__(self, key):
        with _LOCK:
            return key in _STORE

    def __getitem__(self, key):
        with _LOCK:
            return _STORE[key]

    def keys(self):
        with _LOCK:
            return list(_STORE.keys())

    def setdefault(self, key, default=None):
        return get_session_storage(key)

    def clear(self):
        with _LOCK:
            _STORE.clear()


SESSION_STORE = _StoreProxy()
