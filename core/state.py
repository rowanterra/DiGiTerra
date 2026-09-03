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

_STORE: "OrderedDict[str, dict]" = OrderedDict()
_LOCK = threading.RLock()


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
            while len(_STORE) > MAX_SESSIONS:
                _STORE.popitem(last=False)
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
