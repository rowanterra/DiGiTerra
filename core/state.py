"""Session state and request-scoped helpers for DiGiTerra. Used by app and blueprints.

A session holds a whole DataFrame and a fitted model, so the store is bounded and
thread-safe: entries are evicted oldest-first once the cap is reached, and every
access takes a lock because training runs in a background thread while the browser
keeps polling.

Where those entries live is core.session_store's decision. In memory, which is
what the desktop uses and what this application has always done, or on disk, one
directory per session, which lets more than one process serve a deployment and
lets a session outlive a restart. The dictionary a caller receives is the same
either way: it is a write-through mapping that remembers it has changed, and
`flush` is what puts a change where another process can see it. Flushing on every
key would rewrite a parquet file for a one-word change, so it happens at the end
of a request and at the end of a run instead.
"""

import logging
import os
import threading
from collections import OrderedDict

from flask import g, request

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "digiterra_session"

# One entry holds an uploaded frame, a fitted model and its scalers. Keep the cap
# small enough that a stream of cookie-less requests cannot exhaust memory.
MAX_SESSIONS = int(os.environ.get("DIGITERRA_MAX_SESSIONS", "32") or 32)

# The per-frame cap bounds one upload; this bounds all of them together. A client
# that rotates cookies could otherwise hold MAX_SESSIONS frames at the per-frame
# limit. 0 disables the budget. The desktop launcher runs with it disabled.
MAX_TOTAL_FRAME_MB = float(os.environ.get("DIGITERRA_MAX_TOTAL_FRAME_MB", "1024") or 0)
DATA_BYTES_KEY = "_data_bytes"
MODEL_BYTES_KEY = "_model_bytes"

_STORE: "OrderedDict[str, dict]" = OrderedDict()
_LOCK = threading.RLock()


class SessionMapping(dict):
    """A session's contents, which remembers that it has changed.

    Every existing call site writes into the mapping it is handed
    (`store["data"] = df`), so the mapping is what notices. `dirty` says a flush
    would write something; `flush` clears it.
    """

    def __init__(self, session_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = session_id
        self.dirty = False

    def _touch(self):
        self.dirty = True

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._touch()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._touch()

    def pop(self, *args, **kwargs):
        out = super().pop(*args, **kwargs)
        self._touch()
        return out

    def popitem(self):
        out = super().popitem()
        self._touch()
        return out

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._touch()

    def setdefault(self, key, default=None):
        missing = key not in self
        out = super().setdefault(key, default)
        if missing:
            self._touch()
        return out

    def clear(self):
        super().clear()
        self._touch()


def flush(session_id: str | None = None) -> None:
    """Write a session's contents where another process can read them.

    A no-op on the memory backend, where the mapping in this process is already
    the storage. Called at the end of a request and at the end of a run.
    """
    from core.session_store import backend, backend_name

    if backend_name() == "memory":
        return
    with _LOCK:
        ids = [session_id] if session_id else list(_STORE)
        for sid in ids:
            store = _STORE.get(sid)
            if store is None or not getattr(store, "dirty", False):
                continue
            try:
                backend().save(sid, dict(store))
                store.dirty = False
            except Exception as e:
                logger.warning("Session %s not written: %s", sid, e)


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
    """Bytes held across sessions: frames plus the serialized size of fitted objects."""
    with _LOCK:
        return sum(
            int(st.get(DATA_BYTES_KEY, 0) or 0) + int(st.get(MODEL_BYTES_KEY, 0) or 0)
            for sid, st in _STORE.items()
            if sid != exclude
        )


def _budget_bytes() -> int:
    return int(MAX_TOTAL_FRAME_MB * 1024 * 1024) if MAX_TOTAL_FRAME_MB > 0 else 0


def _make_room(session_id: str, own_after_bytes: int) -> None:
    """Evict idle sessions until this session's own total fits beside the rest, or raise.

    The caller passes what the session will hold AFTER the operation, not the
    size of the thing being added, because the two differ: a new frame replaces
    whatever the session held, while a fitted model is charged on top of the
    frame it was trained on. Excluding the whole session and then adding only the
    model was how a frame plus a model could pass the budget together.
    """
    budget = _budget_bytes()
    if not budget:
        return
    protect = _active() | {session_id}
    while total_data_bytes(exclude=session_id) + own_after_bytes > budget:
        if not _evict_one(protect):
            raise MemoryBudgetExceeded(
                f"The server is holding its maximum of {MAX_TOTAL_FRAME_MB:.0f} MB of uploaded data. Try again shortly."
            )


def reserve_frame(session_id: str, estimated_bytes: int) -> None:
    """Check the budget against an estimate before a file is parsed.

    The estimate comes from file or container metadata (core.helpers.
    estimate_decoded_bytes), so an upload that cannot fit is refused before the
    parse allocates anything. admit_frame then records the measured size.
    """
    with _LOCK:
        # A new upload replaces this session's frame and its model, so the estimate
        # is the whole of what the session will hold.
        get_session_storage(session_id)
        _make_room(session_id, int(estimated_bytes))


def admit_model(session_id: str, nbytes: int) -> None:
    """Charge a fitted model against the budget, or raise MemoryBudgetExceeded.

    Unlike a frame, a model cannot be measured before it exists, so this runs
    after the fit: it makes room by evicting idle sessions and refuses when the
    model does not fit beside what the running sessions hold. The caller drops
    the fitted objects on a refusal, so an over-budget model is not retained.
    Measuring after the fact is a weaker control than a worker with a memory
    limit, which is the remaining step.
    """
    nbytes = int(nbytes)
    with _LOCK:
        store = _STORE.get(session_id)
        if store is None:
            return
        store.pop(MODEL_BYTES_KEY, None)
        # The model is held beside the frame it was trained on, so both count.
        frame_charge = int(store.get(DATA_BYTES_KEY, 0) or 0)
        _make_room(session_id, frame_charge + nbytes)
        store[MODEL_BYTES_KEY] = nbytes


def discard_model(session_id: str) -> None:
    """Drop the fitted objects and their charge, leaving the uploaded frame."""
    with _LOCK:
        store = _STORE.get(session_id)
        if store is None:
            return
        store.pop(MODEL_BYTES_KEY, None)
        for key in ("model", "X_scaler", "y_scaler", "feature_order", "model_type"):
            store.pop(key, None)


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
    with _LOCK:
        store = get_session_storage(session_id)
        # A new frame replaces this session's model as well.
        store.pop(MODEL_BYTES_KEY, None)
        _make_room(session_id, nbytes)
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
    """Get or create session-scoped storage dict.

    On the disk backend a session this process has not seen is read from storage,
    which is how a second worker answers a request about a session the first one
    started.
    """
    from core.session_store import backend, backend_name

    sid = session_id or get_session_id()
    with _LOCK:
        store = _STORE.get(sid)
        if store is None:
            loaded = None
            if backend_name() != "memory":
                try:
                    loaded = backend().load(sid)
                except Exception as e:
                    logger.warning("Session %s not read: %s", sid, e)
            store = SessionMapping(sid, loaded or {})
            _STORE[sid] = store
            protect = _active() | {sid}
            while len(_STORE) > MAX_SESSIONS:
                if not _evict_one(protect):
                    break
        else:
            _STORE.move_to_end(sid)
        return store


def clear_session_storage(session_id: str) -> None:
    """Drop one session's storage, here and in the backend."""
    from core.session_store import backend, backend_name

    with _LOCK:
        _STORE.pop(session_id, None)
    if backend_name() != "memory":
        try:
            backend().delete(session_id)
        except Exception as e:
            logger.warning("Session %s not removed: %s", session_id, e)


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
