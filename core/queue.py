"""A queue for training runs that survives a restart.

Admission control refuses a run when the slots are full, which is correct for a
desktop where the person can simply wait, and wrong for a shared installation
where the answer should be that the run is queued. A refusal also loses the
request: nothing remembers that someone asked.

This module records the request instead. SQLite is the store because it is in the
standard library, it gives the atomic claim a queue needs in one statement, and it
survives the process that wrote it; a deployment across several machines would
need a server, and the security review says so. A job holds the request, not the
data: the frame is in the session store, which is where a worker in another
process reads it from.

States: queued -> running -> done or failed, and cancelled from either of the
first two. A job that was running when the process stopped is returned to the
queue on the next sweep, since nothing is producing its result any more.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"

# A job whose worker has not reported in this long is treated as lost. It is
# longer than any run this application is expected to make.
STALE_RUNNING_SECONDS = float(os.environ.get("DIGITERRA_QUEUE_STALE_SECONDS", "7200") or 7200)
MAX_QUEUE_DEPTH = int(os.environ.get("DIGITERRA_MAX_QUEUE_DEPTH", "50") or 0)

_local = threading.local()
_init_lock = threading.Lock()
_initialized: set[str] = set()


def queue_enabled() -> bool:
    """On where runs have to wait for each other, off where one person runs one thing."""
    raw = os.environ.get("DIGITERRA_QUEUE")
    if raw is not None and raw.strip() != "":
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    from core.security import is_server_profile

    return is_server_profile()


def db_path() -> Path:
    explicit = os.environ.get("DIGITERRA_QUEUE_PATH", "").strip()
    if explicit:
        return Path(explicit)
    from python_scripts import config

    return Path(config.APP_SUPPORT_DIR) / "queue.sqlite3"


def _connect() -> sqlite3.Connection:
    path = str(db_path())
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "path", None) == path:
        return conn
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # Write-ahead logging is what lets a reader and a writer work at once, which
    # is the whole point of a queue several processes share.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    _local.conn = conn
    _local.path = path
    with _init_lock:
        if path not in _initialized:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    request TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    heartbeat_at REAL,
                    worker TEXT,
                    detail TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_state_created ON jobs (state, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_session ON jobs (session_id)")
            _initialized.add(path)
    return conn


def reset_for_tests() -> None:
    """Forget the connection and the schema flag, for a test that moves the file."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    _local.conn = None
    _local.path = None
    with _init_lock:
        _initialized.clear()


class QueueFull(Exception):
    """The queue is at its depth and will not take another job."""


def enqueue(job_id: str, session_id: str, request: dict) -> dict:
    """Record a request. Raises QueueFull when the queue is at its depth."""
    conn = _connect()
    now = time.time()
    if MAX_QUEUE_DEPTH:
        depth = conn.execute("SELECT COUNT(*) FROM jobs WHERE state = ?", (QUEUED,)).fetchone()[0]
        if depth >= MAX_QUEUE_DEPTH:
            raise QueueFull(f"The queue is holding its maximum of {MAX_QUEUE_DEPTH} waiting runs.")
    conn.execute(
        "INSERT INTO jobs (id, session_id, request, state, created_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, session_id, json.dumps(request, default=str), QUEUED, now),
    )
    return {"id": job_id, "state": QUEUED, "position": position(job_id)}


def claim(worker: str, session_id: str | None = None) -> dict | None:
    """Take the oldest waiting job, atomically. None when there is nothing to take.

    The UPDATE names the row it expects to change and the state it expects to
    find, so two workers racing for the same job produce one winner and one None
    rather than two runs of the same request.
    """
    conn = _connect()
    while True:
        if session_id:
            row = conn.execute(
                "SELECT id FROM jobs WHERE state = ? AND session_id = ? ORDER BY created_at LIMIT 1",
                (QUEUED, session_id),
            ).fetchone()
        else:
            row = conn.execute("SELECT id FROM jobs WHERE state = ? ORDER BY created_at LIMIT 1", (QUEUED,)).fetchone()
        if row is None:
            return None
        now = time.time()
        cur = conn.execute(
            "UPDATE jobs SET state = ?, started_at = ?, heartbeat_at = ?, worker = ? WHERE id = ? AND state = ?",
            (RUNNING, now, now, worker, row["id"], QUEUED),
        )
        if cur.rowcount == 1:
            got = conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
            return _as_dict(got)
        # Another worker took it between the read and the write; look again.


def heartbeat(job_id: str) -> None:
    _connect().execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ?", (time.time(), job_id))


def finish(job_id: str, state: str = DONE, detail: str | None = None) -> None:
    _connect().execute(
        "UPDATE jobs SET state = ?, finished_at = ?, detail = ? WHERE id = ?",
        (state, time.time(), detail, job_id),
    )


def cancel(job_id: str) -> bool:
    """Cancel a job that has not finished. True when this call is what cancelled it."""
    cur = _connect().execute(
        "UPDATE jobs SET state = ?, finished_at = ? WHERE id = ? AND state IN (?, ?)",
        (CANCELLED, time.time(), job_id, QUEUED, RUNNING),
    )
    return cur.rowcount == 1


def get(job_id: str) -> dict | None:
    row = _connect().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _as_dict(row) if row else None


def position(job_id: str) -> int | None:
    """How many waiting jobs are ahead of this one. 0 means it is next."""
    conn = _connect()
    row = conn.execute("SELECT state, created_at FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None or row["state"] != QUEUED:
        return None
    return conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE state = ? AND created_at < ?", (QUEUED, row["created_at"])
    ).fetchone()[0]


def depth() -> int:
    return _connect().execute("SELECT COUNT(*) FROM jobs WHERE state = ?", (QUEUED,)).fetchone()[0]


def requeue_lost(stale_seconds: float | None = None) -> int:
    """Return jobs whose worker stopped to the queue, and say how many.

    A job marked running with nothing producing its result would otherwise sit
    there for ever. This is what makes a restart recoverable rather than a loss.
    """
    stale = STALE_RUNNING_SECONDS if stale_seconds is None else stale_seconds
    cutoff = time.time() - stale
    cur = _connect().execute(
        "UPDATE jobs SET state = ?, started_at = NULL, worker = NULL "
        "WHERE state = ? AND COALESCE(heartbeat_at, started_at, created_at) < ?",
        (QUEUED, RUNNING, cutoff),
    )
    if cur.rowcount:
        logger.info("Returned %d lost job(s) to the queue", cur.rowcount)
    return cur.rowcount


def purge_finished(older_than_seconds: float = 86400) -> int:
    cur = _connect().execute(
        "DELETE FROM jobs WHERE state IN (?, ?, ?) AND finished_at < ?",
        (DONE, FAILED, CANCELLED, time.time() - older_than_seconds),
    )
    return cur.rowcount


def _as_dict(row) -> dict:
    out = dict(row)
    try:
        out["request"] = json.loads(out["request"])
    except (TypeError, ValueError):
        out["request"] = {}
    return out
