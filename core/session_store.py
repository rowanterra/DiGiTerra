"""Where a session's data lives.

Frames and fitted models have been held in a dictionary in this process, which is
why gunicorn runs one worker: a second worker would answer requests about a
session it cannot see. That is a limit on how a deployment can be run, and the
same limit applies to the rate-limit buckets and to a job queue.

This module puts the storage behind a small interface with two implementations.
The memory backend is what the desktop has always used and stays the default.
The disk backend writes each session to its own directory under the application
support directory: the frame as parquet, the fitted objects through the artifact
format, and everything else as JSON. Several processes can then serve one
deployment, and a session survives a restart.

What a backend must promise is narrow: a mapping in, the same mapping out, and no
partial state visible to a reader. Writes go to a temporary file and are renamed
into place, which is atomic on POSIX, so a reader sees the previous version or
the new one and never half of either.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Keys whose values are too large or too specialized for JSON, and how each is
# written. Everything else in a session is small and JSON-serializable.
FRAME_KEY = "data"
OBJECT_KEYS = (
    "model",
    "X_scaler",
    "y_scaler",
)
META_NAME = "meta.json"
FRAME_NAME = "data.parquet"
OBJECTS_NAME = "objects.skops"
OBJECTS_SIG_NAME = "objects.skops.sig"


def backend_name() -> str:
    """`memory` or `disk`. The server profile defaults to disk."""
    explicit = os.environ.get("DIGITERRA_SESSION_BACKEND", "").strip().lower()
    if explicit in {"memory", "disk"}:
        return explicit
    from core.security import is_server_profile

    return "disk" if is_server_profile() else "memory"


class MemoryBackend:
    """The dictionary this application has always used."""

    def __init__(self):
        self._data: dict[str, dict] = {}

    def load(self, session_id: str) -> dict | None:
        return self._data.get(session_id)

    def save(self, session_id: str, mapping: dict) -> None:
        self._data[session_id] = mapping

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    def ids(self) -> list[str]:
        return list(self._data)

    def touched_at(self, session_id: str) -> float:
        return time.time()


class DiskBackend:
    """One directory per session, written atomically.

    A frame is parquet because it is the format this application already reads and
    writes and it keeps dtypes. Fitted objects go through the artifact format, so
    a session read back does not execute what its file says any more than a bundle
    does. Anything else is JSON.
    """

    def __init__(self, root: Path | None = None):
        if root is None:
            from python_scripts import config

            root = Path(config.APP_SUPPORT_DIR) / "sessions"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, session_id: str) -> Path:
        # The id is 32 hex characters this application issued; anything else is
        # refused rather than used to build a path.
        if not session_id or not all(c in "0123456789abcdef-_" for c in session_id.lower()):
            raise ValueError(f"Unusable session id: {session_id!r}")
        return self.root / session_id

    @staticmethod
    def _write_atomic(path: Path, write) -> None:
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        try:
            write(tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def load(self, session_id: str) -> dict | None:
        d = self._dir(session_id)
        meta_path = d / META_NAME
        if not meta_path.is_file():
            return None
        try:
            mapping = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Session %s has an unreadable record: %s", session_id, e)
            return None
        frame_path = d / FRAME_NAME
        if frame_path.is_file():
            import pandas as pd

            try:
                mapping[FRAME_KEY] = pd.read_parquet(frame_path)
            except Exception as e:
                logger.warning("Session %s frame not read: %s", session_id, e)
        objects_path = d / OBJECTS_NAME
        if objects_path.is_file():
            from python_scripts.reporting import artifact
            from python_scripts.reporting.bundle import signature_valid, signing_key

            # The objects file is read only when it carries the signature this
            # installation wrote at save time, so a file put in place by anything
            # else is refused before a single object is built (review of 2026-09-24).
            try:
                data = objects_path.read_bytes()
                sig_path = d / OBJECTS_SIG_NAME
                sig = sig_path.read_text(encoding="utf-8").strip() if sig_path.is_file() else None
                if not signature_valid(data, sig, signing_key(create=False)):
                    logger.warning("Session %s objects not read: the signature does not verify", session_id)
                else:
                    mapping.update(artifact.loads(data))
            except Exception as e:
                logger.warning("Session %s objects not read: %s", session_id, e)
        return mapping

    def save(self, session_id: str, mapping: dict) -> None:
        d = self._dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        rest = {}
        frame = mapping.get(FRAME_KEY)
        objects = {k: mapping.get(k) for k in OBJECT_KEYS if mapping.get(k) is not None}
        for key, value in mapping.items():
            if key == FRAME_KEY or key in OBJECT_KEYS:
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                continue  # not JSON, not a frame, not a fitted object: not kept
            rest[key] = value
        self._write_atomic(d / META_NAME, lambda p: p.write_text(json.dumps(rest, default=str), encoding="utf-8"))
        if frame is not None:
            self._write_atomic(d / FRAME_NAME, lambda p: frame.to_parquet(p, index=False))
        elif (d / FRAME_NAME).exists():
            (d / FRAME_NAME).unlink()
        if objects and objects.get("model") is not None:
            from python_scripts.reporting import artifact
            from python_scripts.reporting.bundle import sign_bytes, signing_key

            try:
                payload = artifact.dumps(objects)
                key = signing_key(create=True)
                if key is None:
                    raise RuntimeError("no signing key")
                signature = sign_bytes(payload, key)
                self._write_atomic(d / OBJECTS_NAME, lambda p: p.write_bytes(payload))
                self._write_atomic(d / OBJECTS_SIG_NAME, lambda p: p.write_text(signature, encoding="utf-8"))
            except Exception as e:
                logger.warning("Session %s objects not written: %s", session_id, e)
        elif (d / OBJECTS_NAME).exists():
            (d / OBJECTS_NAME).unlink()
            if (d / OBJECTS_SIG_NAME).exists():
                (d / OBJECTS_SIG_NAME).unlink()

    def delete(self, session_id: str) -> None:
        shutil.rmtree(self._dir(session_id), ignore_errors=True)

    def ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return [p.name for p in self.root.iterdir() if p.is_dir() and (p / META_NAME).is_file()]

    def touched_at(self, session_id: str) -> float:
        meta = self._dir(session_id) / META_NAME
        try:
            return meta.stat().st_mtime
        except OSError:
            return 0.0


_backend = None
_backend_kind = None


def backend():
    """The backend for this process, built on first use and rebuilt if it changes."""
    global _backend, _backend_kind
    kind = backend_name()
    if _backend is None or _backend_kind != kind:
        _backend = DiskBackend() if kind == "disk" else MemoryBackend()
        _backend_kind = kind
        logger.info("Session storage: %s", kind)
    return _backend


def reset_backend() -> None:
    """Drop the built backend so the next call reads the environment again."""
    global _backend, _backend_kind
    _backend = None
    _backend_kind = None
