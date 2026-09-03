"""Shared configuration and paths for DiGiTerra.

Single source of truth for BASE_DIR, APP_SUPPORT_DIR, UPLOAD_DIR, LOG_DIR, VIS_DIR, and URL_PREFIX.
"""

import os
import platform
from contextvars import ContextVar
from pathlib import Path

# Repo root (parent of python_scripts/)
BASE_DIR = Path(os.environ.get("DIGITERRA_BASE_DIR", Path(__file__).resolve().parent.parent))

# Platform-specific app support and log directories
if platform.system() == "Windows":
    _default_app_support = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "DiGiTerra"
    LOG_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "DiGiTerra" / "Logs"
elif platform.system() == "Linux":
    xdg_data_home = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    _default_app_support = Path(xdg_data_home) / "DiGiTerra"
    LOG_DIR = Path(xdg_cache_home) / "DiGiTerra" / "logs"
else:
    _default_app_support = Path.home() / "Library" / "Application Support" / "DiGiTerra"
    LOG_DIR = Path.home() / "Library" / "Logs" / "DiGiTerra"

APP_SUPPORT_DIR = Path(os.environ.get("DIGITERRA_APP_SUPPORT_DIR", str(_default_app_support)))
UPLOAD_DIR = APP_SUPPORT_DIR / "uploads"

# URL prefix for deployment (e.g. /digiterra). Set env URL_PREFIX=digiterra.
_url_prefix_raw = os.environ.get("URL_PREFIX", "").strip().strip("/")
URL_PREFIX = ("/" + _url_prefix_raw) if _url_prefix_raw else ""

# Output directory for visualizations and generated files (plots, PDFs, predictions).
# app.py calls update_vis_dir() at startup to set the base to APP_SUPPORT_DIR /
# "user_visualizations" or DIGITERRA_OUTPUT_DIR.
#
# Generated files carry fixed names (predictions.csv, correlation_pearson.png,
# model_performance.xlsx). Writing them all into one directory means the second
# person to run a model overwrites the first person's outputs, and the download
# routes hand back whichever file is current. Each session therefore gets its own
# subdirectory, selected through a context variable that the request handler and
# the training thread both set.
_VIS_BASE = Path(os.environ.get("DIGITERRA_OUTPUT_DIR", str(APP_SUPPORT_DIR / "user_visualizations")))

_session_scope: ContextVar = ContextVar("digiterra_vis_scope", default=None)


def _scope_slug(session_id) -> str:
    """A filesystem-safe directory name for a session id."""
    text = str(session_id or "")
    safe = "".join(ch for ch in text if ch.isalnum() or ch in "-_")[:64]
    return safe or "default"


def set_session_scope(session_id):
    """Point generated-file paths at this session's own directory."""
    return _session_scope.set(_scope_slug(session_id))


def reset_session_scope(token):
    """Undo a set_session_scope call."""
    try:
        _session_scope.reset(token)
    except ValueError:
        _session_scope.set(None)


class _ScopedVisDir:
    """A directory that resolves to the current session's subdirectory.

    Modules bind this object once at import (``from config import VIS_DIR``) and
    build paths from it with ``/``. Resolution happens per use, so one imported
    name serves every session without a global reassignment.
    """

    __slots__ = ("_base",)

    def __init__(self, base: Path):
        self._base = Path(base)

    @property
    def base(self) -> Path:
        return self._base

    def _rebase(self, base: Path):
        self._base = Path(base)

    def current(self) -> Path:
        scope = _session_scope.get()
        path = self._base if scope is None else self._base / scope
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __truediv__(self, other) -> Path:
        return self.current() / other

    def joinpath(self, *parts) -> Path:
        return self.current().joinpath(*parts)

    def mkdir(self, *args, **kwargs):
        return self.current().mkdir(*args, **kwargs)

    def glob(self, pattern):
        return self.current().glob(pattern)

    def iterdir(self):
        return self.current().iterdir()

    def resolve(self) -> Path:
        return self.current().resolve()

    def exists(self) -> bool:
        return self.current().exists()

    def is_dir(self) -> bool:
        return self.current().is_dir()

    @property
    def name(self) -> str:
        return self.current().name

    @property
    def parent(self) -> Path:
        return self.current().parent

    def __fspath__(self) -> str:
        return str(self.current())

    def __str__(self) -> str:
        return str(self.current())

    def __repr__(self) -> str:
        return f"_ScopedVisDir({self._base!r})"

    def __eq__(self, other):
        if isinstance(other, _ScopedVisDir):
            return self._base == other._base
        if isinstance(other, (str, Path)):
            return self.current() == Path(other)
        return NotImplemented

    def __hash__(self):
        return hash(self.current())


VIS_DIR = _ScopedVisDir(_VIS_BASE)
_VIS_BASE.mkdir(parents=True, exist_ok=True)


def update_vis_dir(new_path: Path):
    """Point the base output directory at new_path. Called at startup from app.py."""
    global _VIS_BASE
    _VIS_BASE = Path(new_path)
    _VIS_BASE.mkdir(parents=True, exist_ok=True)
    VIS_DIR._rebase(_VIS_BASE)
