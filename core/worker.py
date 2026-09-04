"""Run one step in a separate process under a memory and time limit.

A budget enforced inside the process doing the allocating cannot bound that
process: by the time a frame or a fitted model is large enough to measure, the
memory is already taken. The measurement DiGiTerra does after a parse and after a
fit is a record, not a control.

This module is the control. The work runs in a child process with RLIMIT_AS set,
so an allocation past the budget raises MemoryError inside the child rather than
taking the machine, and with a wall clock, so a fit that never converges is killed
rather than held. The parent gets an answer, a refusal, or a timeout, and its own
memory is untouched either way.

The child is started with the spawn method so it does not inherit the parent's
loaded state, and the two ends exchange data through a file rather than a pipe,
because a pipe deadlocks when a result is larger than its buffer and the parent is
waiting on the process rather than reading.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import tempfile
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_MB = float(os.environ.get("DIGITERRA_WORKER_MEMORY_MB", "4096") or 0)
DEFAULT_TIMEOUT_S = float(os.environ.get("DIGITERRA_WORKER_TIMEOUT_SECONDS", "1800") or 0)
# RLIMIT_AS counts address space, not resident pages, and a Python child has spent
# several hundred megabytes of it before it touches any data: the interpreter, then
# numpy, pandas and scikit-learn with their allocator arenas. Measured here, a child
# that parses a small CSV needs about 512 MB before the file is read at all. The
# budget for a step is therefore this baseline plus what the step's data may take,
# and a limit below the baseline would refuse every step whatever its size.
BASE_MEMORY_MB = float(os.environ.get("DIGITERRA_WORKER_BASE_MB", "768") or 0)


def workers_enabled() -> bool:
    """Whether a step should be run in a child process.

    Off on the desktop, where the machine belongs to the person using it, and on
    by default under the server profile. DIGITERRA_ISOLATE_WORK overrides either.
    """
    raw = os.environ.get("DIGITERRA_ISOLATE_WORK")
    if raw is not None and raw.strip() != "":
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    from core.security import is_server_profile

    return is_server_profile()


class WorkerError(RuntimeError):
    """The child did not return a result."""


class WorkerMemoryExceeded(WorkerError):
    """The child asked for more memory than its limit."""


class WorkerTimeout(WorkerError):
    """The child ran past its wall clock."""


def _apply_limits(memory_mb: float) -> None:
    """Set the child's address-space limit. POSIX only; a no-op elsewhere."""
    if not memory_mb or memory_mb <= 0:
        return
    try:
        import resource

        limit = int(memory_mb * 1024 * 1024)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        ceiling = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, hard))
    except (ImportError, ValueError, OSError) as e:
        logger.warning("Worker memory limit not applied: %s", e)


def _child(func, args, kwargs, out_path: str, memory_mb: float) -> None:
    """Entry point of the child: apply the limits, run, write the outcome."""
    _apply_limits(memory_mb)
    payload: dict
    try:
        payload = {"ok": True, "value": func(*args, **(kwargs or {}))}
    except MemoryError:
        payload = {"ok": False, "kind": "memory", "detail": "The step asked for more memory than its limit."}
    except BaseException as e:  # noqa: BLE001 - the outcome is reported, not swallowed
        payload = {
            "ok": False,
            "kind": "error",
            "detail": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
        }
    try:
        with open(out_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except MemoryError:
        with open(out_path, "wb") as fh:
            pickle.dump({"ok": False, "kind": "memory", "detail": "The result was too large to return."}, fh)
    except Exception as e:  # a result that cannot be written is a failure, not a hang
        with open(out_path, "wb") as fh:
            pickle.dump({"ok": False, "kind": "error", "detail": f"Result not returned: {e}"}, fh)


def run_isolated(func, *args, memory_mb: float | None = None, timeout_s: float | None = None, **kwargs):
    """Run func in a child process under the limits and return its value.

    The child is trusted: it runs this application's own code on this
    application's own data, and the file between the two processes is written by
    the child and read only by the parent that created it. The limits are there to
    bound the work, not to contain an attacker.
    """
    memory_mb = DEFAULT_MEMORY_MB if memory_mb is None else memory_mb
    timeout_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
    ctx = mp.get_context("spawn")
    fd, out_path = tempfile.mkstemp(prefix="dgt_worker_", suffix=".pkl")
    os.close(fd)
    proc = ctx.Process(target=_child, args=(func, args, kwargs, out_path, memory_mb), daemon=True)
    proc.start()
    try:
        proc.join(timeout_s if timeout_s and timeout_s > 0 else None)
        if proc.is_alive():
            proc.terminate()
            proc.join(10)
            if proc.is_alive():
                proc.kill()
                proc.join(5)
            raise WorkerTimeout(f"The step ran past its limit of {timeout_s:.0f} seconds and was stopped.")
        size = Path(out_path).stat().st_size if Path(out_path).exists() else 0
        if size == 0:
            # The child died without writing: the kernel killed it, which is what
            # an allocation past a hard limit looks like from out here.
            raise WorkerMemoryExceeded(
                f"The step stopped without a result (exit code {proc.exitcode}); "
                "it asked for more memory than its limit allows."
            )
        with open(out_path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("ok"):
            return payload["value"]
        if payload.get("kind") == "memory":
            raise WorkerMemoryExceeded(payload.get("detail", "The step ran out of memory."))
        logger.warning("Isolated step failed: %s", payload.get("traceback") or payload.get("detail"))
        raise WorkerError(payload.get("detail", "The step failed."))
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def read_table_isolated(filepath, max_cells: int, max_frame_mb: float):
    """Parse a table in a child process when the profile asks for it.

    The reader already refuses a file whose metadata says it is too large. This
    bounds what the parse itself may allocate, which metadata cannot promise.
    """
    from core.helpers import read_table_bounded

    if not workers_enabled():
        return read_table_bounded(filepath, max_cells=max_cells, max_frame_mb=max_frame_mb)
    # Four times the frame budget covers the intermediate copies a parser makes.
    data_budget = max_frame_mb * 4 if max_frame_mb else DEFAULT_MEMORY_MB
    return run_isolated(
        read_table_bounded,
        str(filepath),
        memory_mb=BASE_MEMORY_MB + data_budget,
        timeout_s=float(os.environ.get("DIGITERRA_PARSE_TIMEOUT_SECONDS", "300") or 300),
        max_cells=max_cells,
        max_frame_mb=max_frame_mb,
    )
