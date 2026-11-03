"""Measure whether an open progress stream blocks other requests.

Starts gunicorn twice against this repository, once with the default synchronous
worker and once with the threaded worker the Dockerfile uses, runs a training job
in each, holds its progress stream open, and times an unrelated GET of the index
page. Recorded on 2026-09-03 in the audit container: the synchronous worker did
not answer within ten seconds, the threaded worker answered in 0.01 s.

Run with any interpreter that has the project and gunicorn installed; the
repository root is found from this file and the ports are allocated free:

    python scripts/check_stream_starvation.py [--threshold 5.0]

Exit code 0 when the threaded worker answers within the threshold and 1
otherwise, so the script can run as a CI check.
"""

import argparse
import http.cookiejar
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(worker_class, port):
    work = Path(tempfile.mkdtemp(prefix=f"starve_{worker_class}_"))
    env = dict(
        os.environ,
        MPLBACKEND="Agg",
        DIGITERRA_APP_SUPPORT_DIR=str(work),
        DIGITERRA_OUTPUT_DIR=str(work / "vis"),
    )
    args = [
        sys.executable,
        "-m",
        "gunicorn",
        "--bind",
        f"127.0.0.1:{port}",
        "--workers",
        "1",
        "--timeout",
        "300",
        "--log-level",
        "error",
    ]
    if worker_class == "gthread":
        args += ["--worker-class", "gthread", "--threads", "8"]
    p = subprocess.Popen(args + ["app:app"], env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1).read()
            return p
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


def run(worker_class, port):
    server = start(worker_class, port)
    try:
        op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        b = "----x"
        import random

        random.seed(0)
        cols = [f"c{i}" for i in range(25)]
        rows = (
            ",".join(cols + ["y"])
            + "\n"
            + "\n".join(",".join(f"{random.random():.4f}" for _ in range(26)) for _ in range(6000))
        )
        body = (
            f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="t.csv"\r\n'
            f"Content-Type: text/csv\r\n\r\n{rows}\r\n--{b}--\r\n"
        ).encode()
        op.open(
            urllib.request.Request(
                f"http://127.0.0.1:{port}/upload", body, {"Content-Type": f"multipart/form-data; boundary={b}"}
            )
        )
        op.open(
            urllib.request.Request(
                f"http://127.0.0.1:{port}/preprocess",
                json.dumps({"indicators": list(range(25)), "predictors": [25], "stratify": 0}).encode(),
                {"Content-Type": "application/json"},
            )
        )
        payload = {
            "indicators": list(range(25)),
            "predictors": [25],
            "models": "Linear",
            "hyperparameters": {},
            "nonreq": False,
            "scaler": "standard",
            "units": "",
            "sigfig": 3,
            "seedValue": 42,
            "testSize": "0.20",
            "stratifyBool": False,
            "dropMissing": "none",
            "imputeStrategy": "none",
            "dropZero": "none",
            "quantileBinDict": {"quantile": 0, "bin": 0, "binsLabel": ""},
            "useTransformer": "No",
            "transformerCols": [],
            "crossValidationType": "RepeatedKFold",
            "crossValidationFolds": 20,
        }
        sid = json.loads(
            op.open(
                urllib.request.Request(
                    f"http://127.0.0.1:{port}/process",
                    json.dumps(payload).encode(),
                    {"Content-Type": "application/json"},
                )
            ).read()
        )["session_id"]

        holding = threading.Event()

        def hold():
            r = op.open(f"http://127.0.0.1:{port}/progress/{sid}", timeout=60)
            holding.set()
            deadline = time.time() + 20
            while time.time() < deadline:
                if not r.readline():
                    break

        t = threading.Thread(target=hold, daemon=True)
        t.start()
        holding.wait(10)
        time.sleep(1.5)

        t0 = time.time()
        try:
            code = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).status
        except Exception as e:
            code = f"blocked ({type(e).__name__})"
        dt = time.time() - t0
        print(f"{worker_class:8s}  unrelated GET / -> {code}  in {dt:.2f}s")
        return dt, code
    finally:
        server.kill()
        server.wait(timeout=20)


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure whether an open progress stream blocks other requests.")
    ap.add_argument("--threshold", type=float, default=5.0, help="seconds the unrelated request may take")
    args = ap.parse_args()
    sync_dt, sync_code = run("sync", free_port())
    gt_dt, gt_code = run("gthread", free_port())
    print()
    blocked = isinstance(sync_code, str) or sync_dt > args.threshold
    print("sync worker  :", "blocked" if blocked else f"served in {sync_dt:.2f}s")
    gt_blocked = isinstance(gt_code, str) or gt_dt > args.threshold
    print("gthread      :", "blocked" if gt_blocked else f"served in {gt_dt:.2f}s")
    if gt_code != 200 or gt_dt >= args.threshold:
        print("FAIL: gthread did not serve the unrelated request", file=sys.stderr)
        return 1
    print("\nPASS: with gthread an open stream does not block other requests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
