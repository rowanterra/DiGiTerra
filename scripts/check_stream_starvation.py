"""Measure whether an open progress stream blocks other requests.

Starts gunicorn twice against this repository, once with the default synchronous
worker and once with the threaded worker the Dockerfile uses, runs a training job
in each, holds its progress stream open, and times an unrelated GET of the index
page. Recorded on 2026-09-03 in the audit container: the synchronous worker did
not answer within ten seconds, the threaded worker answered in 0.01 s.

Run from the repository root with the virtualenv that has gunicorn installed:

    .venv/bin/python scripts/check_stream_starvation.py
"""

import http.cookiejar
import json
import os
import subprocess
import threading
import time
import urllib.request


def start(worker_class, port):
    env = dict(
        os.environ,
        MPLBACKEND="Agg",
        DIGITERRA_APP_SUPPORT_DIR=f"/tmp/starve_{worker_class}",
        DIGITERRA_OUTPUT_DIR=f"/tmp/starve_{worker_class}/vis",
    )
    args = [
        ".venv/bin/gunicorn",
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
    p = subprocess.Popen(
        args + ["app:app"], env=env, cwd="/root/DiGiTerra", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
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


sync_dt, sync_code = run("sync", 5301)
gt_dt, gt_code = run("gthread", 5302)
print()
print("sync worker  :", "blocked" if isinstance(sync_code, str) or sync_dt > 5 else f"served in {sync_dt:.2f}s")
print("gthread      :", "blocked" if isinstance(gt_code, str) or gt_dt > 5 else f"served in {gt_dt:.2f}s")
assert gt_code == 200 and gt_dt < 5, "gthread did not serve the unrelated request"
print("\nPASS: with gthread an open stream no longer blocks other requests.")
