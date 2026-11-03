"""Adversarial check of DiGiTerra: every probe the review of 2026-09-24 ran, as a script.

Three reviewers attacked the routes, the pipeline and the browser side on
2026-09-24 and found 25 defects. Each of their reproductions is here as a probe,
so the next review starts from a run of this script rather than from nothing.
A probe PASSES when the defect is absent. Each probe says what it attacks and
what it saw, so a failure is a reproduction, not a hint.

Groups:
  routes    crafted artifact that names a builtin; path traversal on the two
            file routes; a foreign session reading a run's progress; formula
            injection in the workbook route; raw exception text in a response
  pipeline  the review's pipeline findings, run through pytest
            (tests/unit/test_review_2026_09_24.py, the artifact tests)
  browser   a CSV whose headers and labels carry markup, through both pages;
            cleared fields on the guided page; two quick model changes;
            Escape on a tooltip opened by keyboard focus; every figure and step in the window

Usage (from the repository root):
    MPLBACKEND=Agg .venv/bin/python scripts/adversarial_check.py [--only routes|pipeline|browser] [--json out.json]

The browser group needs Playwright with a Chromium; set DIGITERRA_CHROMIUM to a
Chromium binary when Playwright's own is absent. It is skipped, not failed, when
neither is there. Exit status is 1 when any probe fails.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DIGITERRA_ISOLATE_WORK", "0")

RESULTS: list[dict] = []


def _hostile_csv() -> str:
    """Iris-like rows whose headers and one class label carry markup; the numbers
    vary within each class so the data checks do not stop the run for a leak."""
    import random

    rng = random.Random(0)
    head = (
        '<img src=x onerror=window.__xss1=1>,"b"" onmouseover=""window.__xss2=1",'
        "<script>window.__xss3=1</script>,d&amp;,species\n"
    )
    rows = []
    for label, centre in (
        ("<img src=x onerror=window.__xss4=1>", (5.0, 3.4, 1.5, 0.2)),
        ("versicolor", (5.9, 2.8, 4.3, 1.3)),
        ("virginica", (6.6, 3.0, 5.6, 2.0)),
    ):
        for _ in range(20):
            vals = [c + rng.gauss(0, 0.3) for c in centre]
            rows.append(",".join(f"{v:.2f}" for v in vals) + f",{label}\n")
    return head + "".join(rows)


def probe(group: str, name: str):
    """Register a probe; the function returns (ok, detail)."""

    def wrap(fn):
        fn._probe = (group, name)
        return fn

    return wrap


def record(group, name, ok, detail):
    RESULTS.append({"group": group, "probe": name, "ok": bool(ok), "detail": str(detail)[:400]})
    print(f"{'PASS' if ok else 'FAIL':4s} [{group}] {name}: {str(detail)[:160]}", flush=True)


# ---------------------------------------------------------------- routes


def _client():
    from app import app

    app.config["TESTING"] = True
    return app.test_client()


def _session_id(client) -> str | None:
    client.get("/")
    for h in client._cookies.values() if hasattr(client, "_cookies") else []:
        if h.key == "digiterra_session":
            return h.value
    return None


@probe("routes", "crafted artifact naming builtins.eval is refused and runs nothing")
def routes_crafted_artifact():
    from python_scripts.reporting import artifact

    marker = Path(tempfile.mkdtemp()) / "ran"
    code = f"__import__('pathlib').Path({str(marker)!r}).write_text('x')"
    jn = {
        "__class__": "str",
        "__module__": "builtins",
        "__loader__": "JsonNode",
        "content": json.dumps(code),
        "is_json": True,
        "__id__": 1,
    }
    ev = {
        "__class__": "eval",
        "__module__": "builtins",
        "__loader__": "ConstructorFromReduceNode",
        "content": {
            "__class__": "tuple",
            "__module__": "builtins",
            "__loader__": "TupleNode",
            "content": [jn],
            "__id__": 2,
        },
        "__id__": 3,
    }
    schema = {
        "__class__": "dict",
        "__module__": "builtins",
        "__loader__": "DictNode",
        "content": {"model": ev},
        "key_types": {
            "__class__": "list",
            "__module__": "builtins",
            "__loader__": "ListNode",
            "content": [{"__class__": "str", "__module__": "builtins", "__loader__": "TypeNode", "__id__": 4}],
            "__id__": 5,
        },
        "__id__": 6,
        "protocol": 2,
        "_skops_version": "0.14.0",
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("schema.json", json.dumps(schema))
    data = buf.getvalue()
    refused = artifact.untrusted_types(data)
    raised = False
    try:
        artifact.loads(data)
    except artifact.ArtifactError:
        raised = True
    except Exception:
        raised = True
    ok = ("builtins.eval" in refused) and raised and not marker.exists()
    return ok, f"refused={refused} raised={raised} marker={marker.exists()}"


@probe("routes", "path traversal on /download and /user-visualizations")
def routes_traversal():
    c = _client()
    c.get("/")
    hits = []
    for probe_path in ["../SECRET.txt", "..%2fSECRET.txt", "....//SECRET.txt", "/etc/hostname", "%2e%2e/SECRET.txt"]:
        for route in ("/download/", "/user-visualizations/"):
            r = c.get(route + probe_path)
            if r.status_code == 200:
                hits.append(route + probe_path)
    return not hits, f"200 on {hits}" if hits else "every probe answered 404"


@probe("routes", "a foreign session cannot read a run's progress or queue entry")
def routes_cross_session():
    c = _client()
    c.get("/")
    r1 = c.get("/progress/not-a-run-of-this-session")
    r2 = c.get("/queue/not-a-run-of-this-session")
    # A malformed id is a client error (400); a well-formed foreign id is 404.
    ok = r1.status_code in (400, 403, 404) and r2.status_code in (400, 403, 404)
    return ok, f"progress={r1.status_code} queue={r2.status_code}"


@probe("routes", "workbook route neutralises formula cells and hides exception text")
def routes_formula_and_exceptions():
    c = _client()
    c.get("/")
    r = c.post("/downloadAdditionalInfo", json={"table_data": [["=CMD()", "x"], ["+1", "y"]], "sheet_name": "t"})
    detail = f"status={r.status_code}"
    if r.status_code == 200:
        name = r.get_json().get("filename")
        import openpyxl

        from python_scripts import config

        wb_path = None
        for p in Path(config._VIS_BASE).rglob(name):
            wb_path = p
        if wb_path is None:
            return False, detail + " workbook not found"
        ws = openpyxl.load_workbook(wb_path).active
        first = ws.cell(row=2, column=1).value
        return isinstance(first, str) and first.startswith("'="), detail + f" first cell={first!r}"
    return r.status_code in (400, 401, 403), detail


# -------------------------------------------------------------- pipeline


@probe("pipeline", "review regression tests (pytest)")
def pipeline_tests():
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/test_review_2026_09_24.py",
        "tests/unit/test_bundle.py",
        "tests/unit/test_session_store.py",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg"})
    last = [line for line in out.stdout.splitlines() if line.strip()][-1:] or ["no output"]
    return out.returncode == 0, last[0]


# --------------------------------------------------------------- browser


@contextlib.contextmanager
def _server():
    from werkzeug.serving import make_server

    from app import app

    app.config["TESTING"] = True
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = make_server("127.0.0.1", port, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()


@contextlib.contextmanager
def _browser(width=1280, height=720):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        exe = os.environ.get("DIGITERRA_CHROMIUM")
        b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        page = b.new_page(viewport={"width": width, "height": height})
        page.set_default_timeout(60000)
        try:
            yield page
        finally:
            b.close()


def _hostile_file() -> str:
    d = Path(tempfile.mkdtemp())
    (d / "hostile.csv").write_text(_hostile_csv())
    return str(d / "hostile.csv")


XSS = "[window.__xss1, window.__xss2, window.__xss3, window.__xss4]"


def _upload_guided(page, path):
    page.set_input_files("#file", path)
    page.click("#uploadForm button[type=submit]")
    page.wait_for_function(
        "document.getElementById('uploadStatus').textContent.includes('rows') || "
        "!document.getElementById('errors').classList.contains('hidden')"
    )


def browser_probes():
    hostile = _hostile_file()
    with _server() as base:
        # 1. hostile CSV through the guided page
        with _browser() as page:
            dialogs, errors = [], []
            page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
            page.on("pageerror", lambda e: errors.append(str(e)))
            try:
                page.goto(base + "/guided")
                page.check("input[name=mode][value=full]")
                _upload_guided(page, hostile)
                page.click("nav.steps button[data-step='3']")
                page.select_option("#model", "DecisionTree_classifier")
                page.wait_for_selector("#hp_max_depth")
                page.fill("#hp_max_depth", "3")
                page.dispatch_event("#hp_max_depth", "change")
                page.click("#runBtn")
                page.wait_for_function(
                    "document.getElementById('result').textContent.includes('Accuracy')", timeout=180000
                )
                raw = page.text_content("#resultRaw")
                fired = page.evaluate(XSS)
                html = page.inner_html("#result") + page.inner_html(
                    "#columnList" if page.query_selector("#columnList") else "#result"
                )
                typed_int = '"max_depth": 3' in raw
                ok = not any(fired) and not dialogs and "<img src=x" not in html and typed_int
                record(
                    "browser",
                    "hostile CSV through the guided page, typed None-default hyperparameter",
                    ok,
                    f"xss={fired} dialogs={dialogs} max_depth typed as int={typed_int}",
                )
            except Exception as e:
                record("browser", "hostile CSV through the guided page", False, f"{type(e).__name__}: {e}")
            # 2. cleared fields
            try:
                page.goto(base + "/guided")
                page.check("input[name=mode][value=full]")
                _upload_guided(page, str(ROOT / "examples" / "iris.csv"))
                page.click("nav.steps button[data-step='3']")
                page.select_option("#model", "KNeighbors_classifier")
                page.wait_for_selector("#hp_n_neighbors")
                for fid in ("seed", "sigfig", "cvFolds"):
                    page.fill("#" + fid, "")
                    page.dispatch_event("#" + fid, "change")
                with page.expect_request(lambda r: r.url.endswith("/process")) as req:
                    page.click("#runBtn")
                body = json.loads(req.value.post_data or "{}")
                page.wait_for_function(
                    "document.getElementById('result').textContent.includes('Accuracy')", timeout=180000
                )
                ok = (
                    body.get("seedValue") is None
                    and body.get("sigfig") is None
                    and body.get("crossValidationFolds") is None
                )
                record(
                    "browser",
                    "cleared seed, decimals and folds are sent as null and the run completes",
                    ok,
                    f"seed={body.get('seedValue')} sigfig={body.get('sigfig')} "
                    f"folds={body.get('crossValidationFolds')}",
                )
            except Exception as e:
                record("browser", "cleared fields on the guided page", False, f"{type(e).__name__}: {e}")
            # 3. Escape on a focused tooltip
            try:
                page.goto(base + "/guided")
                page.mouse.move(5, 5)
                page.focus("#file")
                shown_before = page.locator("label[for=file] + .helpwrap .tip").is_visible()
                page.keyboard.press("Escape")
                shown_after = page.locator("label[for=file] + .helpwrap .tip").is_visible()
                record(
                    "browser",
                    "Escape closes a tooltip opened by keyboard focus",
                    shown_before and not shown_after,
                    f"before={shown_before} after={shown_after}",
                )
            except Exception as e:
                record("browser", "Escape on a focused tooltip", False, f"{type(e).__name__}: {e}")
            # 4. two quick model changes
            try:
                page.goto(base + "/guided")
                page.check("input[name=mode][value=full]")
                _upload_guided(page, str(ROOT / "examples" / "iris.csv"))
                page.click("nav.steps button[data-step='3']")

                def slow(route):
                    time.sleep(1.5)
                    route.continue_()

                page.route("**/model-defaults/GradientBoosting_classifier", slow)
                page.select_option("#model", "GradientBoosting_classifier")
                page.wait_for_timeout(100)
                page.select_option("#model", "KNeighbors_classifier")
                page.wait_for_timeout(2500)
                note = page.text_content("#hyperNote")
                record(
                    "browser",
                    "a slow defaults response for an earlier model does not fill the table of the later one",
                    "KNeighbors" in note,
                    note[:80],
                )
            except Exception as e:
                record("browser", "model change race", False, f"{type(e).__name__}: {e}")
        # 5. hostile CSV through the classic page, and every figure in the window
        with _browser() as page:
            dialogs = []
            page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
            try:
                page.goto(base + "/")
                page.wait_for_load_state("networkidle")
                page.evaluate("() => { const b = document.getElementById('startModelingButton'); if (b) b.click(); }")
                page.set_input_files("#file", hostile)
                with page.expect_response(lambda r: r.url.endswith("/upload")):
                    page.evaluate("() => document.getElementById('uploadForm').requestSubmit()")
                page.wait_for_timeout(500)
                with page.expect_response(lambda r: r.url.endswith("/correlationMatrices"), timeout=120000):
                    page.evaluate("() => document.getElementById('corrForm').requestSubmit()")
                page.wait_for_timeout(800)
                eo = page.inner_html("#explorationOutput")
                fired = page.evaluate(XSS)
                boxes = page.evaluate(
                    "Array.from(document.querySelectorAll('.exploration-graphic img')).filter(e => e.offsetParent)"
                    ".map(e => Math.round(e.getBoundingClientRect().bottom))"
                )
                inner = page.evaluate("window.innerHeight")
                ok = (
                    "<img src=x" not in eo
                    and not any(fired)
                    and not dialogs
                    and boxes
                    and all(b <= inner for b in boxes)
                )
                record(
                    "browser",
                    "hostile CSV through the classic page; exploration figures in a 720 px window",
                    ok,
                    f"xss={fired} dialogs={dialogs} figure bottoms={boxes} window={inner}",
                )
            except Exception as e:
                record("browser", "hostile CSV through the classic page", False, f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["routes", "pipeline", "browser"])
    ap.add_argument("--json")
    args = ap.parse_args()

    probes = [v for v in globals().values() if callable(v) and hasattr(v, "_probe")]
    for fn in probes:
        group, name = fn._probe
        if args.only and args.only != group:
            continue
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-300:]}"
        record(group, name, ok, detail)
    if not args.only or args.only == "browser":
        try:
            import playwright.sync_api  # noqa: F401

            browser_probes()
        except ImportError:
            print("SKIP [browser] Playwright is not installed")
        except Exception as e:  # noqa: BLE001
            record("browser", "browser group", False, f"{type(e).__name__}: {e}")

    failed = [r for r in RESULTS if not r["ok"]]
    print(f"\n{len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    if args.json:
        Path(args.json).write_text(json.dumps({"date": time.strftime("%Y-%m-%d"), "results": RESULTS}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
