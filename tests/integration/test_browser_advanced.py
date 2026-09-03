"""Browser test: Advanced-mode hyperparameter controls reach the /process payload.

Runs headless Chromium through Playwright against a live app instance.
Skipped automatically when Playwright or the browser is unavailable.
"""
import glob
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = os.path.join(ROOT, "examples")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _chromium_executable():
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
    for pattern in ("chromium-*/chrome-linux/chrome", "chromium_headless_shell-*/chrome-linux/headless_shell"):
        hits = sorted(glob.glob(os.path.join(root, pattern)))
        if hits:
            return hits[-1]
    return None


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    support = tmp_path_factory.mktemp("pw_support")
    port = _free_port()
    env = dict(os.environ,
               DIGITERRA_APP_SUPPORT_DIR=str(support),
               DIGITERRA_OUTPUT_DIR=str(support / "vis"),
               DIGITERRA_PORT=str(port),
               MPLBACKEND="Agg")
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "app.py")], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(url + "/", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.skip("app did not start")
    yield url
    proc.kill()


@pytest.fixture(scope="module")
def browser():
    try:
        pw = playwright_sync.sync_playwright().start()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"playwright unavailable: {exc}")
    try:
        try:
            b = pw.chromium.launch(headless=True)
        except Exception:
            exe = _chromium_executable()
            if not exe:
                raise
            b = pw.chromium.launch(headless=True, executable_path=exe)
    except Exception as exc:
        pw.stop()
        pytest.skip(f"chromium unavailable: {exc}")
    yield b
    b.close()
    pw.stop()


class Session:
    """One page with console-error capture and a /process interceptor."""

    def __init__(self, browser, base_url):
        self.page = browser.new_page()
        self.errors = []
        self.payloads = []
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        # Uncaught exceptions and console.error calls fail the test. A "Failed to
        # load resource" line is the browser reporting an HTTP status; the payload
        # assertions cover the requests that matter, and a 4xx from a stale
        # user-visualizations fetch during the full suite is not a script fault.
        self.page.on("console", lambda m: self.errors.append(f"console.error: {m.text}")
                     if m.type == "error" and "Failed to load resource" not in m.text else None)

        def handle(route, request):
            if request.method == "POST":
                self.payloads.append(json.loads(request.post_data or "{}"))
                route.fulfill(status=400, content_type="application/json",
                              body=json.dumps({"error": "intercepted by test"}))
            else:
                route.continue_()

        self.page.route("**/process", handle)
        self.page.goto(base_url + "/")
        self.page.wait_for_load_state("networkidle")

    def close(self):
        self.page.close()

    # --- flow helpers -------------------------------------------------
    def upload_and_preprocess(self, csv_name, task, indicators, predictors):
        p = self.page
        p.evaluate("() => { const b = document.getElementById('startModelingButton'); if (b) b.click(); }")
        p.set_input_files("#file", os.path.join(EXAMPLES, csv_name))
        with p.expect_response(lambda r: r.url.endswith("/upload")) as up:
            p.evaluate("() => document.getElementById('uploadForm').requestSubmit()")
        assert up.value.status == 200, f"upload failed: {up.value.status}"
        p.wait_for_timeout(500)
        p.evaluate("() => { if (window.moveToModelPreprocess) moveToModelPreprocess(); }")
        p.select_option("#outputType1", task)
        p.fill("#indicators", indicators)
        p.fill("#predictors", predictors)
        with p.expect_response(lambda r: r.url.endswith("/preprocess")) as pre:
            p.click("#continueToModelingButton", force=True)
        assert pre.value.status == 200, f"preprocess failed: {pre.value.status} {pre.value.text()[:300]}"
        p.wait_for_timeout(500)

    def select_advanced(self, select_id, model):
        p = self.page
        p.check("#advancedMode", force=True)
        p.evaluate("() => { if (window.switchModelingMode) switchModelingMode('advanced'); }")
        p.select_option("#" + select_id, model)
        p.wait_for_timeout(100)

    def visible(self, el_id):
        return self.page.evaluate("""(id) => {
            let el = document.getElementById(id);
            if (!el) return 'missing';
            for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
                if (getComputedStyle(n).display === 'none') return 'hidden:' + (n.id || n.className);
            }
            return 'visible';
        }""", el_id)

    def toggle(self, el_id):
        # Switch-styled checkboxes sit off-screen behind a .slider span; set them directly.
        self.page.evaluate("""(id) => { const el = document.getElementById(id);
            el.checked = true; el.dispatchEvent(new Event('change', {bubbles: true})); }""", el_id)

    def submit_advanced(self):
        n = len(self.payloads)
        self.page.click("#advancedOptimizationSubmitButton", force=True)
        for _ in range(50):
            if len(self.payloads) > n:
                return self.payloads[-1]
            self.page.wait_for_timeout(100)
        raise AssertionError("no POST /process captured after Advanced submit")


@pytest.fixture
def session(browser, base_url):
    s = Session(browser, base_url)
    yield s
    s.close()


def _assert_visible(session, ids):
    states = {i: session.visible(i) for i in ids}
    bad = {k: v for k, v in states.items() if v != "visible"}
    assert not bad, f"controls not visible: {bad}"


def test_no_duplicate_ids(session):
    total, distinct = session.page.evaluate(
        "() => { const ids=[...document.querySelectorAll('[id]')].map(e=>e.id); return [ids.length, new Set(ids).size]; }")
    assert total == distinct, f"{total - distinct} duplicate ids on the live page"
    assert not session.errors, session.errors


def test_ridge_advanced_controls_reach_payload(session):
    session.upload_and_preprocess("synthetic_regression.csv", "Numeric", "A-J", "K")
    session.select_advanced("advancedNModels", "Ridge")
    session.toggle("advancedNonreqRidgeSlider")
    session.page.select_option("#advancedRidgeSolver", "saga")
    session.page.fill("#advancedRidgeTol", "0.000001")
    session.page.fill("#advancedRidgeMaxIter", "5000")
    _assert_visible(session, ["advancedRidgeFields", "advancedNonreqRidgeSlider",
                              "advancedRidgeSolver", "advancedRidgeTol", "advancedRidgeMaxIter"])
    payload = session.submit_advanced()
    hp = payload.get("hyperparameters", {})
    print("hyperparameters:", json.dumps(hp))
    assert hp.get("solver") == "saga", hp
    assert hp.get("RidgeTol") == pytest.approx(1e-6), hp
    assert hp.get("RidgeMaxIter") == 5000, hp
    assert not session.errors, session.errors


def test_adaboost_relocated_group_reaches_payload(session):
    session.upload_and_preprocess("synthetic_regression.csv", "Numeric", "A-J", "K")
    session.select_advanced("advancedNModels", "AdaBoost")
    session.page.fill("#advancedAdaBoostNEstimators", "123")
    _assert_visible(session, ["advancedAdaBoostFields", "advancedAdaBoostNEstimators"])
    payload = session.submit_advanced()
    hp = payload.get("hyperparameters", {})
    print("hyperparameters:", json.dumps(hp))
    assert hp.get("n_estimators") == 123, hp
    assert not session.errors, session.errors


def test_logistic_classifier_advanced_controls_reach_payload(session):
    session.upload_and_preprocess("iris.csv", "Classifier", "A-D", "E")
    session.select_advanced("advancedClassModels", "Logistic_classifier")
    session.toggle("advancedNonreqLogisticClassifierSlider")
    session.page.select_option("#advancedClass_LogisticSolver", "saga")
    session.page.fill("#advancedClass_LogisticTol", "0.000001")
    _assert_visible(session, ["advancedNonreqLogisticClassifierSlider",
                              "advancedClass_LogisticSolver", "advancedClass_LogisticTol"])
    payload = session.submit_advanced()
    hp = payload.get("hyperparameters", {})
    print("hyperparameters:", json.dumps(hp))
    assert hp.get("Class_LogisticSolver") == "saga", hp
    assert hp.get("Class_LogisticTol") == pytest.approx(1e-6), hp
    assert not session.errors, session.errors
