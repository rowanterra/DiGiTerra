"""Browser test: Advanced-mode hyperparameter controls reach the /process payload.

Runs headless Chromium through Playwright against a live app instance.
Skipped automatically when Playwright or the browser is unavailable.
"""

import glob
import json
import os
import re
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
    env = dict(
        os.environ,
        DIGITERRA_APP_SUPPORT_DIR=str(support),
        DIGITERRA_OUTPUT_DIR=str(support / "vis"),
        DIGITERRA_PORT=str(port),
        MPLBACKEND="Agg",
    )
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "app.py")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
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

    def __init__(self, browser, base_url, intercept=True):
        self.page = browser.new_page()
        self.errors = []
        self.payloads = []
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        # Uncaught exceptions and console.error calls fail the test. A "Failed to
        # load resource" line is the browser reporting an HTTP status; the payload
        # assertions cover the requests that matter, and a 4xx from a stale
        # user-visualizations fetch during the full suite is not a script fault.
        self.page.on(
            "console",
            lambda m: (
                self.errors.append(f"console.error: {m.text}")
                if m.type == "error" and "Failed to load resource" not in m.text
                else None
            ),
        )

        def handle(route, request):
            if request.method == "POST":
                self.payloads.append(json.loads(request.post_data or "{}"))
                route.fulfill(
                    status=400, content_type="application/json", body=json.dumps({"error": "intercepted by test"})
                )
            else:
                route.continue_()

        if intercept:
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
            p.click("#narrativeNext")
        assert pre.value.status == 200, f"preprocess failed: {pre.value.status} {pre.value.text()[:300]}"
        p.wait_for_timeout(500)

    def select_advanced(self, select_id, model):
        p = self.page
        p.check("#advancedMode", force=True)
        p.evaluate("() => { if (window.switchModelingMode) switchModelingMode('advanced'); }")
        p.select_option("#" + select_id, model)
        p.wait_for_timeout(100)

    def visible(self, el_id):
        return self.page.evaluate(
            """(id) => {
            let el = document.getElementById(id);
            if (!el) return 'missing';
            for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
                if (getComputedStyle(n).display === 'none') return 'hidden:' + (n.id || n.className);
            }
            return 'visible';
        }""",
            el_id,
        )

    def toggle(self, el_id):
        # Switch-styled checkboxes sit off-screen behind a .slider span; set them directly.
        self.page.evaluate(
            """(id) => { const el = document.getElementById(id);
            el.checked = true; el.dispatchEvent(new Event('change', {bubbles: true})); }""",
            el_id,
        )

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


@pytest.fixture
def live_session(browser, base_url):
    """A session whose /process reaches the server, so a run can finish."""
    s = Session(browser, base_url, intercept=False)
    yield s
    s.close()


def _assert_visible(session, ids):
    states = {i: session.visible(i) for i in ids}
    bad = {k: v for k, v in states.items() if v != "visible"}
    assert not bad, f"controls not visible: {bad}"


def test_no_duplicate_ids(session):
    total, distinct = session.page.evaluate(
        "() => { const ids=[...document.querySelectorAll('[id]')].map(e=>e.id); return [ids.length, new "
        "Set(ids).size]; }"
    )
    assert total == distinct, f"{total - distinct} duplicate ids on the live page"
    assert not session.errors, session.errors


def test_ridge_advanced_controls_reach_payload(session):
    session.upload_and_preprocess("synthetic_regression.csv", "Numeric", "A-J", "K")
    session.select_advanced("advancedNModels", "Ridge")
    session.toggle("advancedNonreqRidgeSlider")
    session.page.select_option("#advancedRidgeSolver", "saga")
    session.page.fill("#advancedRidgeTol", "0.000001")
    session.page.fill("#advancedRidgeMaxIter", "5000")
    _assert_visible(
        session,
        [
            "advancedRidgeFields",
            "advancedNonreqRidgeSlider",
            "advancedRidgeSolver",
            "advancedRidgeTol",
            "advancedRidgeMaxIter",
        ],
    )
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
    _assert_visible(
        session, ["advancedNonreqLogisticClassifierSlider", "advancedClass_LogisticSolver", "advancedClass_LogisticTol"]
    )
    payload = session.submit_advanced()
    hp = payload.get("hyperparameters", {})
    print("hyperparameters:", json.dumps(hp))
    assert hp.get("Class_LogisticSolver") == "saga", hp
    assert hp.get("Class_LogisticTol") == pytest.approx(1e-6), hp
    assert not session.errors, session.errors


# --- The rail ------------------------------------------------------------
#
# The id inventory cannot catch a dead rail entry, because an id survives being
# moved into a rail that no longer navigates anywhere. These tests click the rail
# and assert the view actually changed.

RAIL_TABS = ["upload", "processing", "model-preprocessing", "modeling", "historic"]

# The container each rail entry is meant to reveal.
VIEW_OF_TAB = {
    "processing": "columnsection",
    "model-preprocessing": "userInputSection",
    "modeling": "columnSelection",
    "historic": "predictionDiv",
    "documentation": "documentation",
}


def _rail_entry(session, tab):
    return session.page.query_selector(f'#narrativeNav .tab-button[data-tab="{tab}"]')


def test_no_rail_before_a_file_is_loaded(session):
    """With nothing loaded the views take the page and the rail is not drawn."""
    p = session.page
    assert p.evaluate("() => document.getElementById('narrativePage').classList.contains('is-single')")
    assert p.evaluate("() => document.getElementById('narrativeRail').offsetParent === null"), (
        "the rail is drawn with no data"
    )
    # Documentation is in the banner, so it works with no data; the tutorial is a
    # section of the documentation, not a second banner entry.
    assert p.query_selector('.header-links [data-tab="documentation"]'), "Documentation left the banner"
    assert not p.query_selector("#tutorialButton"), "the banner carries a separate Tutorial button"
    assert p.query_selector("#doc-tutorial"), "the documentation has no tutorial section"
    # The bar the view buttons came from must not still be drawing them.
    assert p.evaluate(
        "() => { const b=document.getElementById('appTabs');"
        " return !b || b.querySelectorAll('.tab-button').length === 0; }"
    ), "the top tab bar still carries buttons"
    assert not session.errors, session.errors


def test_documentation_opens_full_width_with_no_data(session):
    p = session.page
    p.click('.header-links [data-tab="documentation"]')
    p.wait_for_timeout(400)
    assert session.visible("documentation") == "visible"
    assert p.evaluate("() => document.getElementById('narrativeRail').offsetParent === null")
    assert not session.errors, session.errors


def test_the_rail_appears_when_a_file_loads_and_every_entry_reaches_its_view(session):
    """A cold load has no rail; an upload brings it, and each entry then navigates."""
    p = session.page
    assert p.evaluate("() => document.getElementById('narrativeRail').offsetParent === null")

    session.upload_and_preprocess("iris.csv", "Classification", "A-D", "E")

    assert not p.evaluate("() => document.getElementById('narrativePage').classList.contains('is-single')"), (
        "the page stayed single-column after a file loaded"
    )
    assert p.evaluate("() => document.getElementById('narrativeRail').offsetParent !== null"), (
        "the rail did not appear after a file loaded"
    )
    for tab in RAIL_TABS:
        assert _rail_entry(session, tab), f"rail has no entry for {tab}"
        assert _rail_entry(session, tab).get_attribute("disabled") is None, f"{tab} still refuses"

    for tab, view in VIEW_OF_TAB.items():
        if tab == "documentation":
            p.click('.header-links [data-tab="documentation"]')
        else:
            _rail_entry(session, tab).click()
        p.wait_for_timeout(350)
        assert session.visible(view) == "visible", f"{tab} did not reveal {view}"
    assert not session.errors, session.errors


def test_the_empty_states_read_as_the_application(session):
    """Once the rail exists but before an upload: an empty data panel and no model catalog."""
    p = session.page
    p.evaluate("() => { const pg=document.getElementById('narrativePage'); pg.classList.remove('is-single'); }")
    assert p.inner_text("#narrativeDataCount").strip() == "0"
    assert "No data loaded" in p.inner_text("#narrativeDataBody")
    # The model is reported in the choices list, so the rail has no catalog panel.
    assert not p.query_selector("#narrativeAppsPanel"), "the rail still carries a model catalog"
    assert not session.errors, session.errors


def test_the_rail_gates_on_state_and_says_why(session):
    """Everything past Upload is refused, with a reason, until a file is loaded."""
    session.upload_and_preprocess("iris.csv", "Classification", "A-D", "E")
    # The rail exists now. Reset the session so the entries are gated again.
    p = session.page
    p.evaluate("() => { uploadedFileName = ''; const c=document.getElementById('columnList'); if (c) c.innerHTML=''; }")
    p.evaluate("() => { if (window.narrativeSync) narrativeSync(); }")
    for tab in RAIL_TABS[1:]:
        el = _rail_entry(session, tab)
        assert el.get_attribute("disabled") is not None, f"{tab} is reachable with no data"
        assert (el.get_attribute("title") or "").strip(), f"{tab} is disabled with no reason given"
    assert _rail_entry(session, "upload").get_attribute("disabled") is None, "Upload should stay reachable"

    session.upload_and_preprocess("iris.csv", "Classification", "A-D", "E")
    for tab in RAIL_TABS[1:]:
        el = _rail_entry(session, tab)
        assert el.get_attribute("disabled") is None, f"{tab} still gated after an upload"


def test_clicking_the_rail_changes_the_view(session):
    """The only check that catches a rail entry that no longer navigates."""
    session.upload_and_preprocess("iris.csv", "Classification", "A-D", "E")
    p = session.page
    for tab, view in VIEW_OF_TAB.items():
        if tab == "documentation":
            p.click('.header-links [data-tab="documentation"]')
        else:
            _rail_entry(session, tab).click()
        p.wait_for_timeout(350)
        assert session.visible(view) == "visible", f"entry {tab} did not reveal {view}"
        if tab != "documentation":
            active = p.evaluate(
                "() => { const b=document.querySelector('#narrativeNav .tab-button.active');"
                " return b ? b.dataset.tab : null; }"
            )
            assert active == tab, f"rail entry {tab} left {active} marked active"
    assert not session.errors, session.errors


def test_the_rail_carries_a_run_from_a_cold_load(live_session):
    """Cold load to a finished run, navigating only by the rail."""
    s = live_session
    p = s.page
    assert p.inner_text("#narrativeDataCount").strip() == "0"

    s.upload_and_preprocess("iris.csv", "Classification", "A-D", "E")
    # The data panel learned about the file without anything telling it to.
    assert p.inner_text("#narrativeDataCount").strip() == "1"
    assert "iris.csv" in p.inner_text("#narrativeDataBody")

    _rail_entry(s, "modeling").click()
    p.wait_for_timeout(350)
    assert s.visible("columnSelection") == "visible"
    # The strip is inside Modeling and is progress, not navigation.
    assert p.query_selector("#columnSelection #narrativeFlow"), "the run strip left the Modeling view"
    assert not p.query_selector("#narrativeRail #narrativeFlow"), "the run strip became rail navigation"

    # Choose the model on the page; the rail reports it under Modeling.
    p.select_option("#simpleClassModels", "Logistic_classifier")
    p.wait_for_timeout(200)
    label = p.eval_on_selector("#simpleClassModels", "e => e.options[e.selectedIndex].textContent.trim()")
    assert label in p.inner_text("#narrativeDataBody .narrative-choices"), "the rail does not report the chosen model"

    with p.expect_response(lambda r: r.url.endswith("/process"), timeout=120000) as run:
        p.click("#processButton", force=True)
    # A run is accepted and then streamed, so /process answers 202 and the result
    # arrives over the progress stream. Wait for the strip to reach Results, which
    # is the page's own statement that the run finished.
    assert run.value.status in (200, 202), f"run refused: {run.value.status} {run.value.text()[:200]}"
    p.wait_for_function("document.querySelector('#narrativeFlow .narrative-step[data-step=\"results\"].is-current') !== null", timeout=180000)

    # Inference is reachable from the rail once a model exists.
    _rail_entry(s, "historic").click()
    p.wait_for_timeout(350)
    assert s.visible("predictionDiv") == "visible"


def _figures_in_view(p, selector):
    """Every visible image matching the selector ends above the bottom of the window."""
    boxes = p.evaluate(
        "(sel) => Array.from(document.querySelectorAll(sel)).filter(e => e.offsetParent)"
        ".map(e => { const r = e.getBoundingClientRect(); return [Math.round(r.top), Math.round(r.bottom)]; })",
        selector,
    )
    inner = p.evaluate("window.innerHeight")
    assert boxes, f"no visible figure for {selector}"
    assert all(b <= inner and t >= 0 for t, b in boxes), f"{selector} at {boxes} against a {inner}px window"


def test_the_rail_carries_the_forward_step(session):
    """The continue button of each view is offered at the foot of the rail, with the
    view's own label and enabled state, and the in-page original is hidden."""
    session.upload_and_preprocess("iris.csv", "Classification", "A-D", "E")
    p = session.page

    _rail_entry(session, "processing").click()
    p.wait_for_timeout(350)
    assert p.text_content("#narrativeNext").strip() == "Continue to Model Preprocessing"
    assert p.is_visible("#narrativeNext")
    assert not p.is_visible("#continueToPreprocessButton")
    p.click("#narrativeNext")
    p.wait_for_timeout(350)
    assert session.visible(VIEW_OF_TAB["model-preprocessing"]) == "visible"
    assert p.text_content("#narrativeNext").strip() == "Continue to Modeling"

    # Continue to Modeling waits for a task type, and the proxy follows without a
    # page change, title included.
    p.evaluate(
        "() => { const t=document.getElementById('outputType1'); t.value=''; t.dispatchEvent(new Event('change')); }"
    )
    p.wait_for_timeout(200)
    assert p.is_disabled("#narrativeNext")
    assert "task type" in (p.get_attribute("#narrativeNext", "title") or "")
    p.select_option("#outputType1", "Classifier")
    p.wait_for_timeout(200)
    assert p.is_enabled("#narrativeNext")

    _rail_entry(session, "historic").click()
    p.wait_for_timeout(350)
    assert p.is_hidden("#narrativeNext"), "the last page has no forward step in the rail"
    assert not session.errors, session.errors


def test_the_rail_reports_the_model_from_modeling_on(session):
    """The Modeling lines join the choices list on Modeling and stay on Inference;
    Model Preprocessing, where no model is chosen yet, does not show them."""
    p = session.page
    session.upload_and_preprocess("iris.csv", "Classification", "A-D", "E")
    _rail_entry(session, "modeling").click()
    p.wait_for_timeout(400)
    p.select_option("#simpleClassModels", "Logistic_classifier")
    p.wait_for_timeout(200)
    label = p.eval_on_selector("#simpleClassModels", "e => e.options[e.selectedIndex].textContent.trim()")
    choices = p.inner_text("#narrativeDataBody .narrative-choices")
    assert "Mode " in choices and label in choices, choices

    _rail_entry(session, "model-preprocessing").click()
    p.wait_for_timeout(400)
    assert label not in p.inner_text("#narrativeDataBody"), "the model line shows before Modeling"

    _rail_entry(session, "modeling").click()
    p.wait_for_timeout(400)
    assert label in p.inner_text("#narrativeDataBody .narrative-choices")
    assert not session.errors, session.errors


def test_try_an_example_loads_penguins_and_opens_exploration(session):
    """The app has to be usable with no file of one's own."""
    p = session.page
    assert p.query_selector("#tryExampleButton"), "no Try an Example control"
    with p.expect_response(lambda r: r.url.endswith("/upload")) as up:
        p.click("#tryExampleButton")
    assert up.value.status == 200, f"example upload failed: {up.value.status}"
    p.wait_for_timeout(1200)
    assert p.evaluate("() => uploadedFileName") == "penguins.csv"
    assert session.visible("columnsection") == "visible", "Data Exploration did not open"
    # The rail arrives with the data, as it does for a chosen file.
    assert p.evaluate("() => document.getElementById('narrativeRail').offsetParent !== null")
    assert p.inner_text("#narrativeDataCount").strip() == "1"
    # Evaluate draws the two figures side by side, both in the window.
    with p.expect_response(lambda r: r.url.endswith("/correlationMatrices"), timeout=60000):
        p.click("#columnsection button[type=submit]")
    p.wait_for_selector("#pairplotImage", timeout=60000)
    p.wait_for_timeout(800)
    _figures_in_view(p, ".exploration-graphic img")
    assert not session.errors, session.errors


def test_pairplot_same_column_and_its_errors_stay_in_the_pairplot_panel(session):
    """2026-09-25: X and Y set to one column failed, and the message went to the Modeling form."""
    p = session.page
    with p.expect_response(lambda r: r.url.endswith("/upload")):
        p.click("#tryExampleButton")
    p.wait_for_timeout(1200)
    with p.expect_response(lambda r: r.url.endswith("/correlationMatrices"), timeout=60000):
        p.click("#columnsection button[type=submit]")
    p.wait_for_selector("#pairplotImage", timeout=60000)
    p.wait_for_timeout(800)
    column = p.eval_on_selector("#pairplotXSelect", "e => e.value")
    with p.expect_response(lambda r: r.url.endswith("/pairplot"), timeout=60000) as resp:
        p.select_option("#pairplotYSelect", column)
    assert resp.value.status == 200, f"same-column pairplot returned {resp.value.status}"
    p.wait_for_timeout(1200)
    assert p.inner_text("#errorDiv").strip() == ""
    assert "hidden" in (p.get_attribute("#pairplotError", "class") or "")
    assert not session.errors, session.errors

    # A failure is shown beside the pairplot, not in the Modeling form.
    p.route(
        "**/pairplot",
        lambda route: route.fulfill(status=500, content_type="application/json", body='{"error": "test failure"}'),
    )
    other = p.eval_on_selector("#pairplotXSelect", "e => [...e.options].map(o => o.value).find(v => v !== e.value)")
    p.select_option("#pairplotXSelect", other)
    p.wait_for_timeout(1200)
    assert "test failure" in p.inner_text("#pairplotError")
    assert p.inner_text("#errorDiv").strip() == ""
    p.unroute("**/pairplot")
    session.errors.clear()


# --- The example path, end to end -----------------------------------------

# The fields Try an Example fills in, and the tutorial's step for each.
EXAMPLE_FIELDS = {
    "outputType1": "Classifier",
    "indicators": "B-G",
    "predictors": "A",
    "imputeStrategy": "none",
    "useTransformer": "Yes",
    "transformerColumn": "B, G",
    "scalingYesNo": "Yes",
    "specificVariableSelect": "A",
    "seedValue": "33",
    "corrCols": "C-F",
}


def test_try_an_example_fills_the_run_in(session):
    """A reader should not have to type nine settings before anything will run."""
    p = session.page
    with p.expect_response(lambda r: r.url.endswith("/upload")):
        p.click("#tryExampleButton")
    p.wait_for_timeout(1500)
    p.evaluate("() => { if (window.showTab) showTab('model-preprocessing'); }")
    p.wait_for_timeout(800)

    for field, expected in EXAMPLE_FIELDS.items():
        actual = p.evaluate("(id) => { const e = document.getElementById(id); return e ? e.value : 'MISSING'; }", field)
        assert actual == expected, f"{field} is {actual!r}, expected {expected!r}"
    assert p.evaluate("() => document.getElementById('missScopeAll').checked"), "the blank scope is not set"

    # Stratify is open, because the run uses it. The cleaning
    # and transform controls stay behind their sentences until a reader asks: the
    # value is set either way, and the page says what it will do.
    assert p.evaluate("() => !document.getElementById('scalingYes').classList.contains('hidden')"), (
        "the stratify controls did not open"
    )
    for section in ("preprocessPanel_text", "preprocessPanel_cleaning"):
        assert p.evaluate("(id) => document.getElementById(id).classList.contains('hidden')", section), (
            f"{section} should stay behind its sentence"
        )
    assert not session.errors, session.errors


def test_a_persons_own_file_is_not_prefilled(session):
    """The values only make sense for the columns penguins.csv has."""
    p = session.page
    p.evaluate("() => { const b = document.getElementById('startModelingButton'); if (b) b.click(); }")
    p.set_input_files("#file", os.path.join(EXAMPLES, "iris.csv"))
    with p.expect_response(lambda r: r.url.endswith("/upload")):
        p.evaluate("() => document.getElementById('uploadForm').requestSubmit()")
    p.wait_for_timeout(1200)
    p.evaluate("() => { if (window.showTab) showTab('model-preprocessing'); }")
    p.wait_for_timeout(600)
    for field in ("indicators", "predictors", "seedValue"):
        assert p.evaluate("(id) => document.getElementById(id).value", field) == "", (
            f"{field} was filled for a file the app knows nothing about"
        )
    assert not session.errors, session.errors


def test_the_page_states_what_the_detection_found(session):
    """Model Preprocessing reports blanks and text columns without a button."""
    p = session.page
    with p.expect_response(lambda r: r.url.endswith("/upload")):
        p.click("#tryExampleButton")
    p.wait_for_timeout(1500)
    p.evaluate("() => { if (window.showTab) showTab('model-preprocessing'); }")
    p.wait_for_selector("#preprocessFindings .preprocess-finding", timeout=30000)
    text = p.inner_text("#preprocessFindings")
    assert "11 rows have a blank" in text, text
    assert "island and sex are text" in text, text
    # The controls stay behind the sentence until a reader asks for them.
    # Each sentence opens its own panel (pass 4); the wrapper itself is not toggled.
    panel_hidden = "() => document.getElementById('dataCleaningMissingWrapper').closest('.preprocess-panel').classList.contains('hidden')"
    assert p.evaluate(panel_hidden)
    p.click("#preprocessFindings .preprocess-change")
    p.wait_for_timeout(300)
    assert not p.evaluate(panel_hidden)
    assert not session.errors, session.errors


def test_the_example_path_runs_from_the_welcome_screen_to_a_prediction(live_session):
    """Cold load to 22 predictions, taking only the controls the tutorial names."""
    s = live_session
    p = s.page
    with p.expect_response(lambda r: r.url.endswith("/upload")):
        p.click("#tryExampleButton")
    p.wait_for_timeout(1500)

    p.evaluate("() => { if (window.showTab) showTab('model-preprocessing'); }")
    p.wait_for_timeout(1200)
    with p.expect_response(lambda r: r.url.endswith("/preprocess")) as pre:
        p.click("#narrativeNext")
    assert pre.value.status == 200, pre.value.text()[:300]
    p.wait_for_timeout(900)

    # The tutorial's model is chosen for the example once Modeling opens.
    p.wait_for_function("document.getElementById('simpleClassModels').value === 'RF_classifier'", timeout=5000)
    p.wait_for_timeout(400)
    with p.expect_response(lambda r: r.url.endswith("/process"), timeout=180000) as run:
        p.click("#processButton", force=True)
    assert run.value.status in (200, 202), run.value.text()[:300]
    p.wait_for_function("document.querySelector('#narrativeFlow .narrative-step[data-step=\"results\"].is-current') !== null", timeout=300000)
    p.wait_for_timeout(800)
    # The figures are the first thing in the results and both are in the window.
    _figures_in_view(p, ".result-graphic-box img")

    # The way on appears only once there is a model to apply. The page's own button
    # is hidden; the rail repeats it as the one forward step, with the same label.
    p.wait_for_selector("#continueToInferenceButton", state="attached", timeout=30000)
    p.wait_for_function("document.getElementById('narrativeNext').textContent === 'Continue to Inference on New Data'")
    assert p.is_visible("#narrativeNext")
    assert not p.is_visible("#continueToInferenceButton")
    p.click("#narrativeNext")
    p.wait_for_timeout(700)
    assert s.visible("predictionDiv") == "visible"
    assert p.is_hidden("#narrativeNext"), "no forward step from the last page"

    with p.expect_response(lambda r: r.url.endswith("/predict"), timeout=180000) as pred:
        p.click("#usePenguinRowsButton")
    assert pred.value.status == 200, pred.value.text()[:300]
    # The upload form gives way to the result, and the training figure is in the window.
    p.wait_for_selector(".inference-model-graphic-img", timeout=60000)
    p.wait_for_timeout(500)
    assert p.is_hidden("#uploadPredictDf")
    _figures_in_view(p, ".inference-model-graphic-img")
    p.wait_for_selector("#exampleRowAgreement", timeout=60000)

    line = p.inner_text("#exampleRowAgreement")
    match = re.search(r"(\d+) of (\d+) rows match", line)
    assert match, line
    agreed, total = int(match.group(1)), int(match.group(2))
    assert total == 22, line
    # The model scores near 0.99 on held-out real rows, so this is a loose floor.
    assert agreed >= 15, f"only {agreed} of {total} synthetic rows came back as drawn"
    assert not s.errors, s.errors
