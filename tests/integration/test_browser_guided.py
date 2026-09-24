"""The guided page, driven through Chromium: keyboard walk, live regions, a full run.

Skipped when Playwright or its Chromium is absent. The run goes iris ->
species -> k-NN classifier with n_neighbors typed as 3 in Full mode, and the
test reads the fitted n_neighbors back from the result, which is the E1
finding F3 checked through the page rather than the route.
"""

import os
import re
import socket
import threading

import pytest

pw = pytest.importorskip("playwright.sync_api")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def server():
    os.environ.setdefault("DIGITERRA_ISOLATE_WORK", "0")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from werkzeug.serving import make_server

    from app import app

    app.config["TESTING"] = True
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = make_server("127.0.0.1", port, app, threaded=True)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


@pytest.fixture(scope="module")
def page(server):
    try:
        p = pw.sync_playwright().start()
        exe = os.environ.get("DIGITERRA_CHROMIUM")  # a pinned Chromium when Playwright's own is absent
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Chromium not available: {e}")
    pg = browser.new_page()
    yield pg
    browser.close()
    p.stop()


def test_landmarks_labels_and_keyboard_order(server, page):
    page.goto(server + "/guided")
    assert page.locator("main#main").count() == 1
    assert page.locator("header").count() == 1
    assert page.locator("a.skip").count() == 1
    # every form control has a label
    ids = page.eval_on_selector_all("input:not([type=radio]):not([type=hidden]), select", "els => els.map(e => e.id)")
    for i in ids:
        assert i, "control without id"
        assert page.locator(f"label[for='{i}']").count() == 1, f"no label for #{i}"
    # radios sit in fieldsets with legends
    assert page.locator("fieldset.mode legend").count() == 1
    # the alert region exists and is empty
    assert page.locator("#errors[role=alert]").count() == 1
    # keyboard: tab from the skip link reaches the mode radio, then the file input
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.className") == "skip"
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.name") == "mode"
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.tagName") == "A"  # the Classic page link
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.id") == "file"


def test_mode_switch_is_a_radio_group_and_persists(server, page):
    page.goto(server + "/guided")
    assert page.evaluate("document.body.dataset.mode") == "guided"
    page.check("input[name=mode][value=full]")
    assert page.evaluate("document.body.dataset.mode") == "full"
    page.reload()
    assert page.evaluate("document.body.dataset.mode") == "full"
    page.check("input[name=mode][value=guided]")


def test_full_run_and_typed_hyperparameter_reaches_the_estimator(server, page):
    page.goto(server + "/guided")
    page.check("input[name=mode][value=full]")
    page.set_input_files("#file", os.path.join(ROOT, "examples", "iris.csv"))
    page.click("#uploadForm button[type=submit]")
    page.wait_for_function("document.getElementById('uploadStatus').textContent.includes('rows')", timeout=30000)
    # focus moved to step 2's heading
    assert page.evaluate("document.activeElement.id") == "h2s"
    assert "classification" in page.text_content("#targetHint")
    page.select_option("#model", "KNeighbors_classifier")
    page.wait_for_selector("#hp_n_neighbors", timeout=10000)
    page.fill("#hp_n_neighbors", "3")
    page.dispatch_event("#hp_n_neighbors", "change")
    page.click("#runBtn")
    page.wait_for_function("document.getElementById('result').textContent.includes('Accuracy')", timeout=180000)
    raw = page.text_content("#resultRaw")
    m = re.search(r'"n_neighbors":\s*(\d+)', raw)
    assert m and m.group(1) == "3", "the typed n_neighbors did not reach the estimator"
    assert page.locator("#result table caption").first.text_content() == "Metrics"
    assert page.evaluate("document.activeElement.id") == "h5s"


def test_axe_core_has_no_serious_violations(server, page):
    page.goto(server + "/guided")
    axe = os.path.join(ROOT, "node_modules", "axe-core", "axe.min.js")
    if not os.path.exists(axe):
        pytest.skip("axe-core not installed (npm install)")
    page.add_script_tag(path=axe)
    tags = "['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']"
    res = page.evaluate("axe.run(document, {runOnly: {type: 'tag', values: " + tags + "}})")
    serious = [v for v in res["violations"] if v["impact"] in ("serious", "critical")]
    assert not serious, [(v["id"], v["help"], [n["target"] for n in v["nodes"]]) for v in serious]
