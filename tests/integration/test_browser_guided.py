"""The guided page, driven through Chromium: keyboard walk, live regions, a full run,
no page scrolling, and an explanation on every control.

Skipped when Playwright or its Chromium is absent; DIGITERRA_CHROMIUM names a
Chromium binary to use instead of Playwright's own. The run goes iris ->
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
VIEWPORTS = [(1280, 800), (1366, 768), (1440, 900)]


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
    pg = browser.new_page(viewport={"width": 1280, "height": 800})
    yield pg
    browser.close()
    p.stop()


def no_scroll(page):
    """True when neither the page nor the visible step needs a scrollbar."""
    return page.evaluate(
        "document.documentElement.scrollHeight <= window.innerHeight + 1 && "
        "document.documentElement.scrollWidth <= window.innerWidth + 1 && "
        "(() => { const s = document.querySelector('section.step.current'); "
        "return s.scrollHeight <= s.clientHeight + 1; })()"
    )


def upload_iris(page):
    page.set_input_files("#file", os.path.join(ROOT, "examples", "iris.csv"))
    page.click("#uploadForm button[type=submit]")
    page.wait_for_function("document.getElementById('uploadStatus').textContent.includes('rows')", timeout=30000)


def test_landmarks_labels_and_keyboard_order(server, page):
    page.goto(server + "/guided")
    assert page.locator("main#main").count() == 1
    assert page.locator("header").count() == 1
    assert page.locator("a.skip").count() == 1
    assert page.locator("nav.steps button[aria-current=step]").text_content().startswith("1")
    ids = page.eval_on_selector_all("input:not([type=radio]):not([type=hidden]), select", "els => els.map(e => e.id)")
    for i in ids:
        assert i, "control without id"
        assert page.locator(f"label[for='{i}']").count() == 1, f"no label for #{i}"
    assert page.locator("fieldset.mode legend").count() == 1
    assert page.locator("#errors[role=alert]").count() == 1
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.className") == "skip"
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.name") == "mode"


def test_every_control_has_an_explanation(server, page):
    page.goto(server + "/guided")
    page.check("input[name=mode][value=full]")
    ids = page.eval_on_selector_all("input:not([type=radio]), select", "els => els.map(e => e.id)")
    for i in ids:
        desc = page.get_attribute(f"#{i}", "aria-describedby")
        assert desc, f"#{i} has no explanation"
        assert page.locator(f"#{desc}[role=tooltip]").count() == 1
        assert len(page.text_content(f"#{desc}")) > 20
    btn = page.locator("label[for=file] button.help")
    btn.focus()
    assert page.locator("label[for=file] .tip").is_visible()
    page.locator("h1").click()
    assert not page.locator("label[for=file] .tip").is_visible()
    page.check("input[name=mode][value=guided]")


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
    upload_iris(page)
    assert page.evaluate("document.activeElement.id") == "h2s"
    assert "classification" in page.text_content("#targetHint")
    page.click("nav.steps button[data-step='3']")
    assert page.evaluate("document.activeElement.id") == "h3s"
    page.select_option("#model", "KNeighbors_classifier")
    page.wait_for_selector("#hp_n_neighbors", timeout=10000)
    assert page.get_attribute("#hp_n_neighbors", "aria-describedby")
    page.fill("#hp_n_neighbors", "3")
    page.dispatch_event("#hp_n_neighbors", "change")
    page.click("#runBtn")
    page.wait_for_function("document.getElementById('result').textContent.includes('Accuracy')", timeout=180000)
    raw = page.text_content("#resultRaw")
    m = re.search(r'"n_neighbors":\s*(\d+)', raw)
    assert m and m.group(1) == "3", "the typed n_neighbors did not reach the estimator"
    assert page.locator("#result table caption").first.text_content() == "Metrics"
    assert page.evaluate("document.activeElement.id") == "h4s"
    assert page.locator("#result th[data-metric=accuracy] button.help").count() == 1


def test_no_step_needs_scrolling(server, page):
    """Every step, in both modes, at three common laptop viewports, fits its window."""
    for w, h in VIEWPORTS:
        page.set_viewport_size({"width": w, "height": h})
        for mode in ("guided", "full"):
            page.goto(server + "/guided")
            page.check(f"input[name=mode][value={mode}]")
            assert no_scroll(page), f"step 1 scrolls at {w}x{h} in {mode}"
            upload_iris(page)
            assert no_scroll(page), f"step 2 scrolls at {w}x{h} in {mode}"
            page.click("nav.steps button[data-step='3']")
            page.select_option("#model", "GradientBoosting_classifier")  # the longest hyperparameter table
            page.wait_for_selector("#hp_n_estimators", state="attached", timeout=10000)
            assert no_scroll(page), f"step 3 scrolls at {w}x{h} in {mode}"
    page.set_viewport_size({"width": 1280, "height": 800})


def test_result_step_fits_the_window(server, page):
    page.goto(server + "/guided")
    page.check("input[name=mode][value=guided]")
    upload_iris(page)
    page.click("nav.steps button[data-step='3']")
    page.select_option("#model", "KNeighbors_classifier")
    page.click("#runBtn")
    page.wait_for_function("document.getElementById('result').textContent.includes('Accuracy')", timeout=180000)
    assert no_scroll(page), "the result step scrolls"


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
