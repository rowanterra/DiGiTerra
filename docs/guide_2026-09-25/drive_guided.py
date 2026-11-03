"""Drive the guided page with examples/penguins.csv and record what it shows.

Needs `python app.py` running on 127.0.0.1:5000 and Playwright with Chromium.
Writes shots/step1.png ... classic.png, shots/drive_log_rf.json and shots/bundle_rf.zip.
The screenshots in the guide are step4.png, step3_full_knn.png (drive_guided_knn.py) and classic.png."""
import json, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[2]; EX = ROOT/"examples"; OUT = Path(__file__).resolve().parent/"shots"
log = {}
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium") if Path("/opt/pw-browsers/chromium").is_file() else p.chromium.launch()
    pg = b.new_page(viewport={"width":1280,"height":800}, accept_downloads=True)
    pg.goto("http://127.0.0.1:5000/guided"); pg.wait_for_load_state("networkidle")
    pg.set_input_files("#file", str(EX/"penguins.csv")); pg.screenshot(path=OUT/"step1.png"); pg.click("#uploadForm button[type=submit]")
    pg.wait_for_function("document.getElementById('uploadStatus').textContent.includes('rows')", timeout=60000)
    log["upload"] = pg.inner_text("#uploadStatus"); time.sleep(1)
    pg.select_option("#target", label="species (text)"); time.sleep(1)
    log["targetHint"] = pg.inner_text("#targetHint"); pg.screenshot(path=OUT/"step2.png")
    pg.click("#step2 .next"); time.sleep(1.5)
    log["task"] = pg.eval_on_selector("input[name=task]:checked", "e=>e.value")
    log["model"] = pg.eval_on_selector("#model", "e=>e.value")
    pg.screenshot(path=OUT/"step3_guided.png")
    pg.check("input[name=mode][value=full]"); time.sleep(1.5); pg.screenshot(path=OUT/"step3_full.png")
    log["hyperNote"] = pg.inner_text("#hyperNote")
    log["hyper_rows"] = pg.eval_on_selector_all("#hyperTable tbody tr", "rs=>rs.map(r=>[...r.cells].slice(0,2).map(c=>c.innerText.trim()))")
    pg.check("input[name=mode][value=guided]"); time.sleep(1)
    pg.click("#runBtn")
    pg.wait_for_function("document.getElementById('result').innerText.length>50", timeout=600000)
    time.sleep(2)
    time.sleep(1); log["result"] = pg.inner_text("#result"); pg.screenshot(path=OUT/"step4.png")
    pg.set_input_files("#predictFile", str(EX/"penguins_new.csv")); pg.click("#predictBtn")
    pg.wait_for_function("document.getElementById('predictStatus').textContent.length>10", timeout=120000)
    time.sleep(1); log["predict"] = pg.inner_text("#predictStatus"); pg.screenshot(path=OUT/"step4_predict.png")
    log["raw"] = pg.inner_text("#resultRaw")
    log["links"] = pg.eval_on_selector_all("#step4 a", "as=>as.map(a=>[a.innerText,a.getAttribute('href')])")
    r = pg.request.get("http://127.0.0.1:5000/export-bundle"); OUT/"bundle_rf.zip".write_bytes(r.body()); log["bundle_status"]=r.status
    pg.goto("http://127.0.0.1:5000/"); pg.wait_for_load_state("networkidle"); time.sleep(1); pg.screenshot(path=OUT/"classic.png")
    b.close()
OUT/"drive_log_rf.json".write_text(json.dumps(log, indent=1))
print(json.dumps({k:(v if k!='raw' else v[:3000]) for k,v in log.items()}, indent=1))
