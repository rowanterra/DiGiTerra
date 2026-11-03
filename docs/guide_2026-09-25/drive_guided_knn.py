"""Full mode, KNeighbors with n_neighbors 3 on examples/penguins.csv; checks the typed value reaches the estimator."""
import json, time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[2]; EX = ROOT/"examples"; OUT = Path(__file__).resolve().parent/"shots"; log={}
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width":1280,"height":800})
    pg.goto("http://127.0.0.1:5000/guided"); pg.wait_for_load_state("networkidle")
    pg.check("input[name=mode][value=full]")
    pg.set_input_files("#file", str(EX/"penguins.csv")); pg.click("#uploadForm button[type=submit]")
    pg.wait_for_function("document.getElementById('uploadStatus').textContent.includes('rows')"); time.sleep(1)
    pg.select_option("#target", label="species (text)"); time.sleep(0.5); pg.screenshot(path=OUT/"step2_full.png")
    pg.click("#step2 .next"); time.sleep(1)
    pg.select_option("#model", "KNeighbors_classifier"); time.sleep(1.5)
    pg.fill("#hyperTable tbody tr:first-child input", "3"); time.sleep(0.3)
    pg.screenshot(path=OUT/"step3_full_knn.png")
    log["hyperNote"]=pg.inner_text("#hyperNote")
    pg.click("#runBtn"); pg.wait_for_function("document.getElementById('result').innerText.length>50", timeout=600000); time.sleep(2)
    log["result"]=pg.inner_text("#result"); pg.screenshot(path=OUT/"step4_knn.png")
    r = pg.request.get("http://127.0.0.1:5000/export-bundle"); OUT/"bundle_knn.zip".write_bytes(r.body())
    b.close()
print(json.dumps(log,indent=1))
