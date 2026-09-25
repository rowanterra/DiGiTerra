"""Screenshots of the locally running DiGiTerra (127.0.0.1:5000) and H2O Flow (127.0.0.1:54321)
with Playwright and the pinned Chromium, 1280 by 800, white ground. Output: figures/screens/."""
import os, time
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(__file__).parent / "figures" / "screens"; OUT.mkdir(parents=True, exist_ok=True)
CHROME = os.environ.get("DIGITERRA_CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME)
    pg = b.new_page(viewport={"width": 1280, "height": 800})
    pg.goto("http://127.0.0.1:5000/guided"); pg.wait_for_load_state("networkidle"); time.sleep(1)
    pg.screenshot(path=OUT / "shot_digiterra_guided_step1.png")
    pg.goto("http://127.0.0.1:5000/"); pg.wait_for_load_state("networkidle"); time.sleep(1)
    pg.screenshot(path=OUT / "shot_digiterra_classic.png")
    pg.goto("http://127.0.0.1:54321/flow/index.html"); pg.wait_for_load_state("networkidle"); time.sleep(4)
    pg.screenshot(path=OUT / "shot_h2o_flow_start.png")
    # open the Assist Me menu and the build model form through Flow's own menu
    try:
        pg.click("text=Model"); time.sleep(1); pg.screenshot(path=OUT / "shot_h2o_flow_model_menu.png")
        pg.click("text=Generalized Linear Modeling"); time.sleep(3); pg.screenshot(path=OUT / "shot_h2o_flow_glm_form.png")
    except Exception as e:
        print("flow menu step failed:", e)
    b.close()
print(sorted(x.name for x in OUT.iterdir()))
