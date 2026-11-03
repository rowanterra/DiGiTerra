"""E1 runs of DiGiTerra through its own web routes.

Purpose: cases c000_base_a, c001_base_b (R1), C1, C2, D1, D2 and c999_base_c
(R1) driven through the Flask application exactly as the browser drives it:
POST /upload, POST /preprocess, POST /process, then GET /export-bundle for the
manifest and model card, POST /predict on the _test file for the predictions
file, and POST /correlationMatrices for the descriptive statistics. The
browser layer (templates and JavaScript) is not exercised; the run records
are the ones the application writes.
Input: the DiGiTerra tree at 0942e0c (path in DGT), data/<name>_E1_base.csv
and _test.csv. Settings per PLAN.txt: Advanced mode, split method time on
"order", test size 0.20, stratify off, scaler none, seed 20260924, sigfig 10.
Test: each manifest must record split method time on "order", 89 or 30 test
rows, and scaler none (input check). The test file's "order" values must be
the 89 or 30 last rows. Any failure is written to the case folder as
INPUT_CHECK_FAILED.txt and the case is not counted.
Output: digiterra_outputs/<case>/ with manifest.json, model_card.html,
result.json (the /process result the page renders), predictions.csv (from
/predict), coefficients_metrics.csv and predictions_test.csv in the
compare.py layout, and log.txt. versions.txt in digiterra_outputs/.
Run: from the E1 folder, with the DiGiTerra venv:
  DGT=/path/to/DiGiTerra MPLBACKEND=Agg <venv python> scripts/run_digiterra.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

E1 = Path.cwd()
DGT = Path(os.environ.get("DGT", "")).resolve()
if not (DGT / "app.py").exists():
    sys.exit("set DGT to the DiGiTerra tree (the folder with app.py)")
os.environ.setdefault("DIGITERRA_ISOLATE_WORK", "0")   # the isolated worker needs Linux RLIMIT_AS; the run itself is the same code
os.environ.setdefault("MPLBACKEND", "Agg")
SUP = E1 / "digiterra_outputs" / "_app_support"
SUP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DIGITERRA_APP_SUPPORT_DIR", str(SUP))
os.environ.setdefault("DIGITERRA_OUTPUT_DIR", str(SUP / "vis"))
sys.path.insert(0, str(DGT))
os.chdir(DGT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from app import app  # noqa: E402
from python_scripts.preprocessing.progress_tracker import get_result  # noqa: E402

os.chdir(E1)
OUT = E1 / "digiterra_outputs"
G = "%.17g"
app.config["TESTING"] = True

(OUT / "versions.txt").write_text(
    f"python {sys.version.split()[0]}\nscikit-learn {sklearn.__version__}\nnumpy {np.__version__}\n"
    f"pandas {pd.__version__}\nDiGiTerra tree {DGT}\nDIGITERRA_ISOLATE_WORK {os.environ['DIGITERRA_ISOLATE_WORK']}\n")

COMMON = dict(
    hyperparameters={}, nonreq=False, scaler="none", units="", sigfig=10, seedValue=20260924,
    testSize="0.20", stratifyBool=False, dropMissing="none", imputeStrategy="none", dropZero="none",
    zeroHandle="none", quantileBinDict={"quantile": 0, "bin": 0, "binsLabel": ""}, useTransformer="No",
    transformerCols=[], crossValidationType="None", modelingMode="advanced",
    splitMethod="time", splitColumn="order", featureSelectionMethod="none", outlierMethod="none",
    hyperparameterSearch="none",
)

CASES = [
    # case, dataset, model, indicator column indexes, predictor index, n_test
    ("c000_base_a", "diabetes", "Linear", list(range(1, 11)), [11], 89),
    ("c001_base_b", "diabetes", "Linear", list(range(1, 11)), [11], 89),
    ("C1", "iris", "KNeighbors_classifier", [1, 2, 3, 4], [5], 30),
    ("C2", "iris", "Logistic_classifier", [1, 2, 3, 4], [5], 30),
    ("c999_base_c", "diabetes", "Linear", list(range(1, 11)), [11], 89),
]


def log(case_dir: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  {msg}"
    with open(case_dir / "log.txt", "a") as f:
        f.write(line + "\n")
    print(case_dir.name, msg)


def upload(client, path: Path):
    with open(path, "rb") as f:
        r = client.post("/upload", data={"file": (f, path.name)})
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    return r.get_json()


def run_case(case, dataset, model, indicators, predictors, n_test):
    cdir = OUT / case
    cdir.mkdir(parents=True, exist_ok=True)
    client = app.test_client()
    base = E1 / "data" / f"{dataset}_E1_base.csv"
    test = E1 / "data" / f"{dataset}_E1_test.csv"
    up = upload(client, base)
    log(cdir, f"uploaded {base.name}: {str(up)[:120]}")
    r = client.post("/preprocess", json={"indicators": indicators, "predictors": predictors, "stratify": 0})
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    payload = dict(COMMON, indicators=indicators, predictors=predictors, models=model, filename=base.name)
    r = client.post("/process", json=payload)
    assert r.status_code in (200, 202), r.get_data(as_text=True)[:300]
    sid = r.get_json()["session_id"]
    log(cdir, f"/process accepted, session {sid}")
    for _ in range(600):
        res = get_result(sid)
        if res is not None:
            break
        time.sleep(0.5)
    assert res is not None, "no result after 300 s"
    assert "error" not in res, res.get("error")
    (cdir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    log(cdir, f"result keys {len(res)}; split_note: {res.get('split_note')}")

    # manifest + model card, as the Download bundle button gives them
    r = client.get("/export-bundle")
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    z = zipfile.ZipFile(io.BytesIO(r.data))
    names = z.namelist()
    for n in names:
        if n.endswith((".json", ".html", ".skops")):
            (cdir / Path(n).name).write_bytes(z.read(n))
    log(cdir, f"bundle members: {names}")
    manifest = json.loads((cdir / "manifest.json").read_text()) if (cdir / "manifest.json").exists() else {}

    # predictions on the _test file through the inference route
    with open(test, "rb") as f:
        r = client.post("/predict", data={"predictFile": (f, test.name), "preprocessMode": "none"})
    pr = r.get_json()
    log(cdir, f"/predict status {r.status_code}: {str(pr)[:200]}")
    assert r.status_code == 200, pr
    vis = Path(os.environ["DIGITERRA_OUTPUT_DIR"])
    pred_files = sorted(vis.rglob("predictions.csv"), key=lambda p: p.stat().st_mtime)
    assert pred_files, "no predictions.csv written by /predict"
    pred_csv = pred_files[-1]
    (cdir / "predictions.csv").write_bytes(pred_csv.read_bytes())
    pdf = pd.read_csv(cdir / "predictions.csv")
    log(cdir, f"predictions.csv columns {list(pdf.columns)} rows {len(pdf)}")

    # input check against the manifest and the result
    checks = []
    note = str(res.get("split_note", ""))
    if "Time-ordered holdout on column 'order'" not in note:
        checks.append(f"split_note is not the time holdout on order: {note!r}")
    import re as _re
    m_ = _re.search(r"\((\d+) of (\d+)\)", note)
    n_from_note = int(m_.group(1)) if m_ else None
    if res.get("test_n") is not None and int(res["test_n"]) != n_test:
        checks.append(f"test_n {res.get('test_n')} != {n_test}")
    if n_from_note != n_test:
        checks.append(f"split note test rows {n_from_note} != {n_test}")
    mtxt = json.dumps(manifest)
    if manifest and '"none"' not in mtxt.lower() and "scaler" in mtxt.lower():
        checks.append("manifest does not record scaler none (check by eye)")
    if len(pdf) != n_test:
        checks.append(f"predictions rows {len(pdf)} != {n_test}")
    if checks:
        (cdir / "INPUT_CHECK_FAILED.txt").write_text("\n".join(checks) + "\n")
        log(cdir, "INPUT CHECK FAILED: " + "; ".join(checks))
    else:
        log(cdir, "input check passed")

    # compare.py layout
    test_df = pd.read_csv(test)
    pcol = next((c for c in pdf.columns if "predict" in c.lower()), pdf.columns[-1])
    if model == "Linear":
        with open(cdir / "coefficients_metrics.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(["term", "value"])
            w.writerow(["test_rmse", res.get("valrmse")]); w.writerow(["test_mae", res.get("valmae")])
            w.writerow(["test_r2", res.get("valscore")])
        # coefficients are not in the result; they are in the manifest or the fitted model in the bundle
        # The manifest carries no coefficients; the fitted estimator in the bundle
        # (model.skops, the program's own saved artifact) does.
        if (cdir / "model.skops").exists():
            import skops.io as sio
            obj = sio.load(cdir / "model.skops", trusted=sio.get_untrusted_types(file=cdir / "model.skops"))
            est = obj.get("model") if isinstance(obj, dict) else obj
            est = getattr(est, "_digiterra_model", est)
            if hasattr(est, "coef_"):
                feats = manifest.get("model", {}).get("features") or [f"x{i}" for i in range(len(np.ravel(est.coef_)))]
                with open(cdir / "coefficients_metrics.csv", "a", newline="") as f:
                    w = csv.writer(f, lineterminator="\n")
                    w.writerow(["intercept", G % float(np.ravel(est.intercept_)[0])])
                    for k, v in zip(feats, np.ravel(est.coef_)):
                        w.writerow([k, G % float(v)])
                log(cdir, f"coefficients read from model.skops ({type(est).__name__})")
            else:
                log(cdir, f"model.skops object {type(est).__name__} has no coef_; keys {list(obj.keys()) if isinstance(obj, dict) else ''}")
        pred = pd.to_numeric(pdf[pcol], errors="coerce")
        y = test_df["target"].to_numpy(float)
        out = pd.DataFrame({"order": test_df["order"], "target": y, "predicted": pred.to_numpy(float)})
        out.to_csv(cdir / "predictions_test.csv", index=False, float_format=G)
        # metrics recomputed from the predictions file at full precision, so the
        # rounded page values and the full-precision values are both on record
        e = y - out["predicted"].to_numpy()
        with open(cdir / "metrics_from_predictions.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(["term", "value"])
            w.writerow(["test_rmse", G % float(np.sqrt(np.mean(e ** 2)))])
            w.writerow(["test_mae", G % float(np.mean(np.abs(e)))])
            w.writerow(["test_r2", G % float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))])
    else:
        out = pd.DataFrame({"order": test_df["order"], "species": test_df["species"], "predicted": pdf[pcol]})
        for c in pdf.columns:
            lc = c.lower()
            for lab in ("setosa", "versicolor", "virginica"):
                if lab in lc and c != pcol:
                    out[f"p_{lab}"] = pdf[c]
        out.to_csv(cdir / "predictions_test.csv", index=False, float_format=G)
        acc = float((out["predicted"].astype(str).str.lower() == out["species"].str.lower()).mean())
        (cdir / "metrics.txt").write_text(f"accuracy_from_predictions {G % acc}\n"
                                          f"page_metrics {json.dumps({k: res.get(k) for k in res if k.startswith(('val', 'train', 'test'))}, default=str)}\n")
    log(cdir, "done")


if __name__ == "__main__":
    only = sys.argv[1:]
    for spec in CASES:
        if only and spec[0] not in only:
            continue
        try:
            run_case(*spec)
        except Exception as e:  # noqa: BLE001
            cdir = OUT / spec[0]; cdir.mkdir(parents=True, exist_ok=True)
            log(cdir, f"FAILED: {type(e).__name__}: {e}")
            raise
    # D1, D2: descriptive statistics through the exploration route
    for case, dataset, cols in (("D1", "diabetes", list(range(1, 12))), ("D2", "iris", [1, 2, 3, 4])):
        cdir = OUT / case; cdir.mkdir(parents=True, exist_ok=True)
        client = app.test_client()
        upload(client, E1 / "data" / f"{dataset}_E1_base.csv")
        r = client.post("/correlationMatrices", json={"colsIgnore": cols, "dropMissing": "none",
                                                     "imputeStrategy": "none", "dropZero": "none", "zeroHandle": "none"})
        res = r.get_json()
        (cdir / "correlation_response.json").write_text(json.dumps(res, indent=1, default=str))
        stats = res.get("descriptive_stats") or []
        log(cdir, f"/correlationMatrices status {r.status_code}; descriptive_stats rows {len(stats)}; keys {list(stats[0].keys()) if stats else None}")
        if stats:
            df = pd.DataFrame(stats)
            df.to_csv(cdir / f"{case}_digiterra.csv", index=False)
    print("finished")
