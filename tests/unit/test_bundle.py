"""Run manifest, model card, bundle export, gated import, and the two scripts.

Training runs through run_model_training with a dict-backed store, as the app
does, so the manifest is the one the app would write. Export and import go
through the Flask test client.
"""

import io
import json
import subprocess
import sys
import uuid
import warnings
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from core.state import get_session_storage
from python_scripts import config
from python_scripts.preprocessing.progress_tracker import get_result, remove_tracker
from python_scripts.reporting.bundle import (
    BUNDLE_FORMAT,
    BundleError,
    load_bundle,
    read_manifest,
    render_model_card,
    sha256_file,
)
from tests.unit.test_pipelines import BASE

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DIABETES = ROOT / "examples" / "diabetes.csv"


@pytest.fixture
def client():
    from app import app

    app.config["TESTING"] = True
    return app.test_client()


def _train_in_session(sid: str, extra=None) -> dict:
    """Upload-equivalent store, then a Linear run scoped to the session directory."""
    from python_scripts.app_model_training import run_model_training

    store = get_session_storage(sid)
    store.clear()
    df = pd.read_csv(DIABETES)
    store["data"] = df
    store["data_fingerprint"] = {
        "filename": "diabetes.csv",
        "sha256": sha256_file(DIABETES),
        "bytes": DIABETES.stat().st_size,
        "rows": len(df),
        "columns": len(df.columns),
        "uploaded_at": "2026-09-04T00:00:00+00:00",
    }
    n = len(df.columns)
    payload = dict(BASE, indicators=list(range(n - 1)), predictors=[n - 1], models="Linear", seedValue=7)
    payload.update(extra or {})
    token = config.set_session_scope(sid)
    try:
        run_model_training(sid, payload, sid, lambda s: get_session_storage(s))
        result = get_result(sid)
    finally:
        config.reset_session_scope(token)
    remove_tracker(sid)
    assert result is not None and "error" not in result, result
    return result


def test_manifest_and_model_card_are_written_with_the_run():
    sid = uuid.uuid4().hex
    result = _train_in_session(sid)
    assert result["model_card_url"].endswith("model_card.html")
    token = config.set_session_scope(sid)
    try:
        vis_dir = config.VIS_DIR.current()
    finally:
        config.reset_session_scope(token)
    manifest = json.loads((vis_dir / "manifest.json").read_text())
    assert manifest["format"] == BUNDLE_FORMAT
    assert manifest["data"]["sha256"] == sha256_file(DIABETES)
    assert manifest["model"]["name"] == "Linear" and manifest["model"]["seed"] == 7
    assert manifest["metrics"]["valscore"] == result["valscore"]
    assert manifest["environment"]["packages"]["scikit-learn"]
    assert manifest["baseline"]["kind"] == "regression"
    assert manifest["request"]["testSize"] == BASE["testSize"]
    card = (vis_dir / "model_card.html").read_text()
    assert "<script" not in card
    assert manifest["data"]["sha256"] in card
    assert "R² (validation)" in card


def test_export_then_import_round_trip_and_predict(client, monkeypatch):
    sid = uuid.uuid4().hex
    _train_in_session(sid)
    client.set_cookie("digiterra_session", sid)

    r = client.get("/export-bundle")
    assert r.status_code == 200, r.get_json()
    assert r.headers["Content-Disposition"].endswith("digiterra_Linear_bundle.zip")
    names = set(zipfile.ZipFile(io.BytesIO(r.data)).namelist())
    assert {"manifest.json", "model_card.html", "model_card.json", "model.skops"} <= names
    manifest = read_manifest(io.BytesIO(r.data))
    assert manifest["artifact"]["sha256"]

    # A fresh session with import disabled: refused before the file is read.
    other = uuid.uuid4().hex
    client.set_cookie("digiterra_session", other)
    monkeypatch.delenv("DIGITERRA_ALLOW_BUNDLE_IMPORT", raising=False)
    r2 = client.post("/import-bundle", data={"bundle": (io.BytesIO(r.data), "b.zip")})
    assert r2.status_code == 403

    monkeypatch.setenv("DIGITERRA_ALLOW_BUNDLE_IMPORT", "1")
    r3 = client.post("/import-bundle", data={"bundle": (io.BytesIO(r.data), "b.zip")})
    assert r3.status_code == 200, r3.get_json()
    assert r3.get_json()["model"]["name"] == "Linear"
    store = get_session_storage(other)
    assert store["model"] is not None and store["model_type"] == "regression"

    rows = pd.read_csv(DIABETES).iloc[:5, :-1]
    buf = io.BytesIO(rows.to_csv(index=False).encode())
    r4 = client.post("/predict", data={"predictFile": (buf, "new.csv")})
    assert r4.status_code == 200, r4.get_json()
    assert r4.get_json()["summary"][0]["n"] == 5


def test_altered_artifact_is_refused(client):
    sid = uuid.uuid4().hex
    _train_in_session(sid)
    client.set_cookie("digiterra_session", sid)
    data = client.get("/export-bundle").data
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for name in src.namelist():
            payload = src.read(name)
            if name == "model.skops":
                payload = payload + b"\x00"
            z.writestr(name, payload)
    out.seek(0)
    with pytest.raises(BundleError, match="does not match"):
        load_bundle(out, force=True)


def test_bundle_from_another_installation_is_refused(client, monkeypatch, tmp_path):
    """A valid hash is not authenticity: a foreign key, or no signature, is refused before unpickling."""
    import python_scripts.reporting.bundle as bundle_mod

    sid = uuid.uuid4().hex
    _train_in_session(sid)
    client.set_cookie("digiterra_session", sid)
    monkeypatch.setenv("DIGITERRA_BUNDLE_KEY", "attacker-key")
    data = client.get("/export-bundle").data  # signed under the attacker's key with a matching hash
    monkeypatch.setenv("DIGITERRA_BUNDLE_KEY", "victim-key")
    with pytest.raises(BundleError, match="signature does not verify"):
        load_bundle(io.BytesIO(data), force=True)
    monkeypatch.setenv("DIGITERRA_ALLOW_BUNDLE_IMPORT", "1")
    r = client.post("/import-bundle", data={"bundle": (io.BytesIO(data), "b.zip")})
    assert r.status_code == 400 and "signature" in r.get_json()["error"]

    # An unsigned bundle (signature stripped from the manifest) is refused as well.
    src = zipfile.ZipFile(io.BytesIO(data))
    manifest = json.loads(src.read("manifest.json"))
    manifest["artifact"].pop("signature")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for name in src.namelist():
            z.writestr(name, json.dumps(manifest).encode() if name == "manifest.json" else src.read(name))
    out.seek(0)
    with pytest.raises(BundleError, match="signature does not verify"):
        load_bundle(out, force=True)
    # The explicit command-line escape loads it.
    out.seek(0)
    manifest2, objects = load_bundle(out, force=True, trust_unsigned=True)
    assert objects["model"] is not None
    assert bundle_mod.key_id(b"victim-key") != manifest2["artifact"]["key_id"]


def test_signing_key_file_is_created_once(monkeypatch, tmp_path):
    import python_scripts.reporting.bundle as bundle_mod

    monkeypatch.delenv("DIGITERRA_BUNDLE_KEY", raising=False)
    monkeypatch.setattr(bundle_mod, "_key_file", lambda: tmp_path / "bundle_signing.key")
    k1 = bundle_mod.signing_key()
    k2 = bundle_mod.signing_key()
    assert k1 == k2 and len(k1) == 64
    assert bundle_mod.signing_key(create=False) == k1
    assert bundle_mod.signature_valid(b"abc", bundle_mod.sign_bytes(b"abc", k1), k1)
    assert not bundle_mod.signature_valid(b"abc", bundle_mod.sign_bytes(b"abc", k1), b"other")


def test_import_rejects_a_zip_without_a_manifest(client, monkeypatch):
    monkeypatch.setenv("DIGITERRA_ALLOW_BUNDLE_IMPORT", "1")
    client.set_cookie("digiterra_session", uuid.uuid4().hex)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("readme.txt", "not a bundle")
    out.seek(0)
    r = client.post("/import-bundle", data={"bundle": (out, "x.zip")})
    assert r.status_code == 400
    # The directory check names the stray file before any member is read.
    assert "unexpected parts" in r.get_json()["error"]

    # A zip holding only allowed names but no manifest is refused on the manifest.
    only_card = io.BytesIO()
    with zipfile.ZipFile(only_card, "w") as z:
        z.writestr("model_card.html", "<p>no manifest</p>")
    only_card.seek(0)
    r = client.post("/import-bundle", data={"bundle": (only_card, "y.zip")})
    assert r.status_code == 400 and "manifest" in r.get_json()["error"]


def test_model_card_escapes_markup():
    manifest = {
        "format": BUNDLE_FORMAT,
        "model": {"name": "<img src=x onerror=alert(1)>", "features": ["a"], "targets": ["b"]},
        "data": {"filename": "x.csv"},
        "metrics": {},
        "environment": {"packages": {}},
    }
    card = render_model_card(manifest)
    assert "<img" not in card and "&lt;img" in card


def test_score_and_rerun_scripts(client, tmp_path):
    sid = uuid.uuid4().hex
    _train_in_session(sid)
    client.set_cookie("digiterra_session", sid)
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(client.get("/export-bundle").data)
    rows = tmp_path / "rows.csv"
    pd.read_csv(DIABETES).iloc[:8, :-1].to_csv(rows, index=False)

    out = tmp_path / "pred.csv"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "score_bundle.py"), str(bundle), str(rows), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    pred = pd.read_csv(out)
    assert len(pred) == 8 and any(c.startswith("Predicted_") for c in pred.columns)

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "rerun_bundle.py"), str(bundle), str(DIABETES)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all recorded metrics reproduced" in proc.stdout

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "rerun_bundle.py"), str(bundle), str(rows)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
    )
    assert proc.returncode == 2 and "hash mismatch" in proc.stderr


def _zip_bomb_bundle(good_bundle: bytes, member: str, inflated_mb: int) -> io.BytesIO:
    """The manifest of a real bundle with one member replaced by a compressible run."""
    src = zipfile.ZipFile(io.BytesIO(good_bundle))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in src.namelist():
            payload = b"\x00" * (inflated_mb << 20) if name == member else src.read(name)
            z.writestr(name, payload)
    out.seek(0)
    return out


def test_compressed_bundle_is_refused_before_the_payload_is_inflated(client, monkeypatch):
    """The signature can only be checked after model.skops is read, so the zip is bounded first."""
    import tracemalloc

    import python_scripts.reporting.bundle as bundle_mod

    sid = uuid.uuid4().hex
    _train_in_session(sid)
    client.set_cookie("digiterra_session", sid)
    good = client.get("/export-bundle").data
    monkeypatch.setenv("DIGITERRA_ALLOW_BUNDLE_IMPORT", "1")

    bomb = _zip_bomb_bundle(good, "model.skops", inflated_mb=64)
    assert len(bomb.getvalue()) < 1 << 20  # a small file that declares 64 MB
    tracemalloc.start()
    with pytest.raises(BundleError, match="is 64 MB|inflates|larger than"):
        load_bundle(bomb, force=True)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 8 << 20, f"peak {peak / 2**20:.1f} MB"

    # The same refusal reaches the route, and the manifest read is bounded too.
    bomb.seek(0)
    r = client.post("/import-bundle", data={"bundle": (bomb, "b.zip")})
    assert r.status_code == 400
    assert "model.skops" in r.get_json()["error"]

    manifest_bomb = _zip_bomb_bundle(good, "manifest.json", inflated_mb=64)
    with pytest.raises(BundleError, match="inflates|is 64 MB|larger than"):
        bundle_mod.read_manifest(manifest_bomb)


def test_bundle_with_an_unexpected_member_is_refused(client):
    sid = uuid.uuid4().hex
    _train_in_session(sid)
    client.set_cookie("digiterra_session", sid)
    good = client.get("/export-bundle").data
    src = zipfile.ZipFile(io.BytesIO(good))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for name in src.namelist():
            z.writestr(name, src.read(name))
        z.writestr("../../etc/passwd", b"root:x:0:0:")
    out.seek(0)
    with pytest.raises(BundleError, match="unexpected parts"):
        load_bundle(out, force=True)


def test_a_declared_size_that_lies_is_still_bounded_while_streaming(client, monkeypatch):
    """The directory is data from the file; the streaming read enforces the cap again."""
    import python_scripts.reporting.bundle as bundle_mod

    sid = uuid.uuid4().hex
    _train_in_session(sid)
    client.set_cookie("digiterra_session", sid)
    good = client.get("/export-bundle").data
    bomb = _zip_bomb_bundle(good, "model.skops", inflated_mb=32)
    # Preflight passes at a generous member cap; the byte cap still stops the read.
    monkeypatch.setattr(bundle_mod, "MAX_BUNDLE_MEMBER_MB", 4)
    monkeypatch.setattr(bundle_mod, "MAX_BUNDLE_TOTAL_MB", 4096)
    with pytest.raises(BundleError, match="4 MB"):
        load_bundle(bomb, force=True)


def test_the_artifact_is_not_a_pickle_and_names_what_it_builds(client):
    """Format v2 stores the objects as data; loading refuses any type outside the allowlist."""
    from python_scripts.reporting import artifact
    from python_scripts.reporting.bundle import BUNDLE_FORMAT, MODEL_NAME

    sid = uuid.uuid4().hex
    _train_in_session(sid)
    client.set_cookie("digiterra_session", sid)
    data = client.get("/export-bundle").data
    names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
    assert MODEL_NAME == "model.skops" and MODEL_NAME in names
    assert "model.joblib" not in names
    manifest = read_manifest(io.BytesIO(data))
    assert manifest["format"] == BUNDLE_FORMAT
    assert manifest["artifact"]["serializer"].startswith("skops")

    payload = zipfile.ZipFile(io.BytesIO(data)).read(MODEL_NAME)
    # Nothing in a DiGiTerra artifact falls outside the allowlist.
    assert artifact.untrusted_types(payload) == []
    # And it is not a pickle: joblib cannot read it.
    # And it is not a pickle: the bytes are a zip of named parts, not a pickle
    # stream, so nothing in them can be an instruction to run.
    assert payload[:2] == b"PK"
    assert b"\x80\x04" not in payload[:16]
    inner = set(zipfile.ZipFile(io.BytesIO(payload)).namelist())
    assert "schema.json" in inner


def test_an_artifact_naming_an_unlisted_type_is_refused(monkeypatch):
    """The allowlist is what stops a foreign file, before any object is built."""
    import skops.io as sio

    from python_scripts.reporting import artifact

    class Sneaky:
        def __init__(self):
            self.x = 1

    payload = sio.dumps({"model": Sneaky()})
    refused = artifact.untrusted_types(payload)
    assert refused and any("Sneaky" in name for name in refused)
    with pytest.raises(artifact.ArtifactError, match="does not load"):
        artifact.loads(payload)


def test_a_legacy_pickle_bundle_needs_the_flag(monkeypatch, tmp_path):
    """A v1 bundle still opens, but only where the operator says pickles are allowed."""
    import joblib

    from python_scripts.reporting import artifact
    from python_scripts.reporting.bundle import LEGACY_BUNDLE_FORMAT, sha256_bytes, sign_bytes, signing_key

    monkeypatch.setenv("DIGITERRA_BUNDLE_KEY", "legacy-key")
    import numpy as np
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(np.arange(20).reshape(10, 2), np.arange(10))
    payload = joblib.dumps({"model": model}) if hasattr(joblib, "dumps") else None
    if payload is None:
        buf = io.BytesIO()
        joblib.dump({"model": model}, buf)
        payload = buf.getvalue()
    key = signing_key(create=True)
    manifest = {
        "format": LEGACY_BUNDLE_FORMAT,
        "artifact": {
            "file": "model.joblib",
            "sha256": sha256_bytes(payload),
            "signature": sign_bytes(payload, key),
            "key_id": None,
        },
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("manifest.json", json.dumps(manifest))
        z.writestr("model.joblib", payload)
    out.seek(0)

    monkeypatch.delenv("DIGITERRA_ALLOW_LEGACY_PICKLE", raising=False)
    with pytest.raises(BundleError, match="older format"):
        load_bundle(out, force=True)
    out.seek(0)
    monkeypatch.setenv("DIGITERRA_ALLOW_LEGACY_PICKLE", "1")
    _, objects = load_bundle(out, force=True)
    assert objects["model"] is not None
    assert artifact.legacy_pickle_allowed()
