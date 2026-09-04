"""Run manifest, model card, and portable model bundle.

A run's result lives in one browser session and dies with it. The manifest
records what was run, on which data, with which package versions, and what
came out, so a result can be cited and rerun. The model card renders the same
record as a page. The bundle is a zip of manifest, model card, and the fitted
objects, readable by scripts/score_bundle.py without the web app.

The artifact is written by python_scripts.reporting.artifact, which stores the
fitted objects as data with their types named and refuses to build a type outside
its allowlist, so loading one does not execute what the file says. Older bundles
carry a joblib pickle instead and are read only when DIGITERRA_ALLOW_LEGACY_PICKLE
is set. On top of that, every artifact is signed with HMAC-SHA256 under a key that
belongs to the installation (DIGITERRA_BUNDLE_KEY, or a key file generated once
in the app support directory). Import verifies the signature before anything is
unpickled: a bundle from another installation, or one altered after export, is
refused. Two installations that share a key can exchange bundles. Import is also
off unless DIGITERRA_ALLOW_BUNDLE_IMPORT is set; the desktop launcher sets it,
the server images do not.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import html
import json
import logging
import os
import platform
import secrets
import zipfile
from importlib import metadata
from pathlib import Path

import numpy as np

from python_scripts.reporting.artifact import ArtifactError
from python_scripts.reporting.artifact import dumps as artifact_dumps
from python_scripts.reporting.artifact import loads as artifact_loads
from python_scripts.reporting.artifact import loads_legacy as artifact_loads_legacy
from python_scripts.reporting.artifact import serialized_size as artifact_serialized_size

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = "digiterra-bundle/2"
LEGACY_BUNDLE_FORMAT = "digiterra-bundle/1"
# A bundle is a small, known set of files. Anything else in the zip, or a part
# larger than these caps, is refused from the directory before a byte is
# inflated, because the signature can only be checked after model.joblib is read.
MAX_BUNDLE_MEMBERS = 32
MAX_BUNDLE_MEMBER_MB = float(os.environ.get("DIGITERRA_MAX_BUNDLE_MEMBER_MB", "512") or 512)
MAX_BUNDLE_TOTAL_MB = float(os.environ.get("DIGITERRA_MAX_BUNDLE_TOTAL_MB", "1024") or 1024)
MANIFEST_NAME = "manifest.json"
CARD_HTML_NAME = "model_card.html"
CARD_JSON_NAME = "model_card.json"
MODEL_NAME = "model.skops"
LEGACY_MODEL_NAME = "model.joblib"
BUNDLE_NAME = "model_bundle.zip"
TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "shap", "xgboost", "lightgbm", "catboost", "joblib")

# Keys of the training request with no reproducibility value, or that
# the manifest records under another name.
_REQUEST_SKIP = {"filename"}


def import_allowed() -> bool:
    return os.environ.get("DIGITERRA_ALLOW_BUNDLE_IMPORT", "").strip().lower() in {"1", "true", "yes", "on"}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Signing. The key never leaves the machine unless the user copies it.
KEY_FILE_NAME = "bundle_signing.key"


def _key_file() -> Path:
    from python_scripts import config

    return Path(config.APP_SUPPORT_DIR) / KEY_FILE_NAME


def signing_key(create: bool = True) -> bytes | None:
    """The installation's signing key: DIGITERRA_BUNDLE_KEY, else the key file.

    With create=True a missing key file is written with 32 random bytes (hex) and
    owner-only permissions. Returns None when there is no key and create=False.
    """
    env = os.environ.get("DIGITERRA_BUNDLE_KEY", "").strip()
    if env:
        return env.encode("utf-8")
    path = _key_file()
    try:
        if path.is_file():
            data = path.read_text(encoding="utf-8").strip()
            if data:
                return data.encode("utf-8")
        if not create:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        token = secrets.token_hex(32)
        path.write_text(token, encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return token.encode("utf-8")
    except OSError as e:
        logger.warning("Bundle signing key unavailable: %s", e)
        return None


def key_id(key: bytes) -> str:
    """A short public identifier for a key, so a mismatch can be explained."""
    return hashlib.sha256(b"digiterra-key-id:" + key).hexdigest()[:16]


def sign_bytes(data: bytes, key: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def signature_valid(data: bytes, signature: str | None, key: bytes | None) -> bool:
    if not signature or key is None:
        return False
    return hmac.compare_digest(sign_bytes(data, key), signature)


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    try:
        out["digiterra"] = metadata.version("digiterra")
    except metadata.PackageNotFoundError:
        out["digiterra"] = _read_pyproject_version()
    return out


def _read_pyproject_version() -> str | None:
    try:
        import tomllib

        root = Path(__file__).resolve().parents[2]
        with open(root / "pyproject.toml", "rb") as f:
            return tomllib.load(f).get("project", {}).get("version")
    except Exception:
        return None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return str(value)


_METRIC_KEYS = {
    "regression": [
        "trainscore",
        "valscore",
        "trainrmse",
        "valrmse",
        "trainmae",
        "valmae",
        "train_n",
        "test_n",
    ],
    "classification": [
        "accuracy",
        "precision",
        "recall",
        "f1score",
        "macro_precision",
        "macro_recall",
        "macro_f1score",
        "support",
    ],
    "cluster": [
        "best_k",
        "train_silhouette",
        "test_silhouette",
        "train_calinski_harabasz",
        "test_calinski_harabasz",
        "train_davies_bouldin",
        "test_davies_bouldin",
    ],
}

_METRIC_LABELS = {
    "trainscore": "R² (training)",
    "valscore": "R² (validation)",
    "trainrmse": "RMSE (training)",
    "valrmse": "RMSE (validation)",
    "trainmae": "MAE (training)",
    "valmae": "MAE (validation)",
    "train_n": "Training rows",
    "test_n": "Validation rows",
    "accuracy": "Accuracy",
    "precision": "Precision (weighted)",
    "recall": "Recall (weighted)",
    "f1score": "F1 (weighted)",
    "macro_precision": "Precision (macro)",
    "macro_recall": "Recall (macro)",
    "macro_f1score": "F1 (macro)",
    "support": "Validation rows",
    "best_k": "Clusters",
    "train_silhouette": "Silhouette (training)",
    "test_silhouette": "Silhouette (validation)",
    "train_calinski_harabasz": "Calinski-Harabasz (training)",
    "test_calinski_harabasz": "Calinski-Harabasz (validation)",
    "train_davies_bouldin": "Davies-Bouldin (training)",
    "test_davies_bouldin": "Davies-Bouldin (validation)",
}


def build_manifest(store: dict, request: dict, result: dict, problem_type: str, model_name: str) -> dict:
    """Assemble the run record from the session store, the request, and the result."""
    metrics = {k: result.get(k) for k in _METRIC_KEYS.get(problem_type, []) if k in result}
    cv = result.get("cross_validation_summary")
    manifest = {
        "format": BUNDLE_FORMAT,
        "created_at": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat(),
        "data": store.get("data_fingerprint")
        or {
            "note": "Uploaded before fingerprinting was in place; rows and columns recorded from memory.",
            "rows": int(len(store["data"])) if store.get("data") is not None else None,
            "columns": int(len(store["data"].columns)) if store.get("data") is not None else None,
        },
        "model": {
            "name": model_name,
            "problem_type": problem_type,
            "estimator": type(store.get("model")).__name__ if store.get("model") is not None else None,
            "params": result.get("model_params"),
            "features": list(store.get("feature_order") or []),
            "targets": [str(t) for t in (store.get("predictor_names") or [])],
            "seed": store.get("seed"),
        },
        "request": {k: _json_safe(v) for k, v in request.items() if k not in _REQUEST_SKIP},
        "preprocessing": {
            "inference_config": store.get("inference_config"),
            "impute_fill_values": store.get("impute_fill_values"),
            "zero_fill_values": store.get("zero_fill_values"),
            "split_note": result.get("split_note"),
            "feature_selection": result.get("feature_selection_info"),
            "outliers": result.get("outlier_info"),
        },
        "metrics": metrics,
        "cross_validation": cv,
        "baseline": result.get("baseline"),
        "data_checks": result.get("data_checks"),
        "environment": {
            "packages": package_versions(),
            "platform": platform.platform(),
        },
    }
    return _json_safe(manifest)


def _rows(pairs) -> str:
    out = []
    for k, v in pairs:
        if v is None or v == "" or v == []:
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, indent=None, default=str)
        out.append(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>")
    return "\n".join(out)


def render_model_card(manifest: dict) -> str:
    """A single-file HTML page from the manifest. No script, no external assets."""
    m = manifest
    model = m.get("model", {})
    data = m.get("data", {}) or {}
    metrics = m.get("metrics", {}) or {}
    checks = m.get("data_checks") or {}
    baseline = m.get("baseline") or {}
    env = (m.get("environment") or {}).get("packages", {}) or {}

    title = f"{model.get('name', 'model')} on {data.get('filename', 'uploaded data')}"
    if checks.get("forced"):
        title += " (not valid for decision use)"
    findings_html = ""
    blocking = checks.get("blocking") or []
    warnings = checks.get("warnings") or []
    if blocking:
        findings_html += "<p class='block'><strong>Not valid for decision use.</strong> Data checks blocked this run "
        findings_html += "and the override trained it anyway. "
        findings_html += html.escape(" ".join(f["message"] for f in blocking))
        if checks.get("override_reason"):
            findings_html += " Reason given: " + html.escape(str(checks["override_reason"]))
        findings_html += "</p>"
    if warnings:
        findings_html += "<p class='warn'><strong>Warnings.</strong> " + html.escape(
            " ".join(f["message"] for f in warnings)
        )
        findings_html += "</p>"
    if not blocking and not warnings and checks:
        findings_html = (
            f"<p>Data checks passed on {checks.get('n_rows')} rows and {checks.get('n_features')} features.</p>"
        )

    baseline_html = ""
    if baseline.get("metric"):
        verdict = {
            True: "The model beats the baseline.",
            False: "The model does not beat the baseline.",
            None: "",
        }[baseline.get("model_beats_baseline")]
        baseline_html = (
            f"<p>{html.escape(str(baseline.get('description')))} scores "
            f"{html.escape(str(baseline.get('baseline_value')))} {html.escape(str(baseline.get('metric')))} on the "
            f"validation split; the model scores {html.escape(str(baseline.get('model_value')))}. {verdict}</p>"
        )

    metric_rows = _rows((_METRIC_LABELS.get(k, k), v) for k, v in metrics.items())
    cv = m.get("cross_validation")
    cv_html = ""
    if isinstance(cv, list) and cv:
        cv_html = "<h2>Cross-validation</h2><table>" + _rows(
            (
                str(r.get("Metric", r.get("metric", ""))),
                f"{r.get('Mean', r.get('mean'))} ± {r.get('Std', r.get('std'))}",
            )
            for r in cv
            if isinstance(r, dict)
        )
        cv_html += "</table>"

    intended = (
        "This card describes one training run of one estimator on one file. The scores are from a single "
        "train/validation split at the seed shown; they are not a claim about other data, other sites, or "
        "other time periods. Predictions on rows outside the training ranges are extrapolations."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model card: {html.escape(title)}</title>
<style>
body {{ font-family: "Times New Roman", Times, serif; color: #1c1c1a; background: #faf8f3; margin: 0; }}
main {{ max-width: 860px; margin: 0 auto; padding: 32px 24px 64px; }}
h1 {{ font-size: 1.6em; margin: 0 0 4px; }}
h2 {{ font-size: 1.15em; margin: 28px 0 8px; border-bottom: 1px solid #cfcabc; padding-bottom: 4px; }}
p {{ line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.95em; }}
th, td {{ text-align: left; vertical-align: top; padding: 5px 8px; border-bottom: 1px solid #e4dfd2; }}
th {{ width: 34%; font-weight: normal; color: #4b4a45; }}
td {{ font-family: Menlo, Consolas, monospace; font-size: 0.9em; word-break: break-word; }}
.meta {{ color: #4b4a45; margin: 0 0 20px; }}
.warn {{ border-left: 3px solid #b07a1a; background: #fbf5e8; padding: 8px 12px; }}
.block {{ border-left: 3px solid #8a2a2a; background: #f8ecec; padding: 8px 12px; }}
</style>
</head>
<body>
<main>
<h1>{html.escape(title)}</h1>
<p class="meta">Created {html.escape(str(m.get("created_at")))} UTC. DiGiTerra {html.escape(str(env.get("digiterra")))},
scikit-learn {html.escape(str(env.get("scikit-learn")))}, Python {html.escape(str(env.get("python")))}.</p>

<h2>Data</h2>
<table>
{
        _rows(
            [
                ("File", data.get("filename")),
                ("SHA-256", data.get("sha256")),
                ("Bytes", data.get("bytes")),
                ("Rows", data.get("rows")),
                ("Columns", data.get("columns")),
                ("Uploaded", data.get("uploaded_at")),
                ("Note", data.get("note")),
            ]
        )
    }
</table>

<h2>Model</h2>
<table>
{
        _rows(
            [
                ("Name", model.get("name")),
                ("Problem type", model.get("problem_type")),
                ("Estimator", model.get("estimator")),
                ("Seed", model.get("seed")),
                ("Features", ", ".join(model.get("features") or [])),
                ("Targets", ", ".join(model.get("targets") or [])),
                ("Parameters", model.get("params")),
            ]
        )
    }
</table>

<h2>Preprocessing and split</h2>
<table>
{_rows([(k, v) for k, v in (m.get("preprocessing") or {}).items()])}
</table>

<h2>Data checks</h2>
{findings_html or "<p>No data checks were recorded for this run.</p>"}

<h2>Performance</h2>
<table>
{metric_rows}
</table>
{baseline_html}
{cv_html}

<h2>Intended use and limits</h2>
<p>{intended}</p>

<h2>Environment</h2>
<table>
{_rows(env.items())}
<tr><th>Platform</th><td>{html.escape(str((m.get("environment") or {}).get("platform")))}</td></tr>
</table>
</main>
</body>
</html>
"""


def write_run_record(vis_dir: Path, manifest: dict) -> dict[str, Path]:
    """Write manifest.json, model_card.json, and model_card.html into the session directory."""
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": vis_dir / MANIFEST_NAME,
        "card_json": vis_dir / CARD_JSON_NAME,
        "card_html": vis_dir / CARD_HTML_NAME,
    }
    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(paths["card_json"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(paths["card_html"], "w", encoding="utf-8") as f:
        f.write(render_model_card(manifest))
    return paths


STORE_KEYS = (
    "model",
    "X_scaler",
    "y_scaler",
    "feature_order",
    "predictor_names",
    "model_type",
    "inference_config",
    "impute_fill_values",
    "zero_fill_values",
    "seed",
    "y_train_mean",
    "y_train_std",
    "training_visualization",
)


def fitted_objects(store: dict) -> dict:
    return {k: store.get(k) for k in STORE_KEYS}


def serialized_size(store: dict) -> int:
    """Bytes the session's fitted objects take when written, without holding them."""
    return artifact_serialized_size(fitted_objects(store))


def write_bundle(store: dict, vis_dir: Path, extra_files: list[str] | None = None) -> Path:
    """Zip manifest, model card, fitted objects, and any listed session files."""
    vis_dir = Path(vis_dir)
    manifest_path = vis_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError("This session has no manifest. Train a model, then export the bundle.")
    if store.get("model") is None:
        raise ValueError("No trained model in this session.")

    try:
        model_bytes = artifact_dumps(fitted_objects(store))
    except ArtifactError as e:
        raise ValueError(str(e)) from e

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = signing_key(create=True)
    manifest["artifact"] = {
        "file": MODEL_NAME,
        "sha256": sha256_bytes(model_bytes),
        "bytes": len(model_bytes),
        "serializer": f"skops {metadata.version('skops')}",
        "signature": sign_bytes(model_bytes, key) if key else None,
        "signature_algorithm": "HMAC-SHA256" if key else None,
        "key_id": key_id(key) if key else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (vis_dir / CARD_JSON_NAME).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (vis_dir / CARD_HTML_NAME).write_text(render_model_card(manifest), encoding="utf-8")

    out = vis_dir / BUNDLE_NAME
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(manifest_path, MANIFEST_NAME)
        z.write(vis_dir / CARD_JSON_NAME, CARD_JSON_NAME)
        z.write(vis_dir / CARD_HTML_NAME, CARD_HTML_NAME)
        z.writestr(MODEL_NAME, model_bytes)
        for name in extra_files or []:
            p = vis_dir / name
            if p.is_file():
                z.write(p, name)
    return out


class BundleError(ValueError):
    pass


# Files a bundle may hold. write_bundle adds the run's workbook and plot pages;
# anything outside this set means the zip was not written by DiGiTerra.
ALLOWED_BUNDLE_NAMES = {
    MANIFEST_NAME,
    CARD_JSON_NAME,
    CARD_HTML_NAME,
    MODEL_NAME,
    LEGACY_MODEL_NAME,
    "model_results.xlsx",
    "visualizations.pdf",
}
# A manifest is a few kilobytes of JSON; nothing legitimate approaches this.
MAX_MANIFEST_BYTES = 4 << 20


def preflight_bundle(bundle_path):
    """Refuse an oversized or unexpected bundle zip before any member is inflated."""
    from core.helpers import ZipTooLarge, preflight_zip

    if hasattr(bundle_path, "seek"):
        bundle_path.seek(0)
    try:
        preflight_zip(
            bundle_path,
            max_total_bytes=int(MAX_BUNDLE_TOTAL_MB * 1024 * 1024),
            max_member_bytes=int(MAX_BUNDLE_MEMBER_MB * 1024 * 1024),
            max_members=MAX_BUNDLE_MEMBERS,
            allowed_names=ALLOWED_BUNDLE_NAMES,
            label="bundle",
        )
    except ZipTooLarge as e:
        raise BundleError(str(e)) from e
    except zipfile.BadZipFile as e:
        raise BundleError("The file is not a readable zip.") from e


def read_manifest(bundle_path: str | Path) -> dict:
    from core.helpers import ZipTooLarge, read_zip_member_bounded

    preflight_bundle(bundle_path)
    if hasattr(bundle_path, "seek"):
        bundle_path.seek(0)
    with zipfile.ZipFile(bundle_path) as z:
        try:
            raw = read_zip_member_bounded(z, MANIFEST_NAME, MAX_MANIFEST_BYTES, label="bundle")
        except KeyError as e:
            raise BundleError("The zip has no manifest.json file. The zip is not a DiGiTerra bundle.") from e
        except ZipTooLarge as e:
            raise BundleError(str(e)) from e
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise BundleError("manifest.json is not readable JSON; it is not a DiGiTerra bundle.") from e
    fmt = manifest.get("format") if isinstance(manifest, dict) else None
    if fmt not in (BUNDLE_FORMAT, LEGACY_BUNDLE_FORMAT):
        raise BundleError(f"Unsupported bundle format {fmt!r}; expected {BUNDLE_FORMAT}.")
    return manifest


def load_bundle(bundle_path: str | Path, *, force: bool = False, trust_unsigned: bool = False) -> tuple[dict, dict]:
    """Return (manifest, fitted objects) after checking the signature.

    The artifact is unpickled only when its HMAC-SHA256 signature verifies under
    this installation's key. `force` skips the DIGITERRA_ALLOW_BUNDLE_IMPORT
    deployment switch (command-line tools); `trust_unsigned` skips the signature
    check and exists for the command line only, where the person running it has
    chosen to trust the file.
    """
    if not (force or import_allowed()):
        raise BundleError(
            "Bundle import is disabled on this deployment. Set DIGITERRA_ALLOW_BUNDLE_IMPORT=1 to enable it "
            "where the bundles come from a trusted source."
        )
    from core.helpers import ZipTooLarge, read_zip_member_bounded

    # read_manifest preflights the directory; the streaming read below enforces the
    # same cap on the bytes that actually arrive, since a directory can lie.
    manifest = read_manifest(bundle_path)
    if hasattr(bundle_path, "seek"):
        bundle_path.seek(0)
    with zipfile.ZipFile(bundle_path) as z:
        names = set(z.namelist())
        member = MODEL_NAME if MODEL_NAME in names else LEGACY_MODEL_NAME
        if member not in names:
            raise BundleError(f"The bundle has no {MODEL_NAME}.")
        try:
            model_bytes = read_zip_member_bounded(z, member, int(MAX_BUNDLE_MEMBER_MB * 1024 * 1024), label="bundle")
        except ZipTooLarge as e:
            raise BundleError(str(e)) from e
    artifact_meta = manifest.get("artifact") or {}
    expected = artifact_meta.get("sha256")
    actual = sha256_bytes(model_bytes)
    if expected and expected != actual:
        raise BundleError("model.joblib does not match the hash in the manifest. The bundle was altered.")
    if not trust_unsigned:
        key = signing_key(create=False)
        if not signature_valid(model_bytes, artifact_meta.get("signature"), key):
            theirs = artifact_meta.get("key_id") or "none"
            mine = key_id(key) if key else "none"
            raise BundleError(
                "The bundle's signature does not verify under this installation's key "
                f"(bundle key {theirs}, this key {mine}). Loading it would run code from the file. "
                "Only bundles exported by this installation, or by one that shares its DIGITERRA_BUNDLE_KEY, "
                "can be loaded."
            )
    try:
        objects = artifact_loads(model_bytes) if member == MODEL_NAME else artifact_loads_legacy(model_bytes)
    except ArtifactError as e:
        raise BundleError(str(e)) from e
    return manifest, objects


def package_drift(manifest: dict) -> list[str]:
    """Names of tracked packages whose installed version differs from the manifest."""
    recorded = (manifest.get("environment") or {}).get("packages") or {}
    now = package_versions()
    return [k for k, v in recorded.items() if v is not None and now.get(k) != v]
