"""User visualizations, download, and additional info."""

import uuid
import zipfile

import pandas as pd
from flask import Blueprint, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from core.constants import HTTP_BAD_REQUEST, HTTP_FORBIDDEN, HTTP_INTERNAL_SERVER_ERROR, HTTP_NOT_FOUND
from core.helpers import ensure_user_vis_dir
from core.state import get_session_storage

bp = Blueprint("assets", __name__)
logger = __import__("logging").getLogger(__name__)


@bp.route("/user-visualizations/<path:filename>")
def user_visualizations(filename):
    """Serve user-generated visualization files."""
    vis_dir = ensure_user_vis_dir()
    response = send_from_directory(str(vis_dir), filename)
    if response.status_code == 200:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


@bp.route("/download/<path:filename>")
def download_visualization(filename):
    """Serve a file from user visualizations for download. Path traversal is blocked."""
    vis_dir = ensure_user_vis_dir()
    file_path = (vis_dir / filename).resolve()
    base_resolved = vis_dir.resolve()
    try:
        file_path.relative_to(base_resolved)
    except ValueError:
        return jsonify({"error": "File not found."}), HTTP_NOT_FOUND
    if not file_path.is_file():
        return jsonify({"error": "File not found."}), HTTP_NOT_FOUND
    requested_name = request.args.get("download_name", filename)
    download_name = secure_filename(requested_name) or filename
    return send_file(file_path, as_attachment=True, download_name=download_name)


@bp.route("/download/predictions.xlsx")
def download_predictions_xlsx():
    """Write this session's predictions.csv as a workbook and serve the workbook.

    The CSV is converted with openpyxl on each request, so the workbook is
    current with the last inference run. 404 when no predictions.csv exists.
    """
    vis_dir = ensure_user_vis_dir()
    csv_path = vis_dir / "predictions.csv"
    if not csv_path.is_file():
        return jsonify({"error": "No predictions have been written for this session."}), HTTP_NOT_FOUND
    xlsx_path = vis_dir / "predictions.xlsx"
    try:
        frame = pd.read_csv(csv_path)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="Predictions", index=False)
    except Exception as e:
        logger.error("Error writing predictions workbook: %s", str(e))
        return jsonify({"error": str(e)}), HTTP_INTERNAL_SERVER_ERROR
    return send_file(xlsx_path, as_attachment=True, download_name="predictions.xlsx")


@bp.route("/downloadAdditionalInfo", methods=["POST"])
def download_additional_info():
    """Generate and download additional information Excel file."""
    try:
        data = request.json or {}
        table_data = data.get("table_data", [])
        sheet_name = data.get("sheet_name", "Additional Information")
        if not table_data:
            return jsonify({"error": "No table data provided"}), HTTP_BAD_REQUEST
        if len(table_data[0]) == 2:
            df = pd.DataFrame(table_data, columns=["Property", "Value"])
        elif len(table_data[0]) == 3:
            df = pd.DataFrame(table_data, columns=["Metric", "Mean", "Std"])
        else:
            return jsonify({"error": "Invalid table data format"}), HTTP_BAD_REQUEST
        filename = f"additional_info_{uuid.uuid4().hex[:8]}.xlsx"
        file_path = ensure_user_vis_dir() / filename
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        return jsonify({"filename": filename})
    except Exception as e:
        logger.error("Error generating additional info Excel: %s", str(e))
        return jsonify({"error": str(e)}), HTTP_INTERNAL_SERVER_ERROR


@bp.route("/export-bundle")
def export_bundle():
    """Zip the run manifest, model card, and fitted objects for this session."""
    from python_scripts.reporting.bundle import write_bundle

    store = get_session_storage()
    vis_dir = ensure_user_vis_dir()
    try:
        path = write_bundle(store, vis_dir, extra_files=["model_results.xlsx", "visualizations.pdf"])
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), HTTP_NOT_FOUND
    except Exception as e:
        logger.error("Bundle export failed: %s", e, exc_info=True)
        return jsonify({"error": "The bundle could not be written."}), HTTP_INTERNAL_SERVER_ERROR
    model_name = secure_filename(str(store.get("model_name") or "model")) or "model"
    return send_file(path, as_attachment=True, download_name=f"digiterra_{model_name}_bundle.zip")


@bp.route("/import-bundle", methods=["POST"])
def import_bundle():
    """Load a bundle's fitted objects into this session so /predict can use them.

    Off unless DIGITERRA_ALLOW_BUNDLE_IMPORT is set. The current format is skops,
    which builds only the types the allowlist names, and the signature is checked
    before the artifact is read. A bundle in the older format holds a pickle, which
    runs code when it is read, and opens only where DIGITERRA_ALLOW_LEGACY_PICKLE
    says so. The desktop launcher sets the import flag.
    """
    from python_scripts.reporting.bundle import STORE_KEYS, BundleError, import_allowed, load_bundle, package_drift

    if not import_allowed():
        return jsonify({"error": "Bundle import is disabled on this deployment."}), HTTP_FORBIDDEN
    if "bundle" not in request.files or request.files["bundle"].filename == "":
        return jsonify({"error": "No bundle uploaded."}), HTTP_BAD_REQUEST
    upload = request.files["bundle"]
    if not upload.filename.lower().endswith(".zip"):
        return jsonify({"error": "A bundle is a .zip file."}), HTTP_BAD_REQUEST
    vis_dir = ensure_user_vis_dir()
    tmp = vis_dir / f"import_{uuid.uuid4().hex[:8]}.zip"
    upload.save(tmp)
    try:
        manifest, objects = load_bundle(tmp)
    except (BundleError, zipfile.BadZipFile) as e:
        return jsonify({"error": str(e)}), HTTP_BAD_REQUEST
    except Exception as e:
        logger.error("Bundle import failed: %s", e, exc_info=True)
        return jsonify({"error": "The bundle could not be read."}), HTTP_BAD_REQUEST
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    store = get_session_storage()
    for key in STORE_KEYS:
        store[key] = objects.get(key)
    store["model_name"] = (manifest.get("model") or {}).get("name")
    store["imported_manifest"] = manifest
    drift = package_drift(manifest)
    return jsonify(
        {
            "model": manifest.get("model"),
            "data": manifest.get("data"),
            "created_at": manifest.get("created_at"),
            "package_drift": drift,
            "note": (
                "Installed package versions differ from the bundle for: " + ", ".join(drift) + "."
                if drift
                else "Installed package versions match the bundle."
            ),
        }
    )
