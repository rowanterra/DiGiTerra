"""User visualizations, download, and additional info."""
import uuid

import pandas as pd
from flask import Blueprint, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from core.constants import HTTP_BAD_REQUEST, HTTP_INTERNAL_SERVER_ERROR, HTTP_NOT_FOUND
from core.helpers import ensure_user_vis_dir

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
