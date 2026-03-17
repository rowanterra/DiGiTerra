"""File upload route."""
import os
import uuid
import logging

import pandas as pd
from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from app_constants import (
    BYTES_PER_MB,
    CELL_COUNT_WARNING_LARGE,
    CELL_COUNT_WARNING_MODERATE,
    COLUMN_NAME_DISPLAY_LENGTH,
    ENFORCE_UPLOAD_LIMIT,
    FILENAME_DISPLAY_LENGTH,
    FILE_SIZE_WARNING_MB,
    HTTP_BAD_REQUEST,
    MAX_UPLOAD_MB,
)
from app_helpers import allowed_file
from app_state import get_session_storage

bp = Blueprint("upload", __name__)
logger = logging.getLogger(__name__)


@bp.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and return column information."""
    logger.info("Upload request received. Content-Type: %s, Files: %s", request.content_type, list(request.files.keys()))
    store = get_session_storage()
    if "file" not in request.files:
        logger.warning("No 'file' key in request.files")
        return jsonify({"error": "No file uploaded"}), HTTP_BAD_REQUEST
    file = request.files["file"]
    if file.filename == "":
        logger.warning("File filename is empty")
        return jsonify({"error": "No file selected"}), HTTP_BAD_REQUEST
    logger.info("Processing file upload: %s", file.filename)
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only CSV files are allowed."}), HTTP_BAD_REQUEST
    safe_filename_val = secure_filename(file.filename)
    if not safe_filename_val:
        return jsonify({"error": "Invalid filename."}), HTTP_BAD_REQUEST

    unique_prefix = uuid.uuid4().hex[:8]
    stored_filename = f"{unique_prefix}_{safe_filename_val}"
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], stored_filename)
    file.save(filepath)

    file_size_mb = os.path.getsize(filepath) / BYTES_PER_MB
    if ENFORCE_UPLOAD_LIMIT and MAX_UPLOAD_MB > 0 and file_size_mb > MAX_UPLOAD_MB:
        try:
            os.remove(filepath)
        except OSError:
            logger.warning("Failed to remove oversized uploaded file: %s", filepath)
        return jsonify({"error": f"File exceeds maximum allowed size of {MAX_UPLOAD_MB:.1f} MB."}), HTTP_BAD_REQUEST

    size_warning = f"Large file detected ({file_size_mb:.1f} MB). Processing may take longer." if file_size_mb > FILE_SIZE_WARNING_MB else None

    try:
        data = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        return jsonify({"error": "The uploaded file is empty."}), HTTP_BAD_REQUEST
    except pd.errors.ParserError as e:
        logger.error("CSV parsing error: %s", e)
        return jsonify({"error": f"Error parsing CSV file: {str(e)}"}), HTTP_BAD_REQUEST
    except Exception as e:
        logger.error("Error reading CSV file: %s", e, exc_info=True)
        return jsonify({"error": f"Error reading file: {str(e)}"}), HTTP_BAD_REQUEST

    if data.empty:
        return jsonify({"error": "The uploaded file contains no data."}), HTTP_BAD_REQUEST

    if data.columns.duplicated().any():
        logger.warning("Duplicate column names detected in uploaded file. Renaming duplicates.")
        new_columns = []
        seen = {}
        for col in data.columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}.{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
        data.columns = new_columns

    total_cells = len(data) * len(data.columns)
    if total_cells > CELL_COUNT_WARNING_LARGE:
        cell_warning = f"Large dataset detected ({len(data):,} rows × {len(data.columns)} columns = {total_cells:,} cells). Processing may take longer."
    elif total_cells > CELL_COUNT_WARNING_MODERATE:
        cell_warning = f"Moderate dataset size ({len(data):,} rows × {len(data.columns)} columns = {total_cells:,} cells)."
    else:
        cell_warning = None

    store["data"] = data
    columns = data.columns.tolist()
    firstcol = columns[0][:COLUMN_NAME_DISPLAY_LENGTH]
    if len(columns[0]) > COLUMN_NAME_DISPLAY_LENGTH - 1:
        firstcol += "..."
    numcols = len(columns)
    lastcol = columns[-1][:COLUMN_NAME_DISPLAY_LENGTH]
    if len(columns[-1]) > COLUMN_NAME_DISPLAY_LENGTH - 1:
        lastcol += "..."
    filename_display = file.filename[:FILENAME_DISPLAY_LENGTH]
    if len(file.filename) > FILENAME_DISPLAY_LENGTH - 1:
        filename_display += "..."

    response_data = {
        "filename": filename_display,
        "columns": columns,
        "firstcol": firstcol,
        "numcols": numcols,
        "lastcol": lastcol,
        "rows": len(data),
        "total_cells": total_cells,
    }
    warnings = []
    if size_warning:
        warnings.append(size_warning)
    if cell_warning:
        warnings.append(cell_warning)
    if warnings:
        response_data["warnings"] = warnings
        logger.info("Upload warnings for %s: %s", filename_display, "; ".join(warnings))
    return jsonify(response_data)
