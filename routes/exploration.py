"""Data exploration: auto-detect, correlation matrices, pairplot."""
from flask import Blueprint, jsonify, request

from app_helpers import ensure_user_vis_dir, with_prefix
from app_state import get_session_storage
from python_scripts.app_exploration import (
    handle_auto_detect_nan_zeros,
    handle_auto_detect_transformers,
    handle_corr,
    handle_pairplot,
)

bp = Blueprint("exploration", __name__)


@bp.route("/auto-detect-transformers", methods=["POST"])
def auto_detect_transformers():
    """Auto-detect categorical columns from selected indicators."""
    store = get_session_storage()
    data = request.json or {}
    result, status = handle_auto_detect_transformers(store, data)
    return jsonify(result), status


@bp.route("/auto-detect-nan-zeros", methods=["POST"])
def auto_detect_nan_zeros():
    """Auto-detect NaN or zeros in indicator and target columns."""
    store = get_session_storage()
    data = request.json or {}
    result, status = handle_auto_detect_nan_zeros(store, data)
    return jsonify(result), status


@bp.route("/correlationMatrices", methods=["POST"])
def corr():
    """Generate correlation matrices for numeric columns."""
    store = get_session_storage()
    data = request.json or {}
    result, status = handle_corr(store, data, ensure_user_vis_dir(), with_prefix)
    return jsonify(result), status


@bp.route("/pairplot", methods=["POST"])
def pairplot():
    """Generate pairplot visualization for two numeric columns."""
    store = get_session_storage()
    data = request.json or {}
    result, status = handle_pairplot(store, data, ensure_user_vis_dir(), with_prefix)
    return jsonify(result), status
