### Section 1: Importing packages and models
#
# Handoff note: This file holds all Flask routes, upload/prediction logic, and model
# orchestration. Sections are marked with "### Section N". Use them to jump around.
# Global memStorage = in-memory state (models, data, scalers). Fine for desktop;
# replace with session/DB if you go multi-user web. See HANDOFF.md for more.

from flask import Flask, request, render_template, jsonify, send_from_directory, send_file, Response, stream_with_context
import uuid
from werkzeug.utils import secure_filename
import os
import ast
import json
import threading
import platform
import logging
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

import random
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Configure logging
logger = logging.getLogger(__name__)

BASE_DIR = Path(os.environ.get("DIGITERRA_BASE_DIR", Path(__file__).resolve().parent))

# Platform-specific paths (fix for cross-platform compatibility)
if platform.system() == "Windows":
    APP_SUPPORT_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "DiGiTerra"
elif platform.system() == "Linux":
    xdg_data_home = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    APP_SUPPORT_DIR = Path(xdg_data_home) / "DiGiTerra"
else:  # macOS
    APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "DiGiTerra"
USER_VIS_DIR = Path(
    os.environ.get(
        "DIGITERRA_OUTPUT_DIR",
        APP_SUPPORT_DIR / "user_visualizations",
    )
)
os.environ.setdefault("DIGITERRA_OUTPUT_DIR", str(USER_VIS_DIR))

# Ensure VIS_DIR in config matches USER_VIS_DIR after environment is set
from python_scripts.config import VIS_DIR, update_vis_dir
update_vis_dir(USER_VIS_DIR)

from python_scripts.helpers import (
    write_to_excel,
    preprocess_data,
    prediction,
    write_to_excelClassifier,
    write_to_excelCluster,
    run_cross_validation,
    unpack_classification_result,
)
from python_scripts.models.regression_models.train_linear import train_linear
from python_scripts.models.regression_models.train_lasso import train_lasso
from python_scripts.models.regression_models.train_elasticnet import train_elasticnet
from python_scripts.models.regression_models.train_gb import train_gb
from python_scripts.models.regression_models.train_knn import train_knn
from python_scripts.models.regression_models.train_logistic import train_logistic
from python_scripts.models.regression_models.train_mlp import train_mlp
from python_scripts.models.regression_models.train_perceptron import train_perceptron
from python_scripts.models.regression_models.train_rf import train_rf
from python_scripts.models.regression_models.train_ridge import train_ridge
from python_scripts.models.regression_models.train_svr import train_svr
from python_scripts.models.regression_models.train_bayesian_ridge import train_bayesian_ridge
from python_scripts.models.regression_models.train_ard_regression import train_ard_regression
from python_scripts.models.regression_models.train_extra_trees import train_extra_trees
# Additional regression models
from python_scripts.models.regression_models.train_adaboost_regressor import train_adaboost_regressor
from python_scripts.models.regression_models.train_bagging_regressor import train_bagging_regressor
from python_scripts.models.regression_models.train_decision_tree_regressor import train_decision_tree_regressor
from python_scripts.models.regression_models.train_elasticnet_cv import train_elasticnet_cv
from python_scripts.models.regression_models.train_hist_gradient_boosting_regressor import train_hist_gradient_boosting_regressor
from python_scripts.models.regression_models.train_huber_regressor import train_huber_regressor
from python_scripts.models.regression_models.train_lars import train_lars
from python_scripts.models.regression_models.train_lars_cv import train_lars_cv
from python_scripts.models.regression_models.train_lasso_cv import train_lasso_cv
from python_scripts.models.regression_models.train_lassolars import train_lassolars
from python_scripts.models.regression_models.train_lassolars_cv import train_lassolars_cv
from python_scripts.models.regression_models.train_linearsvr import train_linearsvr
from python_scripts.models.regression_models.train_nusvr import train_nusvr
from python_scripts.models.regression_models.train_orthogonal_matching_pursuit import train_orthogonal_matching_pursuit
from python_scripts.models.regression_models.train_passive_aggressive_regressor import train_passive_aggressive_regressor
from python_scripts.models.regression_models.train_quantile_regressor import train_quantile_regressor
from python_scripts.models.regression_models.train_radius_neighbors_regressor import train_radius_neighbors_regressor
from python_scripts.models.regression_models.train_ransac_regressor import train_ransac_regressor
from python_scripts.models.regression_models.train_ridge_cv import train_ridge_cv
from python_scripts.models.regression_models.train_sgd_regressor import train_sgd_regressor
from python_scripts.models.regression_models.train_theilsen_regressor import train_theilsen_regressor
from python_scripts.models.classify_models.train_logistic_classifier import train_logistic_classifier
from python_scripts.models.classify_models.train_mlp_classifier import train_mlp_classifier
from python_scripts.models.classify_models.train_rf_classifier import train_rf_classifier
from python_scripts.models.classify_models.train_svc import train_svc
from python_scripts.models.classify_models.train_extra_trees_classifier import train_extra_trees_classifier
from python_scripts.models.classify_models.train_gaussian_nb import train_gaussian_nb
from python_scripts.models.classify_models.train_sgd_classifier import train_sgd_classifier
# Additional classification models
from python_scripts.models.classify_models.train_adaboost_classifier import train_adaboost_classifier
from python_scripts.models.classify_models.train_bagging_classifier import train_bagging_classifier
from python_scripts.models.classify_models.train_bernoulli_nb import train_bernoulli_nb
from python_scripts.models.classify_models.train_categorical_nb import train_categorical_nb
from python_scripts.models.classify_models.train_complement_nb import train_complement_nb
from python_scripts.models.classify_models.train_decision_tree_classifier import train_decision_tree_classifier
from python_scripts.models.classify_models.train_gradient_boosting_classifier import train_gradient_boosting_classifier
from python_scripts.models.classify_models.train_hist_gradient_boosting_classifier import train_hist_gradient_boosting_classifier
from python_scripts.models.classify_models.train_kneighbors_classifier import train_kneighbors_classifier
from python_scripts.models.classify_models.train_linear_discriminant_analysis import train_linear_discriminant_analysis
from python_scripts.models.classify_models.train_linearsvc import train_linearsvc
from python_scripts.models.classify_models.train_multinomial_nb import train_multinomial_nb
from python_scripts.models.classify_models.train_nusvc import train_nusvc
from python_scripts.models.classify_models.train_passive_aggressive_classifier import train_passive_aggressive_classifier
from python_scripts.models.classify_models.train_quadratic_discriminant_analysis import train_quadratic_discriminant_analysis
from python_scripts.models.classify_models.train_ridge_classifier import train_ridge_classifier
from python_scripts.models.cluster_models.train_agglomerative import train_agglomerative
from python_scripts.models.cluster_models.train_gmm import train_gmm
from python_scripts.models.cluster_models.train_kmeans import train_kmeans
from python_scripts.models.cluster_models.train_dbscan import train_dbscan
from python_scripts.models.cluster_models.train_birch import train_birch
from python_scripts.models.cluster_models.train_spectral import train_spectral
# Additional clustering models
from python_scripts.models.cluster_models.train_affinity_propagation import train_affinity_propagation
from python_scripts.models.cluster_models.train_bisecting_kmeans import train_bisecting_kmeans
from python_scripts.models.cluster_models.train_hdbscan import train_hdbscan
from python_scripts.models.cluster_models.train_meanshift import train_meanshift
from python_scripts.models.cluster_models.train_minibatch_kmeans import train_minibatch_kmeans
from python_scripts.models.cluster_models.train_optics import train_optics

UPLOAD_DIR = APP_SUPPORT_DIR / "uploads"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
## create upload folder for the csv files the user uploads
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
USER_VIS_DIR.mkdir(parents=True, exist_ok=True)
memStorage = {}

# File upload validation
ALLOWED_EXTENSIONS = {'csv'}

# File size and cell count thresholds for warnings (not restrictions)
FILE_SIZE_WARNING_MB = 50  # Warn if file is larger than 50MB
CELL_COUNT_WARNING_LARGE = 1_000_000  # Warn if more than 1 million cells
CELL_COUNT_WARNING_MODERATE = 500_000  # Moderate warning if more than 500k cells

# Default values for model training
DEFAULT_CV_FOLDS = 5
DEFAULT_SEARCH_ITERATIONS = 50

# Progress tracking constants
PROGRESS_STREAM_TIMEOUT_SECONDS = 3600  # 1 hour timeout for SSE streams
PROGRESS_UPDATE_INTERVAL_SECONDS = 0.5  # Update every 500ms
PROGRESS_FINAL_UPDATE_DELAY_SECONDS = 0.5  # Delay before final update
PROGRESS_COMPLETE_THRESHOLD = 100  # Progress percentage for completion

# HTTP status codes
HTTP_OK = 200
HTTP_ACCEPTED = 202
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_SERVER_ERROR = 500

# Image/visualization constants
IMAGE_DPI = 150  # DPI for saved images
CORRELATION_HEATMAP_LINEWIDTH = 0.5
CORRELATION_CBAR_SHRINK = 0.8
COLUMN_NAME_DISPLAY_LENGTH = 11  # Max characters to display for column names
FILENAME_DISPLAY_LENGTH = 31  # Max characters to display for filenames

# File size conversion
BYTES_PER_MB = 1024 * 1024

# Seed generation
RANDOM_SEED_MAX = 1000

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Import progress tracker
from python_scripts.preprocessing.progress_tracker import get_tracker, remove_tracker, set_result


def create_app():
    """Create and return the Flask application instance.
    
    Returns:
        Flask: The configured Flask application.
    """
    return app


def run_app(host=None, port=None, debug=False):
    """Run the Flask application.
    
    Args:
        host (str, optional): Host to bind to. Defaults to '127.0.0.1' or DIGITERRA_HOST env var.
        port (int, optional): Port to bind to. Defaults to 5000 or DIGITERRA_PORT env var.
        debug (bool, optional): Enable debug mode. Defaults to False.
    """
    host = host or os.environ.get('DIGITERRA_HOST', '127.0.0.1')
    port = port or int(os.environ.get('DIGITERRA_PORT', '5000'))
    app.run(host=host, port=port, debug=debug)

# -----------------------------------------------------------------------------
# Utility: guard against accidentally swapped X/y scalers in regression
# -----------------------------------------------------------------------------
def _maybe_fix_swapped_scalers(y_scaler, X_scaler, y_train_array):
    """Heuristic guard against swapped scalers.

    If the scaler stored as y_scaler looks like it was fit on X (or vice versa),
    inverse-transforming predictions will explode (e.g., target values predicted in the
    millions). This check swaps the two when that signature is detected.

    Parameters
    ----------
    y_scaler, X_scaler : sklearn-like scalers or None
    y_train_array : np.ndarray
        The training target values used to fit y_scaler.

    Returns
    -------
    (y_scaler, X_scaler)
    """
    if y_scaler is None or X_scaler is None:
        return y_scaler, X_scaler

    # Only scalers like StandardScaler/RobustScaler/MinMaxScaler expose mean_/scale_
    y_mean_attr = getattr(y_scaler, 'mean_', None)
    x_mean_attr = getattr(X_scaler, 'mean_', None)
    if y_mean_attr is None or x_mean_attr is None:
        return y_scaler, X_scaler

    try:
        y_train_mean = float(np.mean(y_train_array))
        y_train_std = float(np.std(y_train_array))
        ys_mean = float(np.mean(y_mean_attr))
        # If y_scaler's mean is wildly far from the target mean, it's likely swapped.
        if np.isfinite(ys_mean) and abs(ys_mean - y_train_mean) > 10 * max(1.0, y_train_std):
            logger.warning('y_scaler looks incorrect (likely swapped with X_scaler). Swapping them.')
            return X_scaler, y_scaler
    except Exception:
        # If anything goes wrong, do nothing rather than break the app.
        return y_scaler, X_scaler

    return y_scaler, X_scaler


# -----------------------------------------------------------------------------
# Utility: make model params JSON-serializable (get_params() can include estimators)
# -----------------------------------------------------------------------------
def _json_safe_params(obj):
    """Recursively sanitize params from model.get_params() for JSON.
    Replaces estimators, ndarrays, and other non-serializable values with placeholders.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe_params(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_params(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    return f"<{type(obj).__module__}.{type(obj).__name__}>"


### Section 2: Render the main html page
@app.route('/')
def index():
    """Render the main HTML page.
    
    Returns:
        HTML template: The main index.html page.
    """
    return render_template('index.html')


@app.route('/progress/<session_id>')
def progress_stream(session_id):
    """Server-Sent Events endpoint for progress updates.
    
    Args:
        session_id (str): Unique session identifier for progress tracking.
    
    Returns:
        Response: Server-Sent Events stream with progress updates.
    """
    def generate():
        import time
        tracker = get_tracker(session_id)
        timeout = PROGRESS_STREAM_TIMEOUT_SECONDS
        start_time = time.time()
        
        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                yield f"data: {json.dumps({'error': 'Progress stream timeout'})}\n\n"
                break
                
            try:
                progress = tracker.get_progress()
                
                # Always send updates (progress object changes reference)
                # Use json.dumps instead of jsonify for SSE (no Flask context needed)
                progress_json = json.dumps(progress)
                yield f"data: {progress_json}\n\n"
                
                # Stop if complete
                if progress['overall_progress'] >= PROGRESS_COMPLETE_THRESHOLD:
                    time.sleep(PROGRESS_FINAL_UPDATE_DELAY_SECONDS)  # Send final update
                    # Include result if available (with retry in case of race condition)
                    from python_scripts.preprocessing.progress_tracker import get_result
                    result = get_result(session_id)
                    # Retry up to 3 times if result not immediately available (handles race condition)
                    max_retries = 3
                    retry_count = 0
                    while not result and retry_count < max_retries:
                        time.sleep(0.5)
                        result = get_result(session_id)
                        retry_count += 1
                    if result:
                        result_json = json.dumps({'type': 'result', 'data': result})
                        yield f"data: {result_json}\n\n"
                    else:
                        # Send error if result never became available
                        error_json = json.dumps({'error': 'Result not available after training completed'})
                        yield f"data: {error_json}\n\n"
                    remove_tracker(session_id)
                    break
                    
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
                    
            time.sleep(PROGRESS_UPDATE_INTERVAL_SECONDS)  # Update every 500ms
            
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/user-visualizations/<path:filename>')
def user_visualizations(filename):
    """Serve user-generated visualization files.
    
    Args:
        filename (str): Name of the visualization file to serve.
    
    Returns:
        Response: File download response or 404 if not found.
    """
    return send_from_directory(USER_VIS_DIR, filename)


@app.route('/download/<path:filename>')
def download_visualization(filename):
    """Serve a file from USER_VIS_DIR for download. Path traversal is blocked."""
    file_path = (USER_VIS_DIR / filename).resolve()
    base_resolved = USER_VIS_DIR.resolve()
    try:
        file_path.relative_to(base_resolved)
    except ValueError:
        return jsonify({'error': 'File not found.'}), HTTP_NOT_FOUND
    if not file_path.is_file():
        return jsonify({'error': 'File not found.'}), HTTP_NOT_FOUND
    requested_name = request.args.get("download_name", filename)
    download_name = secure_filename(requested_name) or filename
    return send_file(file_path, as_attachment=True, download_name=download_name)

@app.route('/downloadAdditionalInfo', methods=['POST'])
def download_additional_info():
    """Generate and download additional information Excel file.
    
    Returns:
        JSON: Response with filename on success, error message on failure.
    """
    try:
        data = request.json
        table_data = data.get('table_data', [])
        sheet_name = data.get('sheet_name', 'Additional Information')
        
        if not table_data:
            return jsonify({'error': 'No table data provided'}), HTTP_BAD_REQUEST
        
        # Create DataFrame from table data
        if len(table_data) > 0 and len(table_data[0]) == 2:
            # Two-column format (key-value pairs)
            df = pd.DataFrame(table_data, columns=['Property', 'Value'])
        elif len(table_data) > 0 and len(table_data[0]) == 3:
            # Three-column format (e.g., Cross Validation: Metric, Mean, Std)
            df = pd.DataFrame(table_data, columns=['Metric', 'Mean', 'Std'])
        else:
            return jsonify({'error': 'Invalid table data format'}), HTTP_BAD_REQUEST
        
        # Generate filename
        import uuid
        filename = f"additional_info_{uuid.uuid4().hex[:8]}.xlsx"
        file_path = USER_VIS_DIR / filename
        
        # Write to Excel
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return jsonify({'filename': filename})
    except Exception as e:
        logger.error(f'Error generating additional info Excel: {str(e)}')
        return jsonify({'error': str(e)}), HTTP_INTERNAL_SERVER_ERROR

### Section 3: upload route  
    ## creates the route to handle file upload and gets the column names to send to front end
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and return column information.
    
    Validates file, reads CSV, and returns column names with optional warnings
    for large files or datasets.
    
    Returns:
        JSON: Response with filename, columns, and metadata on success.
            Includes warnings if file size or cell count exceeds thresholds.
            Returns error message on failure.
    """
    logger.info(f"Upload request received. Content-Type: {request.content_type}, Files in request: {list(request.files.keys())}")
    
    if 'file' not in request.files:
        logger.warning("No 'file' key in request.files")
        return jsonify({'error': 'No file uploaded'}), HTTP_BAD_REQUEST
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("File filename is empty")
        return jsonify({'error': 'No file selected'}), HTTP_BAD_REQUEST
    
    logger.info(f"Processing file upload: {file.filename}")
    
    # Security: Validate file extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), HTTP_BAD_REQUEST
    
    # Security: Sanitize filename to prevent path traversal
    safe_filename = secure_filename(file.filename)
    if not safe_filename:
        return jsonify({'error': 'Invalid filename.'}), HTTP_BAD_REQUEST
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)

        # Check file size for warning (not restriction)
        file_size_mb = os.path.getsize(filepath) / BYTES_PER_MB
        size_warning = None
        if file_size_mb > FILE_SIZE_WARNING_MB:
            size_warning = f"Large file detected ({file_size_mb:.1f} MB). Processing may take longer."

        # Read the CSV to extract column names with error handling
        # Note: Modern pandas automatically handles duplicate column names by appending suffixes
        try:
            data = pd.read_csv(filepath)
        except pd.errors.EmptyDataError:
            return jsonify({'error': 'The uploaded file is empty.'}), HTTP_BAD_REQUEST
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error: {e}")
            return jsonify({'error': f'Error parsing CSV file: {str(e)}'}), HTTP_BAD_REQUEST
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}", exc_info=True)
            return jsonify({'error': f'Error reading file: {str(e)}'}), HTTP_BAD_REQUEST
        
        if data.empty:
            return jsonify({'error': 'The uploaded file contains no data.'}), HTTP_BAD_REQUEST
        
        # Check for and handle duplicate column names
        if data.columns.duplicated().any():
            logger.warning(f"Duplicate column names detected in uploaded file. Renaming duplicates.")
            # Rename duplicate columns by appending .1, .2, etc.
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
        
        # Check cell count for warning (rows * columns)
        total_cells = len(data) * len(data.columns)
        cell_warning = None
        if total_cells > CELL_COUNT_WARNING_LARGE:
            cell_warning = f"Large dataset detected ({len(data):,} rows × {len(data.columns)} columns = {total_cells:,} cells). Processing may take longer."
        elif total_cells > CELL_COUNT_WARNING_MODERATE:
            cell_warning = f"Moderate dataset size ({len(data):,} rows × {len(data.columns)} columns = {total_cells:,} cells)."
        
        memStorage['data'] = data   # store in memStorage instead of calling it 5 times
        
        # getting the names of first and last columns to send to front end
        columns = data.columns.tolist()
        firstcol = columns[0][:COLUMN_NAME_DISPLAY_LENGTH]
        if len(columns[0]) > COLUMN_NAME_DISPLAY_LENGTH - 1:
            firstcol += "..."
        
        numcols = len(columns)
        lastcol = columns[-1][:COLUMN_NAME_DISPLAY_LENGTH]
        if len(columns[-1]) > COLUMN_NAME_DISPLAY_LENGTH - 1:
            lastcol += "..."

        filename = file.filename[:FILENAME_DISPLAY_LENGTH]
        if len(file.filename) > FILENAME_DISPLAY_LENGTH - 1:
            filename += "..."

        response_data = {
            'filename': filename,
            'columns': columns,
            'firstcol': firstcol,
            'numcols': numcols,
            'lastcol': lastcol,
            'rows': len(data),
            'total_cells': total_cells
        }
        
        # Add warnings if present
        warnings = []
        if size_warning:
            warnings.append(size_warning)
        if cell_warning:
            warnings.append(cell_warning)
        if warnings:
            response_data['warnings'] = warnings
            logger.info(f"Upload warnings for {filename}: {'; '.join(warnings)}")

        return jsonify(response_data)

### Section 4: run correlation matrices route
    ## when user clicks 'get matrices' -> generate matrices in pdf and xlsx
@app.route('/auto-detect-transformers', methods=['POST'])
def auto_detect_transformers():
    """Auto-detect categorical columns from selected indicators.
    
    Analyzes the selected indicator columns and identifies which ones are categorical
    (non-numeric or numeric with low cardinality) that should be transformed.
    
    Returns:
        JSON: Response with list of column indices that should be transformed.
    """
    if 'data' not in memStorage:
        return jsonify({'error': 'No data uploaded. Please upload a file first.'}), HTTP_BAD_REQUEST
    
    data = request.json
    if not data or 'indicators' not in data:
        return jsonify({'error': 'No indicators provided'}), HTTP_BAD_REQUEST
    
    try:
        df = memStorage['data']
        selected_indicators = data['indicators']
        
        # Convert to list if single value
        if isinstance(selected_indicators, (int, np.integer)):
            selected_indicators = [selected_indicators]
        
        # Get indicator column names
        try:
            indicator_names = df.columns.take(selected_indicators).tolist()
        except (KeyError, IndexError, TypeError):
            indicator_names = df.columns[selected_indicators].tolist()
        
        # Auto-detect categorical columns using the same logic as choose_columns_from_df
        # A column is categorical if:
        # 1. It's non-numeric (object type), OR
        # 2. It's numeric but has low cardinality (<= 50 unique values) and looks categorical
        categorical_indices = []
        max_cardinality = 50  # Same threshold as in utilites.py
        
        for idx, col_name in enumerate(indicator_names):
            col_idx = selected_indicators[idx]
            col_data = df[col_name]
            
            # Check if non-numeric
            is_object_type = not pd.api.types.is_numeric_dtype(col_data)
            
            # Check cardinality for numeric columns (might be encoded categories)
            unique_count = col_data.nunique(dropna=True)
            is_low_cardinality = unique_count <= max_cardinality
            
            # Additional check: if numeric but all values are integers in a small range, might be categorical
            is_integer_like = False
            if pd.api.types.is_numeric_dtype(col_data):
                if col_data.dropna().dtype in [np.int64, np.int32, np.int16, np.int8]:
                    # Check if values are in a small discrete set
                    if unique_count <= max_cardinality and col_data.min() >= 0:
                        is_integer_like = True
            
            # Consider categorical if:
            # - Non-numeric type, OR
            # - Numeric but low cardinality and integer-like (likely encoded categories)
            if is_object_type or (is_low_cardinality and is_integer_like):
                categorical_indices.append(col_idx)
        
        return jsonify({
            'transformer_indices': categorical_indices,
            'message': f'Found {len(categorical_indices)} categorical column(s) to transform.'
        })
        
    except Exception as e:
        logger.error(f"Error auto-detecting transformers: {e}", exc_info=True)
        return jsonify({'error': f'Error detecting categorical columns: {str(e)}'}), HTTP_INTERNAL_SERVER_ERROR


@app.route('/correlationMatrices', methods=['POST'])
def corr():
    """Generate correlation matrices for numeric columns.
    
    Creates correlation matrices using different methods (Pearson, Spearman, Kendall)
    and saves them as PDF and Excel files.
    
    Returns:
        JSON: Response with correlation image paths and descriptive statistics on success.
            Returns error message on failure.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), HTTP_BAD_REQUEST

    if 'data' not in memStorage:
        return jsonify({'error': 'No data uploaded. Please upload a file first.'}), HTTP_BAD_REQUEST

    df = memStorage['data']
    drop_missing = data.get('dropMissing', 'none')
    impute_strategy = data.get('imputeStrategy', 'none')
    drop_zero = data.get('dropZero', 'none')
    
    # Handle colsIgnore with validation
    if 'colsIgnore' not in data:
        return jsonify({'error': 'Missing required field: colsIgnore'}), HTTP_BAD_REQUEST
    
    if data['colsIgnore'] == 'all':
        colsNames = df.columns
    else:
        try:
            colsIndices = data['colsIgnore']
            colsNames = df.columns[colsIndices]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error accessing column indices: {e}", exc_info=True)
            return jsonify({'error': 'Invalid column indices provided.'}), HTTP_BAD_REQUEST

    
    dfCorr = df[colsNames].select_dtypes(include='number')
    if dfCorr.empty:
        return jsonify({'error': 'No numeric columns available for correlation matrices.'}), HTTP_BAD_REQUEST

    if impute_strategy in {'0', '0.01'}:
        impute_strategy = float(impute_strategy)

    dfCorr = preprocess_data(
        df=dfCorr,
        target_cols=dfCorr.columns.tolist(),
        indicator_cols=dfCorr.columns.tolist(),
        drop_missing=drop_missing,
        impute_strategy=impute_strategy,
        drop_zero=drop_zero,
    )
    if dfCorr.empty:
        return jsonify({'error': 'No rows available after preprocessing for correlation matrices.'}), HTTP_BAD_REQUEST

    USER_VIS_DIR.mkdir(parents=True, exist_ok=True)
    corr_methods = [
        ("Pearson Correlation", "pearson"),
        ("Spearman Correlation", "spearman"),
        ("Kendall Correlation", "kendall"),
    ]

    correlation_images = {}
    for title, method in corr_methods:
        correlation_image_path = USER_VIS_DIR / f"correlation_{method}.png"
        corr_matrix = dfCorr.corr(method=method)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            corr_matrix,
            ax=ax,
            cmap="vlag",
            center=0,
            annot=False,
            linewidths=CORRELATION_HEATMAP_LINEWIDTH,
            linecolor="white",
            cbar_kws={"shrink": CORRELATION_CBAR_SHRINK},
        )
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(correlation_image_path, dpi=IMAGE_DPI)
        plt.close(fig)
        correlation_images[method] = f"/user-visualizations/correlation_{method}.png"

    pdf_path = USER_VIS_DIR / "correlation_matrices.pdf"
    pairplot_image = None
    with PdfPages(pdf_path) as pdf_pages:
        for title, method in corr_methods:
            corr_matrix = dfCorr.corr(method=method)
            fig, ax = plt.subplots(figsize=(7, 7))
            sns.heatmap(
                corr_matrix,
                ax=ax,
                cmap="vlag",
                center=0,
                annot=True,
                fmt=".2f",
                linewidths=CORRELATION_HEATMAP_LINEWIDTH,
                linecolor="white",
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(title)
            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.close(fig)

        numeric_columns = dfCorr.columns.tolist()
        if len(numeric_columns) >= 2:
            x_col, y_col = numeric_columns[0], numeric_columns[1]
            safe_x = secure_filename(str(x_col))
            safe_y = secure_filename(str(y_col))
            pairplot_path = USER_VIS_DIR / f"pairplot_{safe_x}_{safe_y}.png"
            grid = sns.pairplot(dfCorr[[x_col, y_col]], diag_kind="hist")
            grid.savefig(pairplot_path, dpi=IMAGE_DPI)
            pdf_pages.savefig(grid.fig)
            plt.close("all")
            pairplot_image = f"/user-visualizations/pairplot_{safe_x}_{safe_y}.png"

    xlsx_path = USER_VIS_DIR / "correlation_matrices.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        dfCorr.describe().T.to_excel(writer, sheet_name="Descriptive Statistics")
        for title, method in corr_methods:
            dfCorr.corr(method=method).to_excel(writer, sheet_name=title)

    # Include all columns in descriptive statistics (not just first 5)
    selected_columns = list(dfCorr.columns)
    descriptive_stats = []
    if selected_columns:
        stats_df = dfCorr[selected_columns].describe(percentiles=[0.25, 0.5, 0.75, 1.0]).T
        stats_df = stats_df.rename(columns={
            "count": "n",
            "min": "min",
            "max": "max",
            "mean": "mean",
            "std": "std",
            "25%": "25",
            "50%": "50",
            "75%": "75",
            "100%": "100",
        }).reset_index().rename(columns={"index": "column"})
        stats_df["column"] = stats_df["column"].astype(str).str.slice(0, 10)
        stats_df = stats_df[["column", "n", "min", "max", "mean", "std", "25", "50", "75", "100"]].round(4)
        descriptive_stats = stats_df.to_dict(orient="records")

    numeric_columns = dfCorr.columns.tolist()

    return jsonify({
        "correlation_image": correlation_images.get("pearson"),
        "correlation_images": correlation_images,
        "descriptive_stats": descriptive_stats,
        "numeric_columns": numeric_columns,
        "pairplot_image": pairplot_image,
    })


@app.route('/pairplot', methods=['POST'])
def pairplot():
    """Generate pairplot visualization for two numeric columns.
    
    Creates a pairplot visualization showing the relationship between two selected
    numeric columns and saves it as a PNG image.
    
    Returns:
        JSON: Response with pairplot image path on success, error message on failure.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), HTTP_BAD_REQUEST

    if 'data' not in memStorage:
        return jsonify({'error': 'No data uploaded. Please upload a file first.'}), HTTP_BAD_REQUEST
    
    # Validate required fields
    if 'colsIgnore' not in data:
        return jsonify({'error': 'Missing required field: colsIgnore'}), HTTP_BAD_REQUEST

    df = memStorage['data']
    x_col = data.get('x')
    y_col = data.get('y')
    drop_missing = data.get('dropMissing', 'none')
    impute_strategy = data.get('imputeStrategy', 'none')
    drop_zero = data.get('dropZero', 'none')
    
    if not x_col or not y_col:
        return jsonify({'error': 'Missing required fields: x, y'}), HTTP_BAD_REQUEST

    df_numeric = df.select_dtypes(include='number')
    if df_numeric.empty:
        return jsonify({'error': 'No numeric columns available for pairplot.'}), HTTP_BAD_REQUEST

    if impute_strategy in {'0', '0.01'}:
        impute_strategy = float(impute_strategy)

    df_numeric = preprocess_data(
        df=df_numeric,
        target_cols=df_numeric.columns.tolist(),
        indicator_cols=df_numeric.columns.tolist(),
        drop_missing=drop_missing,
        impute_strategy=impute_strategy,
        drop_zero=drop_zero,
    )

    if df_numeric.empty:
        return jsonify({'error': 'No rows available after preprocessing for pairplot.'}), HTTP_BAD_REQUEST

    if x_col not in df_numeric.columns or y_col not in df_numeric.columns:
        return jsonify({'error': 'Selected columns are not available for pairplot.'}), HTTP_BAD_REQUEST

    safe_x = secure_filename(str(x_col))
    safe_y = secure_filename(str(y_col))
    pairplot_path = USER_VIS_DIR / f"pairplot_{safe_x}_{safe_y}.png"
    grid = sns.pairplot(df_numeric[[x_col, y_col]], diag_kind="hist")
    grid.savefig(pairplot_path, dpi=150)
    plt.close("all")

    return jsonify({'pairplot_image': f"/user-visualizations/pairplot_{safe_x}_{safe_y}.png"})


### Section 5: Route for preprocessing
    ## when user clicks 'Process' this sends column names selected for the predictors + indicators to display 
@app.route('/preprocess', methods=['POST'])
def preprocess():
    """Preprocess data and return selected column names.
    
    Validates selected indicators, predictors, and stratify column indices,
    then returns the corresponding column names for display.
    
    Returns:
        JSON: Response with predictor names, indicator names, and stratify column
            name on success, error message on failure.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), HTTP_BAD_REQUEST
    
    # Input validation
    if 'indicators' not in data or 'predictors' not in data or 'stratify' not in data:
        return jsonify({'error': 'Missing required fields: indicators, predictors, stratify'}), HTTP_BAD_REQUEST
    
    if 'data' not in memStorage:
        return jsonify({'error': 'No data uploaded. Please upload a file first.'}), HTTP_BAD_REQUEST
    
    # Validate required fields
    required_fields = ['indicators', 'predictors', 'stratify']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), HTTP_BAD_REQUEST
    
    try:
        selected_indicators = data['indicators']
        selected_predictors = data['predictors']
        stratify = data['stratify']
        df = memStorage['data']
        # Getting the column names from the dataframe
        # Use take() to select by position, which handles duplicate column names better
        try:
            # Convert to list if single value
            if isinstance(selected_predictors, (int, np.integer)):
                selected_predictors = [selected_predictors]
            if isinstance(selected_indicators, (int, np.integer)):
                selected_indicators = [selected_indicators]
            
            predictor_names = df.columns.take(selected_predictors).tolist()
            indicator_names = df.columns.take(selected_indicators).tolist()
        except (KeyError, IndexError, TypeError) as e:
            # Fallback to direct indexing
            try:
                predictor_names = df.columns[selected_predictors].tolist()
                indicator_names = df.columns[selected_indicators].tolist()
            except Exception as e2:
                logger.error(f"Error in preprocess: {e2}", exc_info=True)
                return jsonify({'error': f'Invalid column indices provided. This may be due to duplicate column names in your CSV file: {str(e2)}'}), HTTP_BAD_REQUEST
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Error in preprocess: {e}", exc_info=True)
        return jsonify({'error': f'Invalid column indices provided: {str(e)}'}), HTTP_BAD_REQUEST

    # Get stratify column name safely
    try:
        if isinstance(stratify, (int, np.integer)):
            stratify_name = df.columns.take([stratify]).tolist()[0]
        else:
            stratify_name = df.columns[stratify]
            if isinstance(stratify_name, pd.Index):
                stratify_name = stratify_name.tolist()[0] if len(stratify_name) > 0 else str(stratify_name)
            else:
                stratify_name = str(stratify_name)
    except (KeyError, IndexError):
        stratify_name = str(df.columns[stratify]) if stratify < len(df.columns) else ''

    return jsonify({
        'predictors': [str(p)[:COLUMN_NAME_DISPLAY_LENGTH] for p in predictor_names],
        'indicators': [str(i)[:COLUMN_NAME_DISPLAY_LENGTH] for i in indicator_names],
        'stratify': stratify_name[:COLUMN_NAME_DISPLAY_LENGTH] if stratify_name else ''
    })


### Section 6: Route for running model
    ## when user clicks 'Run my Model' this runs the model with the selected parameters and returns the performance results

def _safe_select_columns(df, column_names_or_indices):
    """Safely select columns from a dataframe, handling duplicate column names.
    
    If column_names_or_indices contains duplicate names and the dataframe has
    duplicate columns, selects by position using the original indices.
    
    Args:
        df: DataFrame to select from
        column_names_or_indices: List of column names or indices
    
    Returns:
        DataFrame with selected columns
    """
    # Check if we have duplicate column names in the selection
    if isinstance(column_names_or_indices, list) and len(column_names_or_indices) != len(set(column_names_or_indices)):
        # Has duplicates in selection - need to use position-based selection
        # Convert names to indices
        indices = []
        for name in column_names_or_indices:
            # Find all positions where this name appears
            matches = [i for i, col in enumerate(df.columns) if col == name]
            if matches:
                indices.append(matches[0])  # Take first occurrence
            else:
                raise ValueError(f"Column '{name}' not found in dataframe")
        return df.iloc[:, indices]
    else:
        # No duplicates or single column - use normal selection
        try:
            return df[column_names_or_indices]
        except KeyError as e:
            # If selection fails due to duplicates, try position-based
            if "not unique" in str(e).lower():
                # Convert to indices and use iloc
                indices = [df.columns.get_loc(name) if isinstance(name, str) else name 
                          for name in column_names_or_indices]
                # Handle duplicate names by taking first occurrence
                seen = {}
                unique_indices = []
                for i, name in enumerate(column_names_or_indices):
                    if name not in seen:
                        seen[name] = True
                        if isinstance(name, str):
                            # Get first occurrence of this column name
                            pos = df.columns.tolist().index(name)
                            unique_indices.append(pos)
                        else:
                            unique_indices.append(name)
                return df.iloc[:, unique_indices]
            raise

def _run_model_training(session_id: str, data: dict):
    """Run model training in a background thread.
    
    This function handles the complete model training pipeline including data
    preprocessing, model selection, training, and result storage. Progress is
    tracked via the progress tracker and results are stored in memStorage.
    
    Args:
        session_id (str): Unique session identifier for progress tracking.
        data (dict): Dictionary containing all training parameters including:
            - indicators: List of indicator column indices
            - predictors: List of predictor column indices
            - hyperparameters: Dictionary of model hyperparameters
            - models: Model name to train
            - scaler: Scaling method to use
            - seedValue: Random seed value
            - testSize: Test set size fraction
            - stratifyBool: Whether to use stratified splitting
            - And other preprocessing/training parameters
    
    Raises:
        ValueError: If required fields are missing or data is invalid.
    """
    tracker = get_tracker(session_id)
    tracker.start()
    
    try:
        # Input validation
        required_fields = ['indicators', 'predictors', 'hyperparameters', 'models', 'nonreq', 
                          'scaler', 'units', 'sigfig', 'seedValue', 'testSize', 'stratifyBool',
                          'dropMissing', 'imputeStrategy', 'dropZero', 'quantileBinDict',
                          'useTransformer', 'transformerCols']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        if 'data' not in memStorage:
            raise ValueError("No data uploaded. Please upload a file first.")
        
        # getting all the parameters from the front end
        selected_indicators = data['indicators']
        selected_predictors = data['predictors']
        hyperparameters = data['hyperparameters']
        modelName = data['models']
        nonreq = data['nonreq']
        scaler = data['scaler']
        units = data['units']
        sigfig = data['sigfig']
        seed = data['seedValue']
        raw_test = data.get('testSize', '0.20')
        try:
            testSize = float(raw_test) if raw_test not in (None, '') else 0.2
        except (TypeError, ValueError):
            testSize = 0.2
        stratifyBool = data['stratifyBool']
        drop_missing = data['dropMissing']
        impute_strategy = data['imputeStrategy']
        drop_zero = data['dropZero']
        quantileBinDict = data['quantileBinDict']
        useTransformer = data['useTransformer']
        transformerCols = data['transformerCols']
        cross_validation_type = data.get('crossValidationType', 'None')
        cross_validation_folds = int(data.get('crossValidationFolds', DEFAULT_CV_FOLDS) or DEFAULT_CV_FOLDS)
        
        # Advanced features parameters
        hyperparameter_search = data.get('hyperparameterSearch', 'none')
        search_cv_folds = int(data.get('searchCVFolds', 5) or 5)
        search_n_iter = int(data.get('searchNIter', DEFAULT_SEARCH_ITERATIONS) or DEFAULT_SEARCH_ITERATIONS)
        feature_selection_method = data.get('featureSelectionMethod', 'none')
        feature_selection_k = data.get('featureSelectionK')
        outlier_method = data.get('outlierMethod', 'none')
        outlier_action = data.get('outlierAction', 'remove')
        modeling_mode = data.get('modelingMode', 'simple')  # 'simple', 'advanced', or 'automl'
        
        # Normalize cross_validation_type for comparison (handle both 'None' and 'none')
        cv_type_normalized = str(cross_validation_type).strip() if cross_validation_type else 'None'
        cv_enabled = cv_type_normalized.lower() not in ['none', '']
        
        # Set up progress tracking based on selected features
        tracker.set_stage_enabled('outlier_handling', outlier_method != 'none')
        tracker.set_stage_enabled('feature_selection', feature_selection_method != 'none')
        tracker.set_stage_enabled('hyperparameter_search', hyperparameter_search != 'none')
        tracker.set_stage_enabled('cross_validation', cv_enabled)
        
        # Advanced features are now implemented - log when used
        if hyperparameter_search != 'none':
            logger.info(f"Using hyperparameter search ({hyperparameter_search}) with {search_cv_folds} CV folds.")
            tracker.update_stage('hyperparameter_search', 'pending', 0, 
                               f'Will search {search_n_iter} iterations with {search_cv_folds} CV folds')
        if feature_selection_method != 'none':
            logger.info(f"Using feature selection ({feature_selection_method}) with k={feature_selection_k}.")
            tracker.update_stage('feature_selection', 'pending', 0, 
                               f'Selecting {feature_selection_k} features using {feature_selection_method}')
        if outlier_method != 'none':
            logger.info(f"Using outlier handling ({outlier_method}) with action: {outlier_action}.")
            tracker.update_stage('outlier_handling', 'pending', 0, 
                               f'Detecting outliers using {outlier_method} ({outlier_action})')
        if cv_enabled:
            logger.info(f"Using cross-validation ({cross_validation_type}) with {cross_validation_folds} folds.")
            tracker.update_stage('cross_validation', 'pending', 0, 
                               f'Will run {cross_validation_type} with {cross_validation_folds} folds')
        
        # Data preprocessing
        tracker.update_stage('data_preprocessing', 'running', 10, 'Loading data...')
        df = memStorage['data']
        
        # Select columns by position to avoid issues with duplicate column names
        # Convert indices to list if single value, ensure they're integers
        if isinstance(selected_predictors, (int, np.integer)):
            selected_predictors = [selected_predictors]
        if isinstance(selected_indicators, (int, np.integer)):
            selected_indicators = [selected_indicators]
        
        try:
            # Use take() to select by position, then convert to list to avoid duplicate name issues
            predictor_names = df.columns.take(selected_predictors).tolist()
            indicator_names = df.columns.take(selected_indicators).tolist()
        except (KeyError, IndexError) as e:
            # Fallback to direct indexing if take() fails
            try:
                predictor_names = df.columns[selected_predictors].tolist()
                indicator_names = df.columns[selected_indicators].tolist()
            except Exception as e2:
                raise ValueError(f"Error selecting columns: {str(e2)}. This may be due to duplicate column names in your CSV file. Please ensure all column names are unique.")
        
        tracker.update_stage('data_preprocessing', 'running', 30, 'Preparing features and targets...')

        transformer_names = []
        if useTransformer == 'Yes':
            if isinstance(transformerCols, (int, np.integer)):
                transformerCols = [transformerCols]
            try:
                transformer_names = df.columns.take(transformerCols).tolist()
            except (KeyError, IndexError):
                transformer_names = df.columns[transformerCols].tolist()

        stratifyColumn = ''
        stratify_name = ''
        if stratifyBool:
            if 'stratifyColumn' not in data:
                raise ValueError("Missing required field: stratifyColumn (required when stratifyBool is True)")
            stratifyColumn = data['stratifyColumn']
            try:
                if isinstance(stratifyColumn, (int, np.integer)):
                    stratify_name = df.columns.take([stratifyColumn]).tolist()[0]
                else:
                    stratify_name = df.columns[stratifyColumn]
                    if isinstance(stratify_name, pd.Index):
                        stratify_name = stratify_name.tolist()[0] if len(stratify_name) > 0 else stratify_name
            except (KeyError, IndexError):
                stratify_name = df.columns[stratifyColumn]
                if isinstance(stratify_name, pd.Index):
                    stratify_name = stratify_name.tolist()[0] if len(stratify_name) > 0 else stratify_name

        if seed:  # if user gives seed add to storage
            memStorage['seed'] = seed
        else:  # no seed given
            # if 'seed' not in memStorage.keys(): #if seed not already saved then add random seed
            seed = random.randint(0, RANDOM_SEED_MAX)
                #memStorage['seed'] = 'random'
            #else: 
            #    seed = memStorage['seed']   #else use prior randomly generated seed or keep generating random seed? - ask rowan need way to generate new random seed
            #if given seed then given no seed, want new random

        tracker.update_stage('data_preprocessing', 'running', 50, 'Cleaning and preprocessing data...')
    ## Preprocessing
        df = preprocess_data(df=df, target_cols=predictor_names, indicator_cols=indicator_names, drop_missing=drop_missing, impute_strategy=impute_strategy, drop_zero=drop_zero)
        tracker.update_stage('data_preprocessing', 'completed', 100, 'Data preprocessing complete')
        
        # Update model training stage to indicate we're starting
        tracker.update_stage('model_training', 'running', 0, f'Initializing {modelName} model training...')
        quantileBin_results = ''

        if modelName == 'TerraFORMER':
            model

    ## Selecting Model 
    ## Sending targets, indicators, hyperparameters, scaler, stratify yes/no, quantiles/bins, transformers, sigfig, seed/test size to the train_model script
            ## Every train_model script calls the run_classification/clustering/regression_pipeline script
            
        #Regression models
        elif modelName == 'Linear': 
            #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order = train(modelName, predictor_names, df, stratifyBool, df[indicator_names], df[predictor_names], stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig)
            # Use safe column selection to handle potential duplicate column names
            X_data = _safe_select_columns(df, indicator_names)
            y_data = _safe_select_columns(df, predictor_names)
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_linear(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=X_data, y=y_data, stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'BayesianRidge':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_bayesian_ridge(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'ARDRegression':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_ard_regression(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'Ridge': 
            if nonreq:
                RidgeFitIntersept = True
                RidgeNormalize = True
                RidgeCopyX = True
                RidgePositive = True
                if hyperparameters['RidgeFitIntersept'] == 'false':
                    RidgeFitIntersept = False
                if hyperparameters['RidgeNormalize'] == 'false':
                    RidgeNormalize = False
                if hyperparameters['RidgeCopyX'] == 'false':
                    RidgeCopyX = False
                if hyperparameters['RidgePositive'] == 'false':
                    RidgePositive = False
                            
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name,  units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_ridge(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize,
                            alpha=hyperparameters['alpha'], 
                            solver=hyperparameters['solver'],
                            RidgeFitIntersept = RidgeFitIntersept,
                            RidgeNormalize = RidgeNormalize,
                            RidgeCopyX = RidgeCopyX,
                            RidgePositive = RidgePositive,
                            RidgeMaxIter = hyperparameters['RidgeMaxIter'],
                            RidgeTol = hyperparameters['RidgeTol'],
                            RidgeRandomState = seed,
                            feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                            outlier_method=outlier_method, outlier_action=outlier_action,
                            hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            progress_tracker=tracker, modeling_mode=modeling_mode
                            )
            else: 
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_ridge(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                            alpha=hyperparameters['alpha'],
                            RidgeRandomState = seed,
                            feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                            outlier_method=outlier_method, outlier_action=outlier_action,
                            hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            progress_tracker=tracker)

        elif modelName == 'Lasso': 
            if nonreq: 
                LassoFitIntersept = True
                LassoPrecompute = True
                LassoCopyX = True
                LassoWarmStart = True
                LassoPositive = True
                if hyperparameters['LassoFitIntersept'] == 'false':
                    LassoFitIntersept = False
                if hyperparameters['LassoPrecompute'] == 'false':
                    LassoPrecompute = False
                if hyperparameters['LassoCopyX'] == 'false':
                    LassoCopyX = False
                if hyperparameters['LassoWarmStart'] == 'false':
                    LassoWarmStart = False
                if hyperparameters['LassoPositive'] == 'false':
                    LassoPositive = False

                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_lasso(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                            alpha = hyperparameters['alpha'], 
                            max_iter = hyperparameters['max_iter'],
                            fit_intercept = LassoFitIntersept,
                            precompute = LassoPrecompute,
                            copy_X = LassoCopyX,
                            tol = hyperparameters['LassoTol'],
                            warm_start = LassoWarmStart,
                            positive = LassoPositive,
                            random_state = seed,
                            selection = hyperparameters['LassoSelection'],
                            feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                            outlier_method=outlier_method, outlier_action=outlier_action,
                            hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            progress_tracker=tracker, modeling_mode=modeling_mode
                            )
            else:
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_lasso(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                            alpha=hyperparameters['alpha'],
                            random_state = seed,
                            feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                            outlier_method=outlier_method, outlier_action=outlier_action,
                            hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            progress_tracker=tracker, modeling_mode=modeling_mode
                            )

        elif modelName == 'ElasticNet': 
            #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_elasticnet(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                        alpha=hyperparameters['alpha'], 
                        l1_ratio=hyperparameters['l1_ratio'],
                        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                        outlier_method=outlier_method, outlier_action=outlier_action,
                        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                        progress_tracker=tracker)

        elif modelName == 'SVM': #precomputed kernal doesn't work - weird with multiregressor
            if nonreq:
                SVMshrinking = True
                SVMprobability = True
                SVMBreakTies = True
                SVMverbose = True
                if hyperparameters['SVMshrinking'] == 'false':
                    SVMshrinking = False
                if hyperparameters['SVMprobability'] == 'false':
                    SVMprobability = False
                if hyperparameters['SVMBreakTies'] == 'false':
                    SVMBreakTies = False
                if hyperparameters['SVMverbose'] == 'false':
                    SVMverbose = False

                kernel = hyperparameters['kernel']
                if kernel =='rbf':
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_svr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                                C=hyperparameters['C'], 
                                kernel=kernel,  
                                gamma=hyperparameters['gamma'],
                                
                                    coef0=hyperparameters['SVMcoef0'],
                                    shrinking=SVMshrinking,
                                    probability=SVMprobability,
                                    tol=hyperparameters['SVMtol'],
                                    cache_size=hyperparameters['SVMCacheSize'],
                                    class_weight=hyperparameters['SVMClassWeight'],
                                    verbose=SVMverbose,
                                    max_iter=hyperparameters['SVMmaxIter'],
                                    decision_function_shape=hyperparameters['SVMdecisionFunctionShape'],
                                    break_ties=SVMBreakTies,
                                    random_state=seed,
                                    )
                    
                elif kernel =='poly':
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_svr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                                C=hyperparameters['C'], 
                                kernel=kernel, 
                                degree=hyperparameters['degree'], 
                                gamma=hyperparameters['gamma'],
                                coef0=hyperparameters['SVMcoef0'],
                                    shrinking=SVMshrinking,
                                    probability=SVMprobability,
                                    tol=hyperparameters['SVMtol'],
                                    cache_size=hyperparameters['SVMCacheSize'],
                                    class_weight=hyperparameters['SVMClassWeight'],
                                    verbose=SVMverbose,
                                    max_iter=hyperparameters['SVMmaxIter'],
                                    decision_function_shape=hyperparameters['SVMdecisionFunctionShape'],
                                    break_ties=SVMBreakTies,
                                    random_state=seed,)
                else:
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_svr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                                C=hyperparameters['C'], 
                                kernel=kernel,
                                coef0=hyperparameters['SVMcoef0'],
                                    shrinking=SVMshrinking,
                                    probability=SVMprobability,
                                    tol=hyperparameters['SVMtol'],
                                    cache_size=hyperparameters['SVMCacheSize'],
                                    class_weight=hyperparameters['SVMClassWeight'],
                                    verbose=SVMverbose,
                                    max_iter=hyperparameters['SVMmaxIter'],
                                    decision_function_shape=hyperparameters['SVMdecisionFunctionShape'],
                                    break_ties=SVMBreakTies,
                                    random_state=seed,)

            else:
                kernel = hyperparameters['kernel']
                if kernel =='rbf':
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, 
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_svr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                        C=hyperparameters['C'], kernel=kernel,  gamma=hyperparameters['gamma'],
                            random_state = seed)
                elif kernel =='poly':
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, 
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_svr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                        C=hyperparameters['C'], kernel=kernel, degree=hyperparameters['degree'], gamma=hyperparameters['gamma'],
                            random_state = seed)
                else:
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, 
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_svr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                        C=hyperparameters['C'], kernel=kernel,
                            random_state = seed)

        elif modelName == 'RF': 
            if nonreq:
                RFBoostrap = True
                RFoobScore = True
                RFWarmStart = True
                if hyperparameters['RFBoostrap'] == 'false':
                    RFBoostrap = True
                if hyperparameters['RFoobScore'] == 'false':
                    RFoobScore = True
                if hyperparameters['RFWarmStart'] == 'false':
                    RFWarmStart = True

                val = None
                if 'max_depth' in hyperparameters.keys():
                    val = hyperparameters['max_depth']
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_rf(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                            n_estimators=hyperparameters['n_estimators'], 
                            max_depth=val, 
                            min_samples_split=hyperparameters['min_samples_split'], 
                            min_samples_leaf=hyperparameters['min_samples_leaf'], 
                            random_state = seed,
                            min_weight_fraction_leaf=hyperparameters['RFmin_weight_fraction_leaf'],
                            max_leaf_nodes=hyperparameters['RFMaxLeafNodes'],
                            min_impurity_decrease=hyperparameters['RFMinImpurityDecrease'],
                            bootstrap=RFBoostrap,
                            oob_score=RFoobScore,
                            n_jobs=hyperparameters['RFNJobs'],
                            verbose=hyperparameters['RFVerbose'],
                            warm_start=RFWarmStart,
                            feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                            outlier_method=outlier_method, outlier_action=outlier_action,
                            hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            progress_tracker=tracker, modeling_mode=modeling_mode
                            )
            else:
                val = None
                if 'max_depth' in hyperparameters.keys():
                    val = hyperparameters['max_depth']
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_rf(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                        n_estimators=hyperparameters['n_estimators'],
                        feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                        outlier_method=outlier_method, outlier_action=outlier_action,
                        hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                        progress_tracker=tracker
                        )

        elif modelName == 'ExtraTrees':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_extra_trees(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        # NOTE: LogisticRegression is a classifier, not a regressor. It has been removed from regression section.
        # Use Logistic_classifier instead for classification tasks.

        elif modelName == 'MLP': 
            if nonreq:
                    MLPShuffle = True
                    MLPVerbose = True
                    MLPWarmStart = True
                    MLPNesterovsMomentum = True
                    MLPEarlyStopping = True
                    if hyperparameters['MLPShuffle']=='false':
                        MLPShuffle= False
                    if hyperparameters['MLPVerbose']=='false':
                        MLPVerbose= False
                    if hyperparameters['MLPWarmStart']=='false':
                        MLPWarmStart= False
                    if hyperparameters['MLPNesterovsMomentum']=='false':
                        MLPNesterovsMomentum= False
                    if hyperparameters['MLPEarlyStopping']=='false':
                        MLPEarlyStopping= False


                    hiddenlayersizeString = '(' + hyperparameters['hidden_layer_sizes1'] + ',' + hyperparameters['hidden_layer_sizes2']
                    if hyperparameters['hidden_layer_sizes3']:
                        hiddenlayersizeString += ',' + hyperparameters['hidden_layer_sizes3'] + ')'
                    else:
                        hiddenlayersizeString += ')'
                    # if not hyperparameters['hidden_layer_sizes1']:
                    #     hiddenlayersizeString = None
                    
                    #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler, quantileBin_results = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler,  seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                    train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_mlp(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize,
                            feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k,
                            outlier_method=outlier_method, outlier_action=outlier_action,
                            hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                                hidden_layer_sizes=ast.literal_eval(hiddenlayersizeString), 
                                activation=hyperparameters['activation'], 
                                solver=hyperparameters['solver'], 
                                alpha=hyperparameters['alpha'], 
                                learning_rate=hyperparameters['learning_rate'], 
                                max_iter=hyperparameters['MLPMaxIter'],
                                batch_size=hyperparameters['MLPBatchSize'],
                                beta_1=hyperparameters['MLPBeta1'],
                                beta_2=hyperparameters['MLPBeta2'],
                                early_stopping=MLPEarlyStopping,
                                epsilon=hyperparameters['MLPEpsilon'],
                                learning_rate_init=hyperparameters['MLPLearningRateInit'],
                                momentum=hyperparameters['MLPMomentum'],
                                nesterovs_momentum=MLPNesterovsMomentum,
                                power_t=hyperparameters['MLPPowerT'],
                                random_state=seed,
                                shuffle=MLPShuffle,
                                tol=hyperparameters['MLPTol'],
                                validation_fraction=hyperparameters['MLPValidationFraction'],
                                verbose=MLPVerbose,
                                warm_start=MLPWarmStart
                    )
                    #self.model=MLPRegressor()
            
            else:
                hiddenlayersizeString = '(' + hyperparameters['hidden_layer_sizes1'] + ',' + hyperparameters['hidden_layer_sizes2']
                if hyperparameters['hidden_layer_sizes3']:
                    hiddenlayersizeString += ',' + hyperparameters['hidden_layer_sizes3'] + ')'
                else:
                    hiddenlayersizeString += ')'
                # if not hyperparameters['hidden_layer_sizes1']:
                #     hiddenlayersizeString = None
                
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler, quantileBin_results = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_mlp(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            progress_tracker=tracker,
                            hidden_layer_sizes=ast.literal_eval(hiddenlayersizeString), 
                            activation=hyperparameters['activation'], 
                            solver=hyperparameters['solver'],
                            random_state = seed
                            )
                #self.model=MLPRegressor()

        # NOTE: Perceptron is a classifier, not a regressor. It has been removed from regression section.
        # Use Perceptron_classifier instead for classification tasks.

        elif modelName == 'K-Nearest': 
            if nonreq:
                metricParams = None
                if hyperparameters['KNearestMetricParams'] != '':
                    metricParams = hyperparameters['KNearestMetricParams']

                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler,  seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_knn(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                            n_neighbors=hyperparameters['n_neighbors'], 
                            metric=hyperparameters['metric'],
                            algorithm=hyperparameters['KNearestAlgorithm'],
                            leaf_size=hyperparameters['KNearestLeafSize'],
                            metric_params=metricParams,
                            n_jobs=hyperparameters['KNearestNJobs'],
                            p=hyperparameters['KNearestP'],
                            weights=hyperparameters['KNearestWeights'],
                        )

            else: 
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_knn(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                        n_neighbors=hyperparameters['n_neighbors'], 
                        )

        elif modelName == 'gradient_boosting': #weird with multiregressor
            if nonreq:
                GBWarmStart = True
                if hyperparameters['GBWarmStart']:
                    GBWarmStart=False

                init=None
                if hyperparameters['GBInit']!='':
                    init=hyperparameters['GBInit']

                max_features=None
                if hyperparameters['GBMaxFeatrues']!='':
                    max_features=hyperparameters['GBMaxFeatrues']

                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_gb(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                        n_estimators=hyperparameters['n_estimators'], 
                        learning_rate=hyperparameters['learning_rate'], 
                        max_depth=hyperparameters['max_depth'], 
                        loss=hyperparameters['GBLoss'], 
                        subsample=hyperparameters['GBSubsample'], 
                        criterion=hyperparameters['GBCriterion'], 
                        min_samples_split=hyperparameters['GBMinSamplesSplit'], 
                        min_samples_leaf=hyperparameters['GBMinSamplesLeaf'], 
                        min_weight_fraction_leaf=hyperparameters['GBMinWeightFractionLeaf'], 
                        min_impurity_decrease=hyperparameters['GBMinImpurityDecrease'], 
                        init=init,
                        random_state=seed, 
                        max_features=max_features, 
                        alpha=hyperparameters['GBAlpha'], 
                        verbose=hyperparameters['GBVerbose'], 
                        max_leaf_nodes=hyperparameters['GBMaxLeafNodes'], 
                        warm_start=GBWarmStart
                        
                        )
            else:
                #train_results, test_results, params, shapes, train_groups, test_groups, storedModel, y_scaler, X_scaler = train(model_type = modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig,
                train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_gb(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker,
                        n_estimators=hyperparameters['n_estimators'], 
                        learning_rate=hyperparameters['learning_rate'],
                        random_state = seed
                        )

        # Additional Regression Models
        elif modelName == 'AdaBoost':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_adaboost_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'Bagging':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_bagging_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'DecisionTree':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_decision_tree_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'ElasticNetCV':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_elasticnet_cv(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'HistGradientBoosting':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_hist_gradient_boosting_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'Huber':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_huber_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'LARS':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_lars(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'LARSCV':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_lars_cv(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'LassoCV':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_lasso_cv(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'LassoLars':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_lassolars(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'LinearSVR':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_linearsvr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'NuSVR':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_nusvr(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'OMP':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_orthogonal_matching_pursuit(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'PassiveAggressive':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_passive_aggressive_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'Quantile':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_quantile_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'RadiusNeighbors':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_radius_neighbors_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'RANSAC':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_ransac_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'RidgeCV':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_ridge_cv(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'SGD':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_sgd_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        elif modelName == 'TheilSen':
            train_results, test_results, params, shapes, storedModel, y_scaler, X_scaler, quantileBin_results, feature_order, feature_selection_info, outlier_info = train_theilsen_regressor(modelName, target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, y_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, progress_tracker=tracker, modeling_mode=modeling_mode)

        #Special Model
        elif modelName == 'Polynomial': #set up
            model = Model(model = modelName, 
                        degree=hyperparameters['degree_specificity'])

        #Classifier Models


        elif modelName == 'Perceptron': #done with nonreq but cant test
            if nonreq: 
                PerceptronFitIntercept = True
                PerceptronShuffle = True
                PerceptronEarlyStopping = True
                PerceptronWarmStart = True
                if hyperparameters['PerceptronFitIntercept'] == 'false':
                    PerceptronFitIntercept=False
                if hyperparameters['PerceptronShuffle'] == 'false':
                    PerceptronShuffle=False
                if hyperparameters['PerceptronEarlyStopping'] == 'false':
                    PerceptronEarlyStopping=False
                if hyperparameters['PerceptronWarmStart'] == 'false':
                    PerceptronWarmStart=False
                    
                model =  model = Model(model = modelName, 
                        max_iter=hyperparameters['max_iter'], 
                        eta0=hyperparameters['eta0'], 
                            penalty=hyperparameters['PerceptronPenalty'],
                            alpha=hyperparameters['PerceptronAlpha'],
                            fit_intercept=PerceptronFitIntercept,
                            tol=hyperparameters['PerceptronTol'],
                            shuffle=PerceptronShuffle,
                            verbose=hyperparameters['PerceptronVerbose'],
                            n_jobs=hyperparameters['PerceptronNJobs'],
                            random_state=seed,
                            early_stopping=PerceptronEarlyStopping,
                            validation_fraction=hyperparameters['PerceptronValidationFraction'],
                            n_iter_no_change=hyperparameters['PerceptronNIterNoChange'],
                            class_weight=hyperparameters['PerceptronClassWeight'],
                            warm_start=PerceptronWarmStart
                        )

            else:
                model = Model(model = modelName, 
                            max_iter=hyperparameters['max_iter'], 
                            eta0=hyperparameters['eta0'], 
                            )

        elif modelName == 'Logistic_classifier':
            if nonreq:
                Class_LogisticDual = True
                Class_LogisticFitIntercept = True
                Class_LogisticWarmStart = True
                if hyperparameters['Class_LogisticDual']=='false':
                    Class_LogisticDual = False
                if hyperparameters['Class_LogisticFitIntercept']=='false':
                    Class_LogisticFitIntercept = False
                if hyperparameters['Class_LogisticWarmStart']=='false':
                    Class_LogisticWarmStart = False

                Class_LogisticClassWeight = None
                Class_LogisticNJobs = None
                Class_Logisticl1Ratio = None
                if hyperparameters['Class_LogisticClassWeight']!='':
                    Class_LogisticClassWeight = hyperparameters['Class_LogisticClassWeight']
                if hyperparameters['Class_LogisticNJobs']!='':
                    Class_LogisticNJobs = hyperparameters['Class_LogisticNJobs']
                if hyperparameters['Class_Logisticl1Ratio']!='':
                    Class_Logisticl1Ratio = hyperparameters['Class_Logisticl1Ratio']

                result_tuple = train_logistic_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names,testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                        Class_LogisticDual = Class_LogisticDual,
                        Class_LogisticFitIntercept = Class_LogisticFitIntercept,
                        Class_LogisticWarmStart = Class_LogisticWarmStart,
                        Class_LogisticSolver = hyperparameters['Class_LogisticSolver'],
                        Class_LogisticMultiClass = hyperparameters['Class_LogisticMultiClass'],
                        Class_CLogistic = hyperparameters['Class_CLogistic'],
                        Class_Logistic_penalty = hyperparameters['Class_Logistic_penalty'],
                        Class_LogisticTol = hyperparameters['Class_LogisticTol'],
                        Class_Logisticintercept_scaling = hyperparameters['Class_Logisticintercept_scaling'],
                        Class_LogisticClassWeight = Class_LogisticClassWeight,
                        Class_LogisticMaxIterations = hyperparameters['Class_LogisticMaxIterations'],
                        Class_LogisticVerbose = hyperparameters['Class_LogisticVerbose'],
                        Class_LogisticNJobs = Class_LogisticNJobs,
                        Class_Logisticl1Ratio = Class_Logisticl1Ratio, )                                                                                                                                                                                                                                                                                        

            else:
                result_tuple = train_logistic_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
                if len(result_tuple) >= 9:
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = result_tuple
                else:
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order = result_tuple[:8]
                    additional_metrics = None
        
        elif modelName == 'ExtraTrees_classifier':
            result_tuple = train_extra_trees_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'GaussianNB_classifier':
            result_tuple = train_gaussian_nb(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'SGD_classifier':
            result_tuple = train_sgd_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'MLP_classifier':  
            if nonreq:
                    MLPShuffle = True
                    MLPVerbose = True
                    MLPWarmStart = True
                    MLPNesterovsMomentum = True
                    MLPEarlyStopping = True
                    if hyperparameters['MLPShuffle']=='false':
                        MLPShuffle= False
                    if hyperparameters['MLPVerbose']=='false':
                        MLPVerbose= False
                    if hyperparameters['MLPWarmStart']=='false':
                        MLPWarmStart= False
                    if hyperparameters['MLPNesterovsMomentum']=='false':
                        MLPNesterovsMomentum= False
                    if hyperparameters['MLPEarlyStopping']=='false':
                        MLPEarlyStopping= False


                    hiddenlayersizeString = '(' + hyperparameters['hidden_layer_sizes1'] + ',' + hyperparameters['hidden_layer_sizes2']
                    if hyperparameters['hidden_layer_sizes3']:
                        hiddenlayersizeString += ',' + hyperparameters['hidden_layer_sizes3'] + ')'
                    else:
                        hiddenlayersizeString += ')'

                    result_tuple = train_mlp_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize,
                                hidden_layer_sizes=ast.literal_eval(hiddenlayersizeString), 
                                activation=hyperparameters['activation'], 
                                solver=hyperparameters['solver'], 
                                alpha=hyperparameters['alpha'], 
                                learning_rate=hyperparameters['learning_rate'], 
                                max_iter=hyperparameters['MLPMaxIter'],
                                batch_size=hyperparameters['MLPBatchSize'],
                                beta_1=hyperparameters['MLPBeta1'],
                                beta_2=hyperparameters['MLPBeta2'],
                                early_stopping=MLPEarlyStopping,
                                epsilon=hyperparameters['MLPEpsilon'],
                                learning_rate_init=hyperparameters['MLPLearningRateInit'],
                                momentum=hyperparameters['MLPMomentum'],
                                nesterovs_momentum=MLPNesterovsMomentum,
                                power_t=hyperparameters['MLPPowerT'],
                                random_state=seed,
                                shuffle=MLPShuffle,
                                tol=hyperparameters['MLPTol'],
                                validation_fraction=hyperparameters['MLPValidationFraction'],
                                verbose=MLPVerbose,
                                warm_start=MLPWarmStart)
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)
            else:
                hiddenlayersizeString = '(' + hyperparameters['hidden_layer_sizes1'] + ',' + hyperparameters['hidden_layer_sizes2']
                if hyperparameters['hidden_layer_sizes3']:
                    hiddenlayersizeString += ',' + hyperparameters['hidden_layer_sizes3'] + ')'
                else:
                        hiddenlayersizeString += ')'
                result_tuple = train_mlp_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode,
                            hidden_layer_sizes=ast.literal_eval(hiddenlayersizeString), 
                            activation=hyperparameters['activation'], 
                            solver=hyperparameters['solver'],
                            random_state = seed    )
                report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'RF_classifier':
            if nonreq:
                RFBoostrap = True
                RFoobScore = True
                RFWarmStart = True
                if hyperparameters['RFBoostrap'] == 'false':
                    RFBoostrap = True
                if hyperparameters['RFoobScore'] == 'false':
                    RFoobScore = True
                if hyperparameters['RFWarmStart'] == 'false':
                    RFWarmStart = True

                val = None
                if 'max_depth' in hyperparameters.keys():
                    val = hyperparameters['max_depth']

                result_tuple = train_rf_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            n_estimators=hyperparameters['n_estimators'], 
                            max_depth=val, 
                            min_samples_split=hyperparameters['min_samples_split'], 
                            min_samples_leaf=hyperparameters['min_samples_leaf'], 
                            random_state = seed,
                            min_weight_fraction_leaf=hyperparameters['RFmin_weight_fraction_leaf'],
                            max_leaf_nodes=hyperparameters['RFMaxLeafNodes'],
                            min_impurity_decrease=hyperparameters['RFMinImpurityDecrease'],
                            bootstrap=RFBoostrap,
                            oob_score=RFoobScore,
                            n_jobs=hyperparameters['RFNJobs'],
                            verbose=hyperparameters['RFVerbose'],
                            warm_start=RFWarmStart,
                            )
                report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)
            else:
                result_tuple = train_rf_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter,
                            n_estimators=hyperparameters['n_estimators'], )
                report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)
       

        elif modelName == 'SVC_classifier':
            if nonreq:
                SVMshrinking = True
                SVMprobability = True
                SVMBreakTies = True
                SVMverbose = True
                if hyperparameters['SVCshrinking'] == 'false':
                    SVMshrinking = False
                if hyperparameters['SVCprobability'] == 'false':
                    SVMprobability = False
                if hyperparameters['SVCBreakTies'] == 'false':
                    SVMBreakTies = False
                if hyperparameters['SVCverbose'] == 'false':
                    SVMverbose = False

                kernel = hyperparameters['kernel']
                if kernel =='rbf':
                    result_tuple = train_svc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize,
                                C=hyperparameters['C'], 
                                kernel=kernel,  
                                gamma=hyperparameters['gamma'],
                                coef0=hyperparameters['SVCcoef0'],
                                shrinking=SVMshrinking,
                                probability=SVMprobability,
                                tol=hyperparameters['SVCtol'],
                                cache_size=hyperparameters['SVCCacheSize'],
                                class_weight=hyperparameters['SVCClassWeight'],
                                verbose=SVMverbose,
                                max_iter=hyperparameters['SVCmaxIter'],
                                decision_function_shape=hyperparameters['SVCdecisionFunctionShape'],
                                break_ties=SVMBreakTies,
                                random_state=seed,
                                )
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

                    
                elif kernel =='poly':
                    result_tuple = train_svc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize,
                            C=hyperparameters['C'], 
                            kernel=kernel, 
                            degree=hyperparameters['degree'], 
                            gamma=hyperparameters['gamma'],
                            coef0=hyperparameters['SVCcoef0'],
                            shrinking=SVMshrinking,
                            probability=SVMprobability,
                            tol=hyperparameters['SVCtol'],
                            cache_size=hyperparameters['SVCCacheSize'],
                            class_weight=hyperparameters['SVCClassWeight'],
                            verbose=SVMverbose,
                            max_iter=hyperparameters['SVCmaxIter'],
                            decision_function_shape=hyperparameters['SVCdecisionFunctionShape'],
                            break_ties=SVMBreakTies,
                            random_state=seed,)
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

                else:
                    result_tuple = train_svc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize,
                                C=hyperparameters['C'], 
                                kernel=kernel, 
                                degree=hyperparameters['degree'], 
                                gamma=hyperparameters['gamma'],
                                coef0=hyperparameters['SVCcoef0'],
                                shrinking=SVMshrinking,
                                probability=SVMprobability,
                                tol=hyperparameters['SVCtol'],
                                cache_size=hyperparameters['SVCCacheSize'],
                                class_weight=hyperparameters['SVCClassWeight'],
                                verbose=SVMverbose,
                                max_iter=hyperparameters['SVCmaxIter'],
                                decision_function_shape=hyperparameters['SVCdecisionFunctionShape'],
                                break_ties=SVMBreakTies,
                                random_state=seed,)
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)


            else:
                kernel = hyperparameters['kernel']
                if kernel =='rbf':
                    result_tuple = train_svc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode,
                            C=hyperparameters['C'], kernel=kernel,  gamma=hyperparameters['gamma'],
                            random_state = seed)
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

                elif kernel =='poly':
                    result_tuple = train_svc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode,
                            C=hyperparameters['C'], kernel=kernel, degree=hyperparameters['degree'], gamma=hyperparameters['gamma'],
                            random_state = seed)
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)
                else:
                    result_tuple = train_svc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode,
                            C=hyperparameters['C'], kernel=kernel,
                            random_state = seed)
                    report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        # Additional Classification Models
        elif modelName == 'AdaBoost_classifier':
            result_tuple = train_adaboost_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'Bagging_classifier':
            result_tuple = train_bagging_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'BernoulliNB_classifier':
            result_tuple = train_bernoulli_nb(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'CategoricalNB_classifier':
            result_tuple = train_categorical_nb(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'ComplementNB_classifier':
            result_tuple = train_complement_nb(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'DecisionTree_classifier':
            result_tuple = train_decision_tree_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'GradientBoosting_classifier':
            result_tuple = train_gradient_boosting_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'HistGradientBoosting_classifier':
            result_tuple = train_hist_gradient_boosting_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'KNeighbors_classifier':
            result_tuple = train_kneighbors_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'LDA_classifier':
            result_tuple = train_linear_discriminant_analysis(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'LinearSVC_classifier':
            result_tuple = train_linearsvc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'MultinomialNB_classifier':
            result_tuple = train_multinomial_nb(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'NuSVC_classifier':
            result_tuple = train_nusvc(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'PassiveAggressive_classifier':
            result_tuple = train_passive_aggressive_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'QDA_classifier':
            result_tuple = train_quadratic_discriminant_analysis(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        elif modelName == 'Ridge_classifier':
            result_tuple = train_ridge_classifier(target_variables=predictor_names, train_data=df, use_stratified_split=stratifyBool, X=df[indicator_names], y=df[predictor_names], stratifyColumn=stratify_name, units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, testSize=testSize, feature_selection_method=feature_selection_method, feature_selection_k=feature_selection_k, outlier_method=outlier_method, outlier_action=outlier_action, hyperparameter_search=hyperparameter_search, search_cv_folds=search_cv_folds, search_n_iter=search_n_iter, modeling_mode=modeling_mode)
            report, cm, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, additional_metrics = unpack_classification_result(result_tuple)

        #Cluster Models 
        elif modelName == 'kmeans':
            if nonreq:
                copy_x = True
                if hyperparameters['copy_x']=="false":
                    copy_x=False

                if hyperparameters['n_init']=='auto':
                    n_init='auto'
                elif hyperparameters['n_init']=='warn':
                    n_init='warn'
                else:
                    n_init = int(hyperparameters['n_init'])

                train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid  =  train_kmeans(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8,
                                n_clusters=hyperparameters['n_clusters'],
                                init=hyperparameters['init'],
                                n_init=n_init,
                                max_iter=hyperparameters['max_iter'],
                                tol=hyperparameters['tol'],
                                verbose=hyperparameters['verbose'],
                                copy_x=copy_x,
                                algorithm=hyperparameters['algorithm'],)
        
            else:
                train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid  =  train_kmeans(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8,
                                n_clusters=hyperparameters['n_clusters'])
        
        elif modelName == 'gmm':
            if nonreq:
                warm_start = True
                if hyperparameters['warm_start']=='false':
                    warm_start=False

                if hyperparameters['weights_init']=='':
                    weights_init=None
                else:
                    weights_init=hyperparameters['weights_init']

                if hyperparameters['means_init']=='':
                    means_init=None
                else:
                    means_init=hyperparameters['means_init']

                if hyperparameters['precisions_init']=='':
                    precisions_init=None
                else:
                    precisions_init=hyperparameters['precisions_init']

                train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_gmm(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8,
                                n_components=hyperparameters['n_components'],
                                covariance_type=hyperparameters['covariance_type'],
                                tol=hyperparameters['tol'],
                                reg_covar=hyperparameters['reg_covar'],
                                max_iter=hyperparameters['max_iter'],
                                n_init=hyperparameters['n_init'],
                                init_params=hyperparameters['init_params'],
                                weights_init=weights_init,
                                means_init=means_init,
                                precisions_init=precisions_init,
                                warm_start=warm_start,
                                verbose=hyperparameters['verbose'],
                                verbose_interval=hyperparameters['verbose_interval'],
                                )
                
            else:
                train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_gmm(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8,
                                n_components=hyperparameters['n_components'])
        
        elif modelName == 'agglo':
            if nonreq:
                compute_distances=False
                if hyperparameters['distance_threshold']=='true':
                    compute_distances=True

                compute_full_tree = 'auto'
                if hyperparameters['compute_full_tree']=='true':
                    compute_full_tree=True
                elif hyperparameters['compute_full_tree']=='false':
                    compute_full_tree=False


                if hyperparameters['connectivity']=='':
                    connectivity=None
                else:
                    connectivity=hyperparameters['connectivity']

                if hyperparameters['memory']=='':
                    memory=None
                else:
                    memory=hyperparameters['memory']

                if hyperparameters['n_clusters']=='':
                    n_clusters=None
                else:
                    n_clusters=hyperparameters['n_clusters']

                if hyperparameters['distance_threshold']=='':
                    distance_threshold=None
                else:
                    distance_threshold=hyperparameters['distance_threshold']

                train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_agglomerative(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8,
                                n_clusters=n_clusters,
                                metric=hyperparameters['metric'],
                                memory=memory,
                                connectivity=connectivity,
                                compute_full_tree=compute_full_tree,
                                linkage=hyperparameters['linkage'],
                                distance_threshold=distance_threshold,
                                compute_distances=compute_distances,)
                
                
            else:   
                if hyperparameters['n_clusters']=='':
                    n_clusters=None
                else:
                    n_clusters=hyperparameters['n_clusters']

                train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_agglomerative(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8,
                                n_clusters=n_clusters)

        elif modelName == 'dbscan':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_dbscan(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        elif modelName == 'birch':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_birch(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        elif modelName == 'spectral':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_spectral(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        # Additional Clustering Models
        elif modelName == 'affinity_propagation':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_affinity_propagation(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        elif modelName == 'bisecting_kmeans':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_bisecting_kmeans(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8)

        elif modelName == 'hdbscan':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_hdbscan(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        elif modelName == 'meanshift':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_meanshift(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        elif modelName == 'minibatch_kmeans':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_minibatch_kmeans(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize, k_min=2, k_max=8)

        elif modelName == 'optics':
            train_results, test_results, params, shapes, storedModel, X_scaler, quantileBin_results, feature_order, best_k, centers, silhouette_grid = train_optics(train_data=df, X=df[indicator_names], units=units, X_scaler_type=scaler, seed=seed, quantileBinDict=quantileBinDict, sigfig=sigfig, useTransformer=useTransformer, categorical_cols=transformer_names, test_size=testSize)

        else:
            raise ValueError('invalid model architecture specified')

        fmt = f".{sigfig}f" 

    #results go in result dictionary and get written to excel file
    
        # Initialize feature selection and outlier info only if not already set by regression pipeline
        # (Regression models set these values, classification/clustering models don't)
        if 'feature_selection_info' not in locals():
            feature_selection_info = None
        if 'outlier_info' not in locals():
            outlier_info = None
        
        cv_file = None
        # Use normalized check for cross validation
        cv_summary_data = None
        if cv_enabled:
            # Update cross-validation status to running
            tracker.update_stage('cross_validation', 'running', 10, 
                               f'Running {cross_validation_type} with {cross_validation_folds} folds...')
            if modelName.endswith('_classifier'):
                cv_result = run_cross_validation(
                    df=df,
                    indicator_names=indicator_names,
                    predictor_names=predictor_names,
                    model=storedModel,
                    scaler=scaler,
                    cv_type=cross_validation_type,
                    cv_folds=cross_validation_folds,
                    useTransformer=useTransformer,
                    transformer_cols=transformer_names,
                    seed=seed,
                    problem_type="classification",
                )
                if cv_result and isinstance(cv_result, tuple) and cv_result[0] is not None:
                    cv_file_path, cv_summary_df = cv_result
                    cv_file = cv_file_path.name
                    # Convert summary to dict for JSON serialization
                    cv_summary_data = cv_summary_df.to_dict('records')
                    tracker.update_stage('cross_validation', 'completed', 100, 
                                       f'Cross-validation complete ({cross_validation_type}, {cross_validation_folds} folds)')
                elif cv_result and not isinstance(cv_result, tuple):
                    # Backward compatibility - old return format
                    cv_file = cv_result.name
                    tracker.update_stage('cross_validation', 'completed', 100, 
                                       f'Cross-validation complete ({cross_validation_type}, {cross_validation_folds} folds)')
            elif modelName not in ['kmeans', 'gmm', 'agglo', 'dbscan', 'birch', 'spectral', 'affinity_propagation', 'bisecting_kmeans', 'hdbscan', 'meanshift', 'minibatch_kmeans', 'optics']:
                cv_result = run_cross_validation(
                    df=df,
                    indicator_names=indicator_names,
                    predictor_names=predictor_names,
                    model=storedModel,
                    scaler=scaler,
                    cv_type=cross_validation_type,
                    cv_folds=cross_validation_folds,
                    useTransformer=useTransformer,
                    transformer_cols=transformer_names,
                    seed=seed,
                    problem_type="regression",
                    y_scaler_type=scaler,
                )
                if cv_result and isinstance(cv_result, tuple) and cv_result[0] is not None:
                    cv_file_path, cv_summary_df = cv_result
                    cv_file = cv_file_path.name
                    # Convert summary to dict for JSON serialization
                    cv_summary_data = cv_summary_df.to_dict('records')
                    tracker.update_stage('cross_validation', 'completed', 100, 
                                       f'Cross-validation complete ({cross_validation_type}, {cross_validation_folds} folds)')
                elif cv_result and not isinstance(cv_result, tuple):
                    # Backward compatibility - old return format
                    cv_file = cv_result.name
                    tracker.update_stage('cross_validation', 'completed', 100, 
                                       f'Cross-validation complete ({cross_validation_type}, {cross_validation_folds} folds)')

    ## Classification results
        if modelName.endswith('_classifier'): 
            
            # Initialize additional_metrics - will be set by models that return it via unpack_classification_result
            if 'additional_metrics' not in locals():
                additional_metrics = None
            
            # Convert to list if needed (handle both lists and numpy arrays)
            indicator_list = indicator_names.tolist() if hasattr(indicator_names, 'tolist') else list(indicator_names)
            predictor_list = predictor_names.tolist() if hasattr(predictor_names, 'tolist') else list(predictor_names)
            
            # {'Adelie Penguin (Pygoscelis adeliae)': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 34.0}, 
            #  'Chinstrap penguin (Pygoscelis antarctica)': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 11.0}, 
            #  'Gentoo penguin (Pygoscelis papua)': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 22.0}, 
            #  'accuracy': 1.0, 'macro avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 67.0}, 
            #  'weighted avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 67.0}}

            result = {
                'accuracy': format(round(report['accuracy'], sigfig), fmt),
                'precision': format(round(report['weighted avg']['precision'], sigfig), fmt),
                'recall': format(round(report['weighted avg']['recall'], sigfig), fmt),
                'f1score': format(round(report['weighted avg']['f1-score'], sigfig), fmt),
                'support': format(round(report['weighted avg']['support'], sigfig), fmt),
                'macro_precision': format(round(report['macro avg']['precision'], sigfig), fmt),
                'macro_recall': format(round(report['macro avg']['recall'], sigfig), fmt),
                'macro_f1score': format(round(report['macro avg']['f1-score'], sigfig), fmt),
                'macro_support': format(round(report['macro avg']['support'], sigfig), fmt),
                'indicators': [str(i) for i in indicator_list],
                'predictors': [str(p) for p in predictor_list],
                'cross_validation_file': cv_file,
                'model_params': _json_safe_params(params),  # Sanitize: get_params() can include estimators
            }

            #write to excel for classifier with all comprehensive metrics
            write_to_excelClassifier(data, indicator_names, predictor_names, stratify_name, scaler, seed, modelName, params, units, report, cm, 
                                   additional_metrics=additional_metrics)

    ## Cluster results
        elif modelName in ['kmeans', 'gmm', 'agglo', 'dbscan', 'birch', 'spectral', 'affinity_propagation', 'bisecting_kmeans', 'hdbscan', 'meanshift', 'minibatch_kmeans', 'optics']:
            result = {
                'train_silhouette' : format(round(train_results['silhouette'],sigfig), fmt),
                'train_calinski_harabasz': format(round(train_results['calinski_harabasz'],sigfig), fmt),
                'train_davies_bouldin' : format(round(train_results['davies_bouldin'],sigfig), fmt),
                'test_silhouette' : format(round(test_results['silhouette'],sigfig), fmt),
                'test_calinski_harabasz': format(round(test_results['calinski_harabasz'],sigfig), fmt),
                'test_davies_bouldin' : format(round(test_results['davies_bouldin'],sigfig), fmt),
                'best_k': best_k,
                'model_params': _json_safe_params(params),  # Sanitize: get_params() can include estimators
            }
            #write to excel for cluster
            write_to_excelCluster(data, indicator_names, stratify_name, scaler, seed, modelName, params, units, train_results['silhouette'], train_results['calinski_harabasz'], train_results['davies_bouldin'], test_results['silhouette'], test_results['calinski_harabasz'], test_results['davies_bouldin'], best_k, centers, silhouette_grid)

    ## Regression results
        else: 
            trainOverall = train_results.iloc[-1]
            testOverall = test_results.iloc[-1]

            #std of RMSE = np.std(all the target rmse) and np.std(all the test rmse)
            train_rmse_values = train_results['RMSE'][:-1]
            test_rmse_values = test_results['RMSE'][:-1]
            train_mae_values = train_results['MAE'][:-1]
            test_mae_values = test_results['MAE'][:-1]

            train_rmse_std = np.std(train_rmse_values) if len(train_rmse_values) > 1 else None
            train_mae_std = np.std(train_mae_values) if len(train_mae_values) > 1 else None
            test_rmse_std = np.std(test_rmse_values) if len(test_rmse_values) > 1 else None
            test_mae_std = np.std(test_mae_values) if len(test_mae_values) > 1 else None

            # Extract sample counts from shapes dictionary (shapes contains tuples like (n_samples, n_features))
            train_n = shapes.get('X_train', (0,))[0] if isinstance(shapes, dict) and 'X_train' in shapes else 0
            test_n = shapes.get('X_test', (0,))[0] if isinstance(shapes, dict) and 'X_test' in shapes else 0
            
            result = {
                'trainscore': format(round(trainOverall['R²'],sigfig), fmt),
                'valscore': format(round(testOverall['R²'],sigfig), fmt),
                'trainrmse': format(round(trainOverall['RMSE'],sigfig), fmt),
                'trainrmsestd': format(round(train_rmse_std, sigfig), fmt) if train_rmse_std is not None else 'N/A',
                'trainmae': format(round(trainOverall['MAE'],sigfig), fmt),
                'trainmaestd': format(round(train_mae_std, sigfig), fmt) if train_mae_std is not None else 'N/A',
                'valrmse': format(round(testOverall['RMSE'],sigfig), fmt),
                'valrmsestd': format(round(test_rmse_std, sigfig), fmt) if test_rmse_std is not None else 'N/A',
                'valmae': format(round(testOverall['MAE'],sigfig), fmt),
                'valmaestd': format(round(test_mae_std, sigfig), fmt) if test_mae_std is not None else 'N/A',
                'train_n': int(train_n) if train_n else 0,
                'test_n': int(test_n) if test_n else 0,
                'indicators': [str(i) for i in (indicator_names.tolist() if hasattr(indicator_names, 'tolist') else list(indicator_names))],
                'predictors': [str(p) for p in (predictor_names.tolist() if hasattr(predictor_names, 'tolist') else list(predictor_names))],
                'cross_validation_file': cv_file,
                'cross_validation_summary': cv_summary_data,  # Add CV summary data
                'model_params': _json_safe_params(params),  # Sanitize: get_params() can include estimators
                'feature_selection_info': feature_selection_info,  # Feature selection details (None if not set by regression pipeline)
                'outlier_info': outlier_info,  # Outlier handling details (None if not set by regression pipeline)
            }
            regression_visuals = []
            
            # Check for baseline graphics (no advanced options)
            baseline_target_plot_exists = (USER_VIS_DIR / "target_plot_1.png").exists()
            baseline_shap_exists = (USER_VIS_DIR / "shap_summary.png").exists()
            
            # Check for advanced graphics (with advanced options)
            advanced_target_plot_exists = (USER_VIS_DIR / "target_plot_1_advanced.png").exists()
            advanced_shap_exists = (USER_VIS_DIR / "shap_summary_advanced.png").exists()
            
            # Determine mode label
            mode_label = 'DiGiTerra Simple Modeling' if modeling_mode == 'simple' else (
                'DiGiTerra Advanced Modeling' if modeling_mode == 'advanced' else 'DiGiTerra AutoML'
            )
            
            # Add baseline graphics if they exist
            if baseline_target_plot_exists:
                regression_visuals.append({'label': f'Predicted vs Actual + Residuals (per target) - {mode_label}', 'file': 'target_plot', 'type': 'baseline'})
            
            # Add advanced graphics if they exist
            if advanced_target_plot_exists:
                regression_visuals.append({'label': f'Predicted vs Actual + Residuals (per target) - {mode_label}', 'file': 'target_plot_advanced', 'type': 'advanced'})
            
            # If neither exists, add default (for backward compatibility)
            if not baseline_target_plot_exists and not advanced_target_plot_exists:
                regression_visuals.append({'label': 'Predicted vs Actual + Residuals (per target)', 'file': 'target_plot', 'type': 'default'})
            
            regression_candidates = [
                ('regression_predicted_vs_actual.png', 'Predicted vs Actual (summary)'),
                ('regression_residuals_hist.png', 'Residuals Histogram'),
                ('regression_residuals_vs_fitted.png', 'Residuals vs Fitted'),
                ('regression_density.png', 'Actual vs Predicted Density'),
                ('regression_feature_importance.png', 'Feature Importance'),
                ('regression_permutation_importance.png', 'Permutation Importance'),
            ]
            
            # Add baseline SHAP if exists
            if baseline_shap_exists:
                regression_visuals.append({'label': f'SHAP Summary - {mode_label}', 'file': 'shap_summary', 'type': 'baseline'})
            
            # Add advanced SHAP if exists
            if advanced_shap_exists:
                regression_visuals.append({'label': f'SHAP Summary - {mode_label}', 'file': 'shap_summary_advanced', 'type': 'advanced'})
            
            # Add other regression candidates (these don't have baseline/advanced variants)
            for filename, label in regression_candidates:
                if (USER_VIS_DIR / filename).exists():
                    regression_visuals.append({'label': label, 'file': filename, 'type': 'default'})
            
            result['regression_visuals'] = regression_visuals
            #should try catch for writing to excel so if it fails the user still sees output and message that excel file failed?
            write_to_excel(data, indicator_names, predictor_names, stratify_name, modelName, params, units, trainOverall, testOverall, train_results, test_results, scaler, seed, shapes, quantileBin_results, cross_validation_summary=cv_summary_data, feature_selection_info=feature_selection_info, outlier_info=outlier_info)

            # Guard against swapped scalers (common cause of wildly wrong predictions)
            try:
                y_train_array = df[predictor_names].to_numpy()
                y_scaler, X_scaler = _maybe_fix_swapped_scalers(y_scaler, X_scaler, y_train_array)
            except Exception:
                pass

            try:
                _y_train = df[predictor_names].to_numpy()
                memStorage['y_train_mean'] = float(np.mean(_y_train))
                memStorage['y_train_std'] = float(np.std(_y_train))
            except Exception:
                memStorage['y_train_mean'] = None
                memStorage['y_train_std'] = None

            memStorage['y_scaler'] = y_scaler


        memStorage['model'] = storedModel #10/1 Rowan this is how the model is stored, it is returned from the train functions
        memStorage['X_scaler'] = X_scaler
        memStorage['feature_order'] = feature_order
        # Store predictor names for use in prediction (handles multi-target regression)
        memStorage['predictor_names'] = predictor_names
        
        # Mark all stages complete
        tracker.complete()
        result['session_id'] = session_id  # Include session ID in response
        
        # Store result for SSE endpoint
        set_result(session_id, result)
        
    except Exception as e:
        # Ensure tracker is cleaned up on error
        logger.error(f"Error in model training: {e}", exc_info=True)
        error_result = {'error': str(e), 'session_id': session_id}
        set_result(session_id, error_result)
        tracker.update_stage('model_training', 'completed', 0, f'Error: {str(e)}')
        remove_tracker(session_id)


@app.route('/process', methods=['POST'])
def process_columns():
    """Start model training process with progress tracking.
    
    Creates a background thread for model training and returns a session ID
    for progress tracking via Server-Sent Events.
    
    Returns:
        JSON: Response with session_id and status on success (HTTP 202 Accepted),
            error message on failure.
    """
    # Validate request
    if not request.json:
        return jsonify({'error': 'No data provided'}), HTTP_BAD_REQUEST
    
    # Create a session ID for progress tracking BEFORE starting training
    session_id = str(uuid.uuid4())
    tracker = get_tracker(session_id)
    
    # getting all the parameters from the front end
    data = request.json
    
    # Return session_id immediately so client can connect to SSE before training completes
    # Training will run in background thread
    training_thread = threading.Thread(
        target=_run_model_training,
        args=(session_id, data),
        daemon=True
    )
    training_thread.start()
    
    # Return immediately with session_id (202 Accepted)
    return jsonify({'session_id': session_id, 'status': 'processing'}), HTTP_ACCEPTED
    
### Section 7: Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    """Generate predictions using a trained model.
    
    Accepts a CSV file with features matching the training data, preprocesses it,
    and generates predictions using the stored model.
    
    Returns:
        JSON: Response with prediction results on success, error message on failure.
    """
    if 'predictFile' not in request.files:
        return jsonify({'error': 'No file uploaded'}), HTTP_BAD_REQUEST

    file = request.files['predictFile']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), HTTP_BAD_REQUEST

    # Security: Validate file extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), HTTP_BAD_REQUEST

    # Security: Sanitize filename to prevent path traversal
    safe_filename = secure_filename(file.filename)
    if not safe_filename:
        return jsonify({'error': 'Invalid filename.'}), HTTP_BAD_REQUEST

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    file.save(filepath)

    # Read the CSV with error handling
    # Note: Modern pandas automatically handles duplicate column names by appending suffixes
    try:
        data = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'The uploaded file is empty.'}), HTTP_BAD_REQUEST
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        return jsonify({'error': f'Error parsing CSV file: {str(e)}'}), HTTP_BAD_REQUEST
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}", exc_info=True)
        return jsonify({'error': f'Error reading file: {str(e)}'}), HTTP_BAD_REQUEST

    if data.empty:
        return jsonify({'error': 'The uploaded file contains no data.'}), HTTP_BAD_REQUEST
    
    # Check for and handle duplicate column names
    if data.columns.duplicated().any():
        logger.warning(f"Duplicate column names detected in prediction file. Renaming duplicates.")
        # Rename duplicate columns by appending .1, .2, etc.
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

    # Preprocess prediction data (use df downstream)
    df = preprocess_data(data)

    try:
        # Ensure a trained model exists
        if 'model' not in memStorage or memStorage['model'] is None:
            return jsonify({'error': 'No trained model found. Train a model first.'}), HTTP_BAD_REQUEST

        # Ensure feature order exists (this is the exact set/order of training features)
        if 'feature_order' not in memStorage or not memStorage['feature_order']:
            return jsonify({'error': 'No feature_order found. Train a model first.'}), HTTP_BAD_REQUEST

        required_features = list(memStorage['feature_order'])

        # Validate required features exist in uploaded prediction file
        missing = sorted(set(required_features) - set(df.columns))
        if missing:
            return jsonify({'error': f"Missing features in prediction file: {missing}"}), HTTP_BAD_REQUEST

        # y_scaler may not exist for classification/cluster models
        y_scaler = memStorage.get('y_scaler', None)
        
        # Get target names if available (for multi-target regression)
        target_names = memStorage.get('predictor_names', None)
        if target_names is not None:
            # Convert to list if it's a pandas Index
            if hasattr(target_names, 'tolist'):
                target_names = target_names.tolist()

        # Run prediction on the PREPROCESSED df, using stored feature order
        prediction(
            df,
            best_model=memStorage['model'],
            training_features=required_features,
            X_scaler=memStorage.get('X_scaler', None),
            y_scaler=y_scaler,
            feature_order=required_features,
            target_names=target_names
        )

        filename = safe_filename[:31]
        if len(safe_filename) > 30:
            filename += "..."

        return jsonify({'results': '/download/predictions.csv', 'filename': filename})

    except Exception as e:
        logger.error(f"Error in predict: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == "__main__":
    run_app(debug=False)
