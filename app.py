### Section 1: Importing packages and models
#
# Handoff note: This file holds all Flask routes, upload/prediction logic, and model
# orchestration. Sections are marked with "### Section N". Use them to jump around.
# Global in-memory state. Scoped per session to avoid cross-user leakage.
# Replace with session/DB if you go multi-user web. See HANDOFF.md for more.

from flask import Flask, request, render_template, jsonify, send_from_directory, send_file, Response, stream_with_context, g
import uuid
from werkzeug.utils import secure_filename
import os
import ast
import json
import threading
import logging
from pathlib import Path
import pandas as pd
import random
import numpy as np

from python_scripts import config

BASE_DIR = config.BASE_DIR
APP_SUPPORT_DIR = config.APP_SUPPORT_DIR
UPLOAD_DIR = config.UPLOAD_DIR
LOG_DIR = config.LOG_DIR
URL_PREFIX = config.URL_PREFIX

# Single source of truth for user visualizations: config.VIS_DIR (set below and used everywhere)
_user_vis_path = Path(os.environ.get("DIGITERRA_OUTPUT_DIR", str(APP_SUPPORT_DIR / "user_visualizations")))
os.environ.setdefault("DIGITERRA_OUTPUT_DIR", str(_user_vis_path))
_user_vis_path.mkdir(parents=True, exist_ok=True)
config.update_vis_dir(_user_vis_path)


def _configure_logging():
    """Configure logging at app startup: level, format, optional file under LOG_DIR."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S", force=True)
    root = logging.getLogger()
    # Optional file handler so production logs are persisted
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        log_file = LOG_DIR / f"digiterra_{datetime.now().strftime('%Y-%m-%d')}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh)
    except OSError:
        pass  # e.g. read-only filesystem or no permission
    return logging.getLogger(__name__)


logger = _configure_logging()

from python_scripts.helpers import (
    preprocess_data,
    prediction,
    run_cross_validation,
    unpack_classification_result,
    write_to_excel,
    write_to_excelClassifier,
    write_to_excelCluster,
    write_to_excelRegression,
)
from python_scripts.app_model_training import run_model_training
from python_scripts.app_exploration import (
    handle_auto_detect_transformers,
    handle_auto_detect_nan_zeros,
    handle_corr,
    handle_pairplot,
)
from python_scripts.app_prediction import run_predict

def _ensure_user_vis_dir():
    """Ensure the user visualizations directory exists and return it (config.VIS_DIR)."""
    config.VIS_DIR.mkdir(parents=True, exist_ok=True)
    return config.VIS_DIR


def _with_prefix(path: str) -> str:
    """Return an absolute URL path with configured URL_PREFIX applied."""
    normalized_path = path if path.startswith("/") else f"/{path}"
    if not URL_PREFIX:
        return normalized_path
    return f"{URL_PREFIX}{normalized_path}"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
## create upload folder for the csv files the user uploads
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SESSION_STORE = {}

# Session cookie name for per-user state
SESSION_COOKIE_NAME = "digiterra_session"


def route_with_prefix(rule: str, **options):
    """Register each route at both root and prefixed paths."""
    def decorator(func):
        endpoint = options.pop("endpoint", None)
        app.route(rule, endpoint=endpoint, **options)(func)
        if URL_PREFIX:
            prefixed_rule = _with_prefix(rule)
            prefixed_endpoint = f"{endpoint or func.__name__}_prefixed"
            app.route(prefixed_rule, endpoint=prefixed_endpoint, **options)(func)
        return func
    return decorator


def _get_session_id() -> str:
    """Return the current session ID (from cookie or request context)."""
    session_id = getattr(g, "session_id", None)
    if session_id:
        return session_id
    cookie_session = request.cookies.get(SESSION_COOKIE_NAME)
    return cookie_session or "default"


def _get_session_storage(session_id: str | None = None) -> dict:
    """Get or create session-scoped storage dict."""
    sid = session_id or _get_session_id()
    return SESSION_STORE.setdefault(sid, {})


@app.before_request
def _ensure_session_cookie() -> None:
    """Ensure every client has a session cookie for isolated in-memory storage."""
    existing = request.cookies.get(SESSION_COOKIE_NAME)
    if existing:
        g.session_id = existing
        g.new_session_id = None
        return
    new_id = uuid.uuid4().hex
    g.session_id = new_id
    g.new_session_id = new_id


@app.after_request
def _set_session_cookie(response):
    """Set a session cookie if one was created in this request."""
    new_id = getattr(g, "new_session_id", None)
    if new_id:
        response.set_cookie(SESSION_COOKIE_NAME, new_id, httponly=True, samesite="Lax")
    return response

# File upload validation
ALLOWED_EXTENSIONS = {'csv'}

# File size and cell count thresholds
FILE_SIZE_WARNING_MB = float(os.environ.get("DIGITERRA_WARN_UPLOAD_MB", "1"))
MAX_UPLOAD_MB = float(os.environ.get("DIGITERRA_MAX_UPLOAD_MB", "0"))
ENFORCE_UPLOAD_LIMIT = os.environ.get("DIGITERRA_ENFORCE_UPLOAD_LIMIT", "false").strip().lower() in {"1", "true", "yes", "on"}
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

# Display length constants
COLUMN_NAME_DISPLAY_LENGTH = 11  # Max characters to display for column names
FILENAME_DISPLAY_LENGTH = 31  # Max characters to display for filenames

# File size conversion
BYTES_PER_MB = 1024 * 1024

# Seed generation
RANDOM_SEED_MAX = 1000

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _normalize_predict_preprocess_mode(mode: str) -> str:
    """Map training-time preprocessing modes to prediction-safe modes."""
    if mode in {"target", "indicatorAndTarget"}:
        return "indicator"
    return mode


def _parse_class_weight(value):
    """Normalize class_weight input from UI text field."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        if stripped == "balanced":
            return "balanced"
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return value

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

### Section 2: Render the main html page
@route_with_prefix('/')
def index():
    """Render the main HTML page.
    
    Returns:
        HTML template: The main index.html page.
    """
    return render_template(
        'index.html',
        api_root=URL_PREFIX,
        static_root=_with_prefix('/static'),
    )


if URL_PREFIX:
    @app.route(f"{URL_PREFIX}/static/<path:filename>")
    def prefixed_static(filename):
        """Serve static assets at prefixed path (e.g. /digiterra/static/*)."""
        return send_from_directory(app.static_folder, filename)


@route_with_prefix('/progress/<session_id>')
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


@route_with_prefix('/user-visualizations/<path:filename>')
def user_visualizations(filename):
    """Serve user-generated visualization files.
    
    Args:
        filename (str): Name of the visualization file to serve.
    
    Returns:
        Response: File download response or 404 if not found.
    """
    vis_dir = _ensure_user_vis_dir()
    response = send_from_directory(vis_dir, filename)
    if response.status_code == 200:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


@route_with_prefix('/download/<path:filename>')
def download_visualization(filename):
    """Serve a file from the user visualizations directory for download. Path traversal is blocked."""
    vis_dir = _ensure_user_vis_dir()
    file_path = (vis_dir / filename).resolve()
    base_resolved = vis_dir.resolve()
    try:
        file_path.relative_to(base_resolved)
    except ValueError:
        return jsonify({'error': 'File not found.'}), HTTP_NOT_FOUND
    if not file_path.is_file():
        return jsonify({'error': 'File not found.'}), HTTP_NOT_FOUND
    requested_name = request.args.get("download_name", filename)
    download_name = secure_filename(requested_name) or filename
    return send_file(file_path, as_attachment=True, download_name=download_name)

@route_with_prefix('/downloadAdditionalInfo', methods=['POST'])
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
        file_path = _ensure_user_vis_dir() / filename
        
        # Write to Excel
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return jsonify({'filename': filename})
    except Exception as e:
        logger.error(f'Error generating additional info Excel: {str(e)}')
        return jsonify({'error': str(e)}), HTTP_INTERNAL_SERVER_ERROR

### Section 3: upload route  
    ## creates the route to handle file upload and gets the column names to send to front end
@route_with_prefix('/upload', methods=['POST'])
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
    store = _get_session_storage()
    
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
        unique_prefix = uuid.uuid4().hex[:8]
        stored_filename = f"{unique_prefix}_{safe_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)
        file.save(filepath)

        # Check file size warning and optional hard limit for deployed environments
        file_size_mb = os.path.getsize(filepath) / BYTES_PER_MB
        if ENFORCE_UPLOAD_LIMIT and MAX_UPLOAD_MB > 0 and file_size_mb > MAX_UPLOAD_MB:
            try:
                os.remove(filepath)
            except OSError:
                logger.warning("Failed to remove oversized uploaded file: %s", filepath)
            return jsonify({
                'error': f'File exceeds maximum allowed size of {MAX_UPLOAD_MB:.1f} MB.'
            }), HTTP_BAD_REQUEST

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
        
        store['data'] = data   # store in session storage instead of calling it 5 times
        
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
@route_with_prefix('/auto-detect-transformers', methods=['POST'])
def auto_detect_transformers():
    """Auto-detect categorical columns from selected indicators."""
    store = _get_session_storage()
    data = request.json or {}
    result, status = handle_auto_detect_transformers(store, data)
    return jsonify(result), status


@route_with_prefix('/auto-detect-nan-zeros', methods=['POST'])
def auto_detect_nan_zeros():
    """Auto-detect NaN or zeros in indicator and target columns."""
    store = _get_session_storage()
    data = request.json or {}
    result, status = handle_auto_detect_nan_zeros(store, data)
    return jsonify(result), status


@route_with_prefix('/correlationMatrices', methods=['POST'])
def corr():
    """Generate correlation matrices for numeric columns."""
    store = _get_session_storage()
    data = request.json or {}
    result, status = handle_corr(store, data, _ensure_user_vis_dir(), _with_prefix)
    return jsonify(result), status


@route_with_prefix('/pairplot', methods=['POST'])
def pairplot():
    """Generate pairplot visualization for two numeric columns."""
    store = _get_session_storage()
    data = request.json or {}
    result, status = handle_pairplot(store, data, _ensure_user_vis_dir(), _with_prefix)
    return jsonify(result), status


### Section 5: Route for preprocessing
    ## when user clicks 'Process' this sends column names selected for the predictors + indicators to display 
@route_with_prefix('/preprocess', methods=['POST'])
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
    
    store = _get_session_storage()
    if 'data' not in store:
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
        df = store['data']
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

    # Warn if stratifying by target (leaks target into split). Skip for Cluster (no supervised target).
    output_type = (data.get('outputType') or '').strip()
    if output_type != 'Cluster':
        strat_idx = stratify if isinstance(stratify, (int, np.integer)) else None
        if strat_idx is not None and selected_predictors is not None:
            pred_list = selected_predictors if isinstance(selected_predictors, (list, tuple)) else [selected_predictors]
            if strat_idx in pred_list:
                return jsonify({
                    'error': 'Do not stratify by your target column. That would leak target information into the train/test split. Choose a different column to stratify by.'
                }), HTTP_BAD_REQUEST

    store['last_preprocess_request'] = {
        'indicators': selected_indicators,
        'predictors': selected_predictors,
        'stratify': stratify,
    }

    return jsonify({
        'predictors': [str(p)[:COLUMN_NAME_DISPLAY_LENGTH] for p in predictor_names],
        'indicators': [str(i)[:COLUMN_NAME_DISPLAY_LENGTH] for i in indicator_names],
        'stratify': stratify_name[:COLUMN_NAME_DISPLAY_LENGTH] if stratify_name else ''
    })


### Section 6: Route for running model
    ## when user clicks 'Run my Model' this runs the model with the selected parameters and returns the performance results

def _run_model_training(session_id: str, data: dict, storage_session_id: str):
    """Run model training in a background thread. Delegates to app_model_training.run_model_training."""
    return run_model_training(session_id, data, storage_session_id, get_storage=_get_session_storage)



@route_with_prefix('/process', methods=['POST'])
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
    
    # Enforce expected order: user should run preprocess before modeling.
    store = _get_session_storage()
    last_preprocess = store.get('last_preprocess_request')
    if not last_preprocess:
        return jsonify({'error': 'Run Model Preprocessing before Modeling.'}), HTTP_BAD_REQUEST

    req_indicators = request.json.get('indicators')
    req_predictors = request.json.get('predictors')
    if req_indicators != last_preprocess.get('indicators') or req_predictors != last_preprocess.get('predictors'):
        return jsonify({'error': 'Selections changed since preprocessing. Re-run Model Preprocessing before Modeling.'}), HTTP_BAD_REQUEST

    # Create a session ID for progress tracking BEFORE starting training
    session_id = str(uuid.uuid4())
    storage_session_id = _get_session_id()
    tracker = get_tracker(session_id)
    
    # getting all the parameters from the front end
    data = request.json
    
    # Return session_id immediately so client can connect to SSE before training completes
    # Training will run in background thread
    training_thread = threading.Thread(
        target=_run_model_training,
        args=(session_id, data, storage_session_id),
        daemon=True
    )
    training_thread.start()
    
    # Return immediately with session_id (202 Accepted)
    return jsonify({'session_id': session_id, 'status': 'processing'}), HTTP_ACCEPTED
    
### Section 7: Route for Prediction
@route_with_prefix('/predict', methods=['POST'])
def predict():
    """Generate predictions using a trained model."""
    store = _get_session_storage()
    result, status = run_predict(
        store,
        request,
        app.config['UPLOAD_FOLDER'],
        _ensure_user_vis_dir(),
        _with_prefix,
        allowed_file,
        _normalize_predict_preprocess_mode,
    )
    return jsonify(result), status

if __name__ == "__main__":
    run_app(debug=False)
