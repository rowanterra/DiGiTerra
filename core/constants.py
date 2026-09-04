"""Shared constants for DiGiTerra Flask app and blueprints."""

import os

# File upload
ALLOWED_EXTENSIONS = {"csv"}
FILE_SIZE_WARNING_MB = float(os.environ.get("DIGITERRA_WARN_UPLOAD_MB", "1"))
MAX_UPLOAD_MB = float(os.environ.get("DIGITERRA_MAX_UPLOAD_MB", "0"))
ENFORCE_UPLOAD_LIMIT = os.environ.get("DIGITERRA_ENFORCE_UPLOAD_LIMIT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CELL_COUNT_WARNING_LARGE = 1_000_000
CELL_COUNT_WARNING_MODERATE = 500_000
# A parsed frame is kept in process memory for the session, and a DataFrame can
# take several times the bytes of its CSV. Three limits apply in order: the
# request body size, the number of cells the reader is allowed to parse, and the
# measured size of the parsed frame. 0 disables a limit. desktop_app.py disables
# the cell and frame limits and raises the body limit, since a local user's own
# machine is not an attack surface.
MAX_CELLS = int(os.environ.get("DIGITERRA_MAX_CELLS", "5000000") or 0)
MAX_FRAME_MB = float(os.environ.get("DIGITERRA_MAX_FRAME_MB", "256") or 0)
# Request bodies above this size are rejected before they are read.
DEFAULT_MAX_UPLOAD_MB = float(os.environ.get("DIGITERRA_DEFAULT_MAX_UPLOAD_MB", "32") or 32)
BYTES_PER_MB = 1024 * 1024

# Display
COLUMN_NAME_DISPLAY_LENGTH = 11
FILENAME_DISPLAY_LENGTH = 31

# Model training
DEFAULT_CV_FOLDS = 5
DEFAULT_SEARCH_ITERATIONS = 50
RANDOM_SEED_MAX = 1000
# Server-side bounds on request values that schedule work. The search and CV paths
# fan out over every core, so an unbounded fold count or iteration count from one
# request would pin the host. The UI inputs carry the same ranges.
MIN_TEST_SIZE = float(os.environ.get("DIGITERRA_MIN_TEST_SIZE", "0.05"))
MAX_TEST_SIZE = float(os.environ.get("DIGITERRA_MAX_TEST_SIZE", "0.90"))
MAX_CV_FOLDS = int(os.environ.get("DIGITERRA_MAX_CV_FOLDS", "20") or 20)
MAX_SEARCH_CV_FOLDS = int(os.environ.get("DIGITERRA_MAX_SEARCH_CV_FOLDS", "10") or 10)
MAX_SEARCH_ITERATIONS = int(os.environ.get("DIGITERRA_MAX_SEARCH_ITERATIONS", "100") or 100)

# Progress (SSE)
# An open stream occupies a worker thread. Training that runs longer than this
# still finishes; the client reconnects to the same id and receives the result.
PROGRESS_STREAM_TIMEOUT_SECONDS = float(os.environ.get("DIGITERRA_PROGRESS_TIMEOUT_SECONDS", "900") or 900)
PROGRESS_UPDATE_INTERVAL_SECONDS = 0.5
PROGRESS_FINAL_UPDATE_DELAY_SECONDS = 0.5
PROGRESS_COMPLETE_THRESHOLD = 100

# HTTP
HTTP_OK = 200
HTTP_ACCEPTED = 202
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429
HTTP_INTERNAL_SERVER_ERROR = 500
