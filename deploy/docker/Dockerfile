FROM python:3.11-slim

# Writable XDG dirs for non-root user (defaults like /nonexistent break startup when libs try to create cache).
ARG WORK_DIR=/app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DIGITERRA_HOST=0.0.0.0 \
    DIGITERRA_PORT=5000 \
    URL_PREFIX=digiterra \
    XDG_DATA_HOME=${WORK_DIR} \
    XDG_CACHE_HOME=${WORK_DIR}

RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR ${WORK_DIR}

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R app:app .
USER app

EXPOSE 5000

# Use gunicorn for production WSGI server
# IMPORTANT: Using 1 worker because app uses global memStorage dict for state.
# Each worker process has separate memory, so multi-worker breaks data persistence.
# For proper multi-user/multi-worker support, refactor to use session-based storage
# or shared storage (Redis, database, file system).
# Bind to 0.0.0.0 to accept connections from outside container
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
