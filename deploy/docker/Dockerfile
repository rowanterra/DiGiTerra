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

# One worker process, because session state and trained models are in process
# memory; a second process would not see them. Threads within that process let a
# held-open progress stream coexist with other requests, which the default
# synchronous worker does not allow. --timeout is the worker-silence timeout and
# does not apply to a thread serving a request, so a long training run is safe.
# Bind to 0.0.0.0 to accept connections from outside the container.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--worker-class", "gthread", "--threads", "8", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
