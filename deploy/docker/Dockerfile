FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DIGITERRA_HOST=0.0.0.0 \
    DIGITERRA_PORT=5000

RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN chown -R app:app /app
USER app

EXPOSE 5000

# Use gunicorn for production WSGI server
# IMPORTANT: Using 1 worker because app uses global memStorage dict for state.
# Each worker process has separate memory, so multi-worker breaks data persistence.
# For proper multi-user/multi-worker support, refactor to use session-based storage
# or shared storage (Redis, database, file system).
# Bind to 0.0.0.0 to accept connections from outside container
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
