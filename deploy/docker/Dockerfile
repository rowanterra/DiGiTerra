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

CMD ["python", "app.py"]
