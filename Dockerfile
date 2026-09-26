FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

USER appuser
EXPOSE 8000
CMD ["uvicorn", "fraud_detection.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
