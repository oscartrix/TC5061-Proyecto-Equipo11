# API de predicción de MiBici (P1). Funciona igual con docker build y podman build.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/p1_champion

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY mi_bici/ mi_bici/
# Modelo champion descargado antes con scripts/export_model.py
COPY models/p1_champion/ models/p1_champion/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "mi_bici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
