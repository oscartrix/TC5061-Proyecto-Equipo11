# Python 3.10 slim — versión fijada en pyproject.toml y python_env.yaml
FROM python:3.10-slim

LABEL maintainer="TC5061-Equipo11"
LABEL description="MLOps MiBici: ingesta, features y entrenamiento con MLflow"

# No .pyc, logs sin buffer, pip silencioso
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    APP_HOME=/app

# git es requerido por get_git_commit() en train.py
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}

# Estructura de directorios esperada por config.py
RUN mkdir -p data/raw data/interim data/processed data/external \
             models reports/figures config

# Dependencias primero → capa cacheada mientras no cambie requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Código fuente e instalación del paquete mi_bici
COPY pyproject.toml mi_bici/ mi_bici/
COPY config/ config/
RUN pip install -e .

# Archivos de proyecto opcionales (MLproject, Makefile, etc.)
COPY Makefile MLproject python_env.yaml .

# MLflow tracking server
EXPOSE 5000

# Montar en docker run: -v $(pwd)/data:/app/data  -v $(pwd)/mlruns:/app/mlruns
VOLUME ["/app/data", "/app/models", "/app/mlruns", "/app/reports"]

# CMD por defecto: servidor MLflow
# Sobreescribir para otros usos, ej:
#   docker run ... python -m mi_bici.dataset
#   docker run ... python -m mi_bici.modeling.train --question p1 --model ridge
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/app/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
