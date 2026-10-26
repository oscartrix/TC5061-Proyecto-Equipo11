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
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}

# Estructura de directorios esperada por config.py
RUN mkdir -p data/raw data/interim data/processed data/external \
             models reports/figures config db mlartifacts

# Dependencias primero → capa cacheada mientras no cambie requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Código fuente e instalación del paquete mi_bici
COPY pyproject.toml README.md ./
COPY mi_bici/ mi_bici/
COPY config/ config/
RUN pip install -e . --no-deps

# Archivos de proyecto opcionales (MLproject, Makefile, etc.)
COPY Makefile MLproject python_env.yaml ./

# El historial permite que get_git_commit() registre el commit de la imagen
COPY .git/ .git/

# MLflow tracking server
EXPOSE 5000

# Montar en docker run: -v $(pwd)/data:/app/data  -v mlflow-db:/app/db  -v mlflow-art:/app/mlartifacts
VOLUME ["/app/data", "/app/models", "/app/db", "/app/mlartifacts", "/app/reports"]

# CMD por defecto: servidor MLflow
# Sobreescribir para otros usos, ej:
#   docker run ... python -m mi_bici.dataset
#   docker run ... python -m mi_bici.modeling.train --question p1 --model ridge
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////app/db/mlflow.db", \
     "--artifacts-destination", "/app/mlartifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
