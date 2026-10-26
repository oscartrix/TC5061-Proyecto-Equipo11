import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Rutas de datos limpios en Parquet y manifiesto de versionado
TRIPS_PARQUET_PATH = PROCESSED_DATA_DIR / 'trips.parquet'
STATIONS_PARQUET_PATH = PROCESSED_DATA_DIR / 'stations.parquet'
P1_FEATURES_PATH = PROCESSED_DATA_DIR / 'p1_features.parquet'
P2_FEATURES_PATH = PROCESSED_DATA_DIR / 'p2_features.parquet'
P3_TRAIN_FEATURES_PATH = PROCESSED_DATA_DIR / 'p3_train_features.parquet'
P3_HOLDOUT_FEATURES_PATH = PROCESSED_DATA_DIR / 'p3_holdout_features.parquet'

MANIFEST_PATH = PROJ_ROOT / 'config' / 'data_manifest.json'

# Rango de fechas de estudio y cortes de partición temporal (sin fuga del futuro)
START_MONTH = '2024-01'
END_MONTH = '2026-09'

TRAIN_END_DATE = '2025-12-31'
VAL_START_DATE = '2026-01-01'
VAL_END_DATE = '2026-06-30'
TEST_START_DATE = '2026-07-01'

# Parámetros globales de reproducibilidad y MLflow
RANDOM_SEED = 42
MLFLOW_EXPERIMENT_NAME = 'mibici-equipo11'
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
MLFLOW_BACKEND_STORE_URI = f'sqlite:///{PROJ_ROOT / "mlflow.db"}'
TEAM_NAME = 'equipo11'

# Modelos disponibles por pregunta y sus hiperparámetros por default
MODEL_DEFAULTS = {
    'p1': {
        'baseline': {},
        'ridge': {'alpha': 1.0},
        'rf': {'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 20},
        'hgb': {'max_iter': 200, 'learning_rate': 0.08, 'max_depth': 6},
    },
    'p2': {
        'baseline': {},
        'hgb': {'max_iter': 150, 'learning_rate': 0.08, 'max_depth': 6},
        'rf': {'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 20},
        'logreg': {'C': 1.0},
    },
    'p3': {
        'baseline': {'n_clusters': 4},
        'kmeans_usage': {'n_clusters': 4},
        'gmm_usage': {'n_clusters': 4, 'covariance_type': 'full'},
        'kmeans_pca': {'n_clusters': 4, 'n_components': 3},
    },
}

# Combinaciones de hiperparámetros para búsquedas con MLflow.
SEARCH_GRIDS = {
    'p1': {
        'ridge': {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]},
        'rf': {'max_depth': [8, 12, 16], 'min_samples_leaf': [10, 40]},
        'hgb': {'max_iter': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [4, 8]},
    },
    'p2': {
        'hgb': {
            'max_iter': [100, 150, 200],
            'learning_rate': [0.05, 0.08],
            'max_depth': [4, 6],
        },
        'logreg': {'C': [0.01, 0.1, 1.0, 10.0]},
    },
    'p3': {
        'kmeans_usage': {'n_clusters': [3, 4, 5, 6]},
        'gmm_usage': {'n_clusters': [3, 4, 5, 6], 'covariance_type': ['full', 'diag']},
        'kmeans_pca': {'n_clusters': [3, 4, 5, 6], 'n_components': [2, 3, 5]},
    },
}

# Métrica de validación para elegir candidatos y su dirección (min o max)
SELECTION_METRICS = {
    'p1': ('val_mae', 'min'),
    'p2': ('val_f1_macro', 'max'),
    'p3': ('train_silhouette_usage', 'max'),
}

# Estabilidad mínima (ARI contra 2026) para que un clustering de P3 pueda ser champion
P3_MIN_STABILITY = 0.4

# Nombres en el Model Registry y qué predice cada modelo
REGISTERED_MODELS = {
    'p1': ('mibici-equipo11-p1-demanda', 'Viajes de salida por estación al día siguiente (regresión)'),
    'p2': ('mibici-equipo11-p2-riesgo', 'Riesgo de déficit, balance o excedente de bicis en las próximas 3 horas (clasificación)'),
    'p3': ('mibici-equipo11-p3-grupos', 'Grupos de estaciones por perfil de uso (clustering)'),
}

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
