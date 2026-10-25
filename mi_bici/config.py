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

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
