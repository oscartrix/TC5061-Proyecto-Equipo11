"""Carga desde el Model Registry del modelo que sirve la API."""

from dataclasses import dataclass
import os

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
import numpy as np
import pandas as pd

from mi_bici.config import MLFLOW_TRACKING_URI, REGISTERED_MODELS
from mi_bici.modeling.predict import predict_features

DEFAULT_MODEL_NAME = REGISTERED_MODELS['p1'][0]
DEFAULT_MODEL_ALIAS = 'champion'


class ModelLoadError(RuntimeError):
    """No se pudo cargar el modelo configurado."""


@dataclass(frozen=True)
class ModelSettings:
    """Modelo a servir."""

    name: str = DEFAULT_MODEL_NAME
    alias: str = DEFAULT_MODEL_ALIAS
    version: str | None = None

    @classmethod
    def from_env(cls) -> 'ModelSettings':
        return cls(
            name=os.getenv('MODEL_NAME') or DEFAULT_MODEL_NAME,
            alias=os.getenv('MODEL_ALIAS') or DEFAULT_MODEL_ALIAS,
            version=os.getenv('MODEL_VERSION') or None,
        )

    @property
    def reference(self) -> str:
        return f'{self.name}/{self.version}' if self.version else f'{self.name}@{self.alias}'


@dataclass(frozen=True)
class ServedModel:
    """Modelo cargado y la versión que le corresponde."""

    name: str
    version: str
    run_id: str | None
    model: PyFuncModel

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return predict_features(self.model, align_features(features, self.model))


def align_features(features: pd.DataFrame, model: PyFuncModel) -> pd.DataFrame:
    """Cambia los tipos de datos a los que espera el modelo."""
    schema = model.metadata.get_input_schema()
    dtypes = {
        column.name: column.type.to_pandas()
        for column in schema.inputs
        if column.name in features.columns
    }
    return features.astype(dtypes)


def load_served_model(settings: ModelSettings | None = None) -> ServedModel:
    settings = settings or ModelSettings.from_env()
    client = MlflowClient(MLFLOW_TRACKING_URI)
    try:
        if settings.version:
            version = client.get_model_version(settings.name, settings.version)
        else:
            version = client.get_model_version_by_alias(settings.name, settings.alias)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.pyfunc.load_model(f'models:/{settings.name}/{version.version}')
    except (MlflowException, OSError) as error:
        raise ModelLoadError(
            f'No se pudo cargar {settings.reference} desde {MLFLOW_TRACKING_URI}: {error}'
        ) from error

    schema = model.metadata.get_input_schema()
    if schema is None or not schema.has_input_names():
        raise ModelLoadError(f'{settings.reference} no tiene una firma con nombres de columnas.')

    logger.info(f'Modelo cargado: {settings.name} v{version.version}')
    return ServedModel(settings.name, str(version.version), version.run_id, model)
