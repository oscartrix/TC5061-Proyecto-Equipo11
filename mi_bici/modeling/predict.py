"""Inferencia con los modelos champion registrados para P1, P2 y P3.

Uso:
    python -m mi_bici.modeling.predict --question p1 \
        --features-path data/processed/p1_features.parquet \
        --predictions-path data/processed/p1_predictions.parquet
"""

from pathlib import Path

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import typer

from mi_bici.config import MLFLOW_TRACKING_URI, REGISTERED_MODELS

app = typer.Typer()


def predict_features(model, features: pd.DataFrame) -> np.ndarray:
    """Selecciona las entradas por firma y conserva la alineación de las predicciones."""
    schema = model.metadata.get_input_schema()
    if schema is None or not all(isinstance(name, str) for name in schema.input_names()):
        raise ValueError(
            'El modelo registrado debe tener una firma de entrada con nombres de columnas.'
        )
    missing_columns = [
        name for name in schema.required_input_names() if name not in features.columns
    ]
    if missing_columns:
        raise ValueError(f'Faltan columnas requeridas por el modelo: {", ".join(missing_columns)}')
    columns = [name for name in schema.input_names() if name in features.columns]
    predictions = np.asarray(model.predict(features.loc[:, columns]))
    if predictions.ndim != 1 or len(predictions) != len(features):
        raise ValueError('El modelo debe devolver una predicción por fila de entrada.')
    return predictions


def predict_dataset(question: str, features_path: Path, predictions_path: Path) -> pd.DataFrame:
    """Valida la entrada contra la firma del champion y guarda una predicción por fila."""
    if question not in REGISTERED_MODELS:
        raise ValueError(f'La pregunta debe ser una de: {", ".join(REGISTERED_MODELS)}')
    for path in (features_path, predictions_path):
        if path.suffix.lower() not in {'.csv', '.parquet'}:
            raise ValueError(f'Formato no soportado: {path}. Usa CSV o Parquet.')
    if features_path.resolve() == predictions_path.resolve():
        raise ValueError('La salida debe ser distinta del archivo de entrada.')
    if not features_path.is_file():
        raise ValueError(f'Archivo de características no encontrado: {features_path}')

    features = (
        pd.read_parquet(features_path)
        if features_path.suffix.lower() == '.parquet'
        else pd.read_csv(features_path)
    )
    if features.empty:
        raise ValueError('El archivo de características no puede estar vacío.')
    if 'prediction' in features.columns:
        raise ValueError(
            'La entrada ya contiene una columna prediction. Usa las características originales.'
        )

    model_uri = f'models:/{REGISTERED_MODELS[question][0]}@champion'
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f'Cargando modelo {model_uri}')
    model = mlflow.pyfunc.load_model(model_uri)
    # MLflow valida los tipos, se conservan identificadores y metadatos en la salida.
    predictions = predict_features(model, features)
    output = features.assign(prediction=predictions)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.suffix.lower() == '.parquet':
        output.to_parquet(predictions_path, index=False)
    else:
        output.to_csv(predictions_path, index=False)
    logger.success(f'{len(output):,} predicciones guardadas en {predictions_path}')
    return output


@app.command()
def main(
    question: str = typer.Option(..., help='Pregunta: p1, p2 o p3'),
    features_path: Path = typer.Option(..., help='Características de entrada en CSV o Parquet'),
    predictions_path: Path = typer.Option(..., help='Archivo de salida en CSV o Parquet'),
):
    """Carga el champion de la pregunta y guarda las predicciones."""
    try:
        predict_dataset(question, features_path, predictions_path)
    except (ValueError, OSError, mlflow.exceptions.MlflowException) as error:
        raise typer.BadParameter(str(error)) from error


if __name__ == "__main__":
    app()
