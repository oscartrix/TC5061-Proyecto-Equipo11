"""Evaluación de un champion fijo contra el baseline, sin cambiar la selección con test."""

import json
from numbers import Real
import os
from pathlib import Path

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mi_bici.config import (
    MANIFEST_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_DEFAULTS,
    P1_FEATURES_PATH, P2_FEATURES_PATH, P3_TRAIN_FEATURES_PATH, P3_HOLDOUT_FEATURES_PATH,
    RANDOM_SEED, REGISTERED_MODELS, REPORTS_DIR,
)
from mi_bici.modeling.evaluation import (
    classification_reports, clustering_reports, regression_reports,
)
from mi_bici.modeling.predict import predict_features
from mi_bici.modeling.train import P3_USAGE_COLS
from mi_bici.validation.contracts import validate_feature_contract

app = typer.Typer()
OUTPUT_DIR = REPORTS_DIR / 'evaluation'


def read_features(name, path):
    if not path.is_file():
        raise ValueError(f'Archivo de características no encontrado: {path}')
    frame = pd.read_parquet(path)
    errors = validate_feature_contract(name, frame)
    if errors:
        raise ValueError('Datos no aceptables para evaluar:\n' + '\n'.join(errors))
    return frame


def summary_metrics(summary):
    """Excluye métricas indefinidas del tracking y las conserva como vacías en CSV."""
    metrics = {}
    for row in summary.to_dict('records'):
        prefix = '_'.join(str(row[key]) for key in ('model', 'split', 'cohort') if key in row)
        for key, value in row.items():
            if isinstance(value, Real) and np.isfinite(value):
                metrics[f'evaluation_{prefix}_{key}'] = float(value)
    return metrics


def evaluate_question(question: str, output_dir: Path = OUTPUT_DIR):
    if question not in REGISTERED_MODELS:
        raise ValueError(f'La pregunta debe ser una de: {", ".join(REGISTERED_MODELS)}')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    name = REGISTERED_MODELS[question][0]
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(name, 'champion')
    # Fijar la versión evita que un cambio concurrente del alias altere esta evaluación.
    model_uri = f'models:/{name}/{version.version}'
    model = mlflow.pyfunc.load_model(model_uri)
    if question in {'p1', 'p2'}:
        frame = read_features(
            question.upper(), P1_FEATURES_PATH if question == 'p1' else P2_FEATURES_PATH
        )
        frame = frame.loc[frame['split'].isin(['val', 'test'])].copy()
        if set(frame['split']) != {'val', 'test'}:
            raise ValueError('Se requieren observaciones de validación y prueba para evaluar.')
        predictions = predict_features(model, frame)
        reports = (regression_reports if question == 'p1' else classification_reports)(
            frame, predictions
        )
        baseline = 'SeasonalNaive_lag7d' if question == 'p1' else 'WeeklyRule_lag168h'
        baseline_params = {}
    else:
        train = read_features('P3_train', P3_TRAIN_FEATURES_PATH)
        holdout = read_features('P3_holdout', P3_HOLDOUT_FEATURES_PATH)
        baseline_params = MODEL_DEFAULTS['p3']['baseline'].copy()
        geographic = make_pipeline(StandardScaler(), KMeans(
            **baseline_params, random_state=RANDOM_SEED, n_init=10
        ))
        geographic.fit(train[['latitude', 'longitude']])
        reports = clustering_reports(
            train, holdout, predict_features(model, train), predict_features(model, holdout),
            geographic.predict(train[['latitude', 'longitude']]),
            geographic.predict(holdout[['latitude', 'longitude']]), P3_USAGE_COLS,
        )
        baseline = 'KMeans_Geographic'

    metadata = {
        'question': question, 'model_uri': model_uri, 'alias': 'champion',
        'source_run_id': version.run_id, 'model_tags': version.tags,
        'baseline': baseline, 'baseline_params': baseline_params, 'random_seed': RANDOM_SEED,
        'data_version': json.loads(MANIFEST_PATH.read_text()).get('data_version', 'unknown')
        if MANIFEST_PATH.is_file() else 'unknown',
        'evaluation_changes_selection': False,
    }
    if question == 'p3':
        metadata['stability_definition'] = 'Predicciones del mismo modelo en estaciones comunes.'
        metadata['stability_limit'] = (
            'La estabilidad geográfica es trivial si las coordenadas no cambian. '
            'El holdout ya interviene en el piso de estabilidad del registro de P3.'
        )
    directory = output_dir / question
    if not os.environ.get('MLFLOW_RUN_ID'):
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(
        run_id=os.environ.get('MLFLOW_RUN_ID'), run_name=f'{question}-evaluation'
    ):
        mlflow.set_tags({
            'question': question, 'stage_of_work': 'evaluation', 'source_run_id': version.run_id,
        })
        mlflow.log_params({'evaluated_model_uri': model_uri, 'baseline': baseline})
        mlflow.log_metrics(summary_metrics(reports['summary']))
        directory.mkdir(parents=True, exist_ok=True)
        for report_name, report in reports.items():
            report.to_csv(directory / f'{report_name}.csv', index=False)
        (directory / 'metadata.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )
        mlflow.log_artifacts(str(directory), 'evaluation')
    logger.success(f'Evaluación de {question} guardada en {directory}')
    return reports


@app.command()
def main(
    question: str = typer.Option(..., help='Pregunta a evaluar: p1, p2 o p3'),
    output_dir: Path = typer.Option(OUTPUT_DIR, help='Directorio base de reportes CSV'),
):
    """Compara el champion contra su baseline sobre los mismos datos."""
    try:
        reports = evaluate_question(question, output_dir)
        print(reports['summary'].to_string(index=False, float_format=lambda value: f'{value:.4f}'))
    except (ValueError, OSError, mlflow.exceptions.MlflowException) as error:
        raise typer.BadParameter(str(error)) from error


if __name__ == '__main__':
    app()
