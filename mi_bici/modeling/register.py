"""Elige el mejor candidato de cada pregunta, lo registra en el Model Registry y le asigna el alias champion.

Uso:
    python -m mi_bici.modeling.register --question p1
"""

from loguru import logger
import mlflow
from mlflow import MlflowClient
import pandas as pd
import typer

from mi_bici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    P3_MIN_STABILITY,
    REGISTERED_MODELS,
    SELECTION_METRICS,
)

app = typer.Typer()

# Columnas de la tabla comparativa por pregunta
REPORT_METRICS = {
    'p1': ['val_mae', 'val_wape', 'test_mae', 'test_wape'],
    'p2': ['val_f1_macro', 'val_recall_deficit', 'test_f1_macro', 'test_recall_deficit'],
    'p3': ['train_silhouette_usage', 'train_davies_bouldin_usage', 'holdout_ari_stability'],
}


def ranked_runs(client: MlflowClient, experiment_id: str, question: str) -> pd.DataFrame:
    """Corridas candidatas de una pregunta ordenadas de mejor a peor según la métrica de validación."""
    metric, direction = SELECTION_METRICS[question]
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.question = '{question}' and tags.stage_of_work = 'candidate' and status = 'FINISHED'",
    )
    column = f'metrics.{metric}'
    runs = runs.dropna(subset=[column])
    if question == 'p3':
        runs = runs[runs['metrics.holdout_ari_stability'] >= P3_MIN_STABILITY]
    return runs.sort_values(column, ascending=(direction == 'min')).reset_index(drop=True)


def comparison_table(client: MlflowClient, experiment_id: str, question: str) -> pd.DataFrame:
    """Baseline y candidatos con sus métricas, para el reporte."""
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.question = '{question}' and tags.stage_of_work in ('baseline', 'candidate')",
    )
    metrics = [f'metrics.{m}' for m in REPORT_METRICS[question]]
    table = runs[['run_id', 'tags.stage_of_work', 'tags.model_family', *metrics]].copy()
    table.columns = ['run_id', 'etapa', 'modelo', *REPORT_METRICS[question]]
    return table.dropna(subset=REPORT_METRICS[question][:1]).sort_values(
        REPORT_METRICS[question][0], ascending=SELECTION_METRICS[question][1] == 'min'
    )


def logged_model_id(client: MlflowClient, experiment_id: str, run_id: str) -> str:
    """Id del modelo que guardó la corrida con log_model."""
    models = client.search_logged_models(
        experiment_ids=[experiment_id], filter_string=f"source_run_id = '{run_id}'"
    )
    if not models:
        raise ValueError(f'La corrida {run_id} no guardó un modelo')
    return models[0].model_id


def register_run(client: MlflowClient, experiment_id: str, question: str, run: pd.Series, alias: str):
    """Registra la corrida como versión del modelo de la pregunta y le asigna el alias."""
    name, purpose = REGISTERED_MODELS[question]
    metric, _ = SELECTION_METRICS[question]
    run_id = run['run_id']

    try:
        client.get_registered_model(name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(name, description=f'{purpose}. Se promueve con el alias champion.')

    # Si la corrida ya se registró antes, se reutiliza su versión
    version = next(
        (v for v in client.search_model_versions(f"name = '{name}'") if v.run_id == run_id), None
    )
    if version is None:
        version = mlflow.register_model(f'models:/{logged_model_id(client, experiment_id, run_id)}', name)

    family = run['tags.model_family']
    client.update_model_version(
        name,
        version.version,
        description=(
            f'{purpose}. Modelo {family}, elegido por {metric}={run[f"metrics.{metric}"]:.4f} en validación.'
        ),
    )
    for key, value in {
        'git_commit': run['tags.git_commit'],
        'data_version': run['params.data_version'],
        'experiment': MLFLOW_EXPERIMENT_NAME,
        'source_run_id': run_id,
        'model_family': family,
    }.items():
        client.set_model_version_tag(name, version.version, key, str(value))
    client.set_registered_model_alias(name, alias, version.version)
    logger.info(f'{name} v{version.version} ({family}) -> @{alias}')
    return version


def register_question(question: str) -> None:
    client = MlflowClient(MLFLOW_TRACKING_URI)
    experiment_id = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME).experiment_id
    ranking = ranked_runs(client, experiment_id, question)
    if ranking.empty:
        raise typer.BadParameter(f'No hay candidatos con modelo para {question}')

    print(comparison_table(client, experiment_id, question).to_string(index=False))
    register_run(client, experiment_id, question, ranking.iloc[0], 'champion')
    if len(ranking) > 1:
        register_run(client, experiment_id, question, ranking.iloc[1], 'challenger')

    # El champion debe poder cargarse solo con nombre y alias
    name = REGISTERED_MODELS[question][0]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.pyfunc.load_model(f'models:/{name}@champion')
    logger.success(f'models:/{name}@champion se carga correctamente')


@app.command()
def main(question: str = typer.Option(..., help='Pregunta a registrar: p1, p2 o p3')):
    """Registra el mejor candidato de la pregunta como champion."""
    if question not in REGISTERED_MODELS:
        raise typer.BadParameter(f'La pregunta debe ser una de: {", ".join(REGISTERED_MODELS)}')
    register_question(question)


if __name__ == '__main__':
    app()
