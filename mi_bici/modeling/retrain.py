"""Reentrena el champion de P2 con una ventana de datos más reciente y lo registra como nueva versión.

La familia y los hiperparámetros se leen de la corrida del champion, así la única diferencia entre
las dos versiones es la ventana de entrenamiento. La corrida usa stage_of_work='retrain' para que
register.py no la tome como candidata, y la versión recibe su propio alias sin mover @champion ni
@challenger.

Uso:
    python -m mi_bici.modeling.retrain --train-end 2026-08-31
"""

import os

from loguru import logger
import mlflow
from mlflow import MlflowClient
import pandas as pd
import typer

from mi_bici.config import (
    MANIFEST_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_DEFAULTS,
    P2_FEATURES_PATH,
    PROJ_ROOT,
    REGISTERED_MODELS,
)
from mi_bici.modeling.register import logged_model_id
from mi_bici.modeling.train import (
    P2_FEATURES,
    build_p2_model,
    common_params_and_tags,
    log_sklearn_model,
    plain_dtypes,
    resolve_hyperparams,
)

app = typer.Typer()

QUESTION = 'p2'
HORIZON_HOURS = 3
PROTECTED_ALIASES = {'champion', 'challenger'}


def label_cutoff(train_end: str) -> pd.Timestamp:
    """Última hora que se puede usar para entrenar sin ver datos posteriores a ``train_end``.

    risk_class en la hora t usa el flujo de t+1 a t+3, así que las últimas 3 horas del día
    tendrían etiquetas calculadas con datos del día siguiente.
    """
    return pd.Timestamp(train_end) + pd.Timedelta(hours=23 - HORIZON_HOURS)


def training_window(df: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """Filas desde el inicio de los datos hasta el corte, sin importar la columna split."""
    return df[pd.to_datetime(df['hour_ts']) <= label_cutoff(train_end)].copy()


def champion_spec(client: MlflowClient, question: str = QUESTION):
    """Versión, familia e hiperparámetros del champion actual."""
    name = REGISTERED_MODELS[question][0]
    version = client.get_model_version_by_alias(name, 'champion')
    params = client.get_run(version.run_id).data.params
    family = params['model']
    overrides = {key: params[key] for key in MODEL_DEFAULTS[question][family] if key in params}
    return version, family, resolve_hyperparams(question, family, overrides)


def log_retrain_run(
    family: str, hparams: dict, train_df: pd.DataFrame, cutoff: pd.Timestamp, champion
) -> None:
    """Parámetros, etiquetas y archivos de apoyo de la corrida de reentrenamiento."""
    params, tags = common_params_and_tags(QUESTION, 'retrain', family)
    params.update(
        {
            'train_end': str(cutoff),
            'train_start': str(train_df['hour_ts'].min()),
            'base_model_version': champion.version,
            'base_run_id': champion.run_id,
            'horizon': f'{HORIZON_HOURS}_hours',
            'model': family,
            **hparams,
        }
    )
    tags.update({'model_family': family, 'stage_of_work': 'retrain'})
    mlflow.log_params(params)
    mlflow.set_tags(tags)

    mlflow.log_artifact(str(MANIFEST_PATH), 'data')
    mlflow.log_artifact(str(PROJ_ROOT / 'mi_bici' / 'config.py'), 'config')
    mlflow.log_artifact(str(PROJ_ROOT / 'MLproject'), 'config')
    mlflow.log_dict(
        {
            'question': QUESTION,
            'model': family,
            'hyperparameters': hparams,
            'train_end': str(cutoff),
        },
        'config/run.json',
    )
    dataset = mlflow.data.from_pandas(train_df, name=f'{QUESTION}_retrain', targets='risk_class')
    mlflow.log_input(dataset, context='training')


def register_retrain(client: MlflowClient, run_id: str, family: str, cutoff, champion, alias: str):
    """Registra la corrida como nueva versión del modelo de P2 con el alias indicado."""
    name, purpose = REGISTERED_MODELS[QUESTION]
    experiment_id = client.get_run(run_id).info.experiment_id
    version = mlflow.register_model(
        f'models:/{logged_model_id(client, experiment_id, run_id)}', name
    )
    client.update_model_version(
        name,
        version.version,
        description=(
            f'{purpose}. Reentrenamiento de la v{champion.version} ({family}) '
            f'con datos hasta {cutoff}.'
        ),
    )
    run = client.get_run(run_id)
    for key, value in {
        'git_commit': run.data.tags.get('git_commit'),
        'data_version': run.data.params.get('data_version'),
        'experiment': MLFLOW_EXPERIMENT_NAME,
        'source_run_id': run_id,
        'model_family': family,
        'train_end': str(cutoff),
        'base_model_version': champion.version,
    }.items():
        client.set_model_version_tag(name, version.version, key, str(value))
    client.set_registered_model_alias(name, alias, version.version)
    logger.info(f'{name} v{version.version} ({family}, datos hasta {cutoff}) -> @{alias}')
    return version


def retrain(train_end: str, alias: str):
    """Entrena con la ventana nueva, registra la versión y verifica que cargue por alias."""
    client = MlflowClient(MLFLOW_TRACKING_URI)
    champion, family, hparams = champion_spec(client)
    df = plain_dtypes(pd.read_parquet(P2_FEATURES_PATH))
    train_df = training_window(df, train_end)
    cutoff = label_cutoff(train_end)
    logger.info(
        f'Champion v{champion.version} ({family}, {hparams}); '
        f'reentrenando con {len(train_df):,} filas hasta {cutoff}'
    )

    with mlflow.start_run(run_name=f'{QUESTION}-retrain-{family}') as run:
        log_retrain_run(family, hparams, train_df, cutoff, champion)
        clf = build_p2_model(family, hparams)
        clf.fit(train_df[P2_FEATURES], train_df['risk_class'])
        mlflow.log_metrics(
            {
                'train_rows': len(train_df),
                'train_deficit_share': float((train_df['risk_class'] == 'deficit').mean()),
            }
        )
        log_sklearn_model(clf, train_df[P2_FEATURES])
        run_id = run.info.run_id

    version = register_retrain(client, run_id, family, cutoff, champion, alias)
    name = REGISTERED_MODELS[QUESTION][0]
    mlflow.pyfunc.load_model(f'models:/{name}@{alias}')
    logger.success(f'models:/{name}@{alias} (v{version.version}) se carga correctamente')
    return version


@app.command()
def main(
    train_end: str = typer.Option('2026-08-31', help='Último día de datos para entrenar'),
    alias: str = typer.Option('retrain', help='Alias de la nueva versión en el registry'),
):
    """Reentrena el champion de P2 con datos hasta train_end y lo registra con un alias propio."""
    if alias in PROTECTED_ALIASES:
        raise typer.BadParameter(f'El alias {alias!r} lo asigna register.py; usa otro.')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if not os.environ.get('MLFLOW_RUN_ID'):
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    retrain(train_end, alias)


if __name__ == '__main__':
    app()
