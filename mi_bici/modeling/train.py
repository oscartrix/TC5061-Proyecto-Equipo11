"""Entrenamiento y registro en MLflow de un modelo (baseline o candidato) de P1, P2 o P3.

Cada ejecución genera una sola corrida en MLflow.

Uso:
    python -m mi_bici.modeling.train --question p1 --model ridge --alpha 0.5
"""

import json
import subprocess
import os

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    adjusted_rand_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer

from mi_bici.config import (
    END_MONTH,
    FIGURES_DIR,
    MANIFEST_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    P1_FEATURES_PATH,
    P2_FEATURES_PATH,
    P3_HOLDOUT_FEATURES_PATH,
    P3_TRAIN_FEATURES_PATH,
    PROJ_ROOT,
    RANDOM_SEED,
    START_MONTH,
    TEAM_NAME,
    TEST_START_DATE,
    TRAIN_END_DATE,
    VAL_END_DATE,
    VAL_START_DATE,
    MODEL_DEFAULTS,
)

app = typer.Typer()


def get_git_commit() -> str:
    """Obtiene el hash del commit actual de Git."""
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short=12', 'HEAD'], cwd=PROJ_ROOT, text=True
        )
        return out.strip()
    except Exception:
        return 'unknown'


def get_data_version() -> str:
    """Lee la versión de los datos desde config/data_manifest.json."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding='utf-8') as f:
            data = json.load(f)
            return str(data.get('data_version', 'unknown'))
    return 'unknown'


def common_params_and_tags(question: str, role: str, model_type: str) -> tuple[dict, dict]:
    """Parámetros de reproducción y etiquetas comunes a todos los runs."""
    params = {
        'question': question.lower(),
        'role': role,
        'model_type': model_type,
        'start_month': START_MONTH,
        'end_month': END_MONTH,
        'train_end': TRAIN_END_DATE,
        'val_period': f'{VAL_START_DATE}_to_{VAL_END_DATE}',
        'test_start': TEST_START_DATE,
        'data_version': get_data_version(),
        'random_seed': RANDOM_SEED,
    }
    tags = {
        'team': TEAM_NAME,
        'question': question,
        'role': role,
        'git_commit': get_git_commit(),
    }
    return params, tags


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    y_pred_clip = np.clip(y_pred, 0.0, None)
    mae = mean_absolute_error(y_true, y_pred_clip)
    rmse = root_mean_squared_error(y_true, y_pred_clip)
    wape = float(np.sum(np.abs(y_true - y_pred_clip)) / np.maximum(np.sum(np.abs(y_true)), 1.0))
    return {
        f'{prefix}_mae': round(float(mae), 4),
        f'{prefix}_rmse': round(float(rmse), 4),
        f'{prefix}_wape': round(float(wape), 4),
    }


def run_p1(model: str, hparams: dict) -> None:
    """Entrena y registra en la corrida activa el baseline o el candidato de P1 (Regresión)."""
    df = pd.read_parquet(P1_FEATURES_PATH)
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline P1: Seasonal Naive (mismo día de la semana anterior = lag_7d)
    if model == 'baseline':
        params, tags = common_params_and_tags('P1', 'baseline', 'SeasonalNaive_lag7d')
        params['horizon'] = '1_day'
        params['granularity'] = 'station_day'
        mlflow.log_params(params)
        mlflow.set_tags(tags)

        metrics = {
            **regression_metrics(val_df['trips_count'].values, val_df['lag_7d'].values, 'val'),
            **regression_metrics(test_df['trips_count'].values, test_df['lag_7d'].values, 'test'),
        }
        mlflow.log_metrics(metrics)

        fig_path = FIGURES_DIR / 'p1_baseline_predictions.png'
        daily = test_df.assign(pred=test_df['lag_7d']).groupby('date')[['trips_count', 'pred']].sum()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily.index, daily['trips_count'], label='Real (Test)', color='black')
        ax.plot(daily.index, daily['pred'], label='Seasonal Naive (lag_7d)', color='#ff7f0e', linestyle='--')
        ax.set_title('P1 Baseline: Viajes Diarios Reales vs. Seasonal Naive (Test)')
        ax.set_ylabel('Viajes por día')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(fig_path))
        logger.info(f'P1 Baseline registrado: {metrics}')
        return

    # Candidato P1: Ridge Regression
    num_cols = [
        'lag_1d',
        'lag_7d',
        'lag_14d',
        'lag_21d',
        'rolling_mean_7d',
        'capacity',
        'day_of_week',
        'month',
        'is_weekend',
        'is_holiday',
    ]
    cat_cols = ['zone']

    params, tags = common_params_and_tags('P1', 'candidate', 'RidgeRegression')
    params['horizon'] = '1_day'
    params['granularity'] = 'station_day'
    mlflow.log_params(params)
    mlflow.set_tags(tags)

    pipe = Pipeline(
        [
            (
                'prep',
                ColumnTransformer(
                    [
                        ('num', StandardScaler(), num_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
                    ]
                ),
            ),
            ('reg', Ridge(alpha=hparams['alpha'], random_state=RANDOM_SEED)),
        ]
    )
    pipe.fit(train_df[num_cols + cat_cols], train_df['trips_count'])
    val_pred = pipe.predict(val_df[num_cols + cat_cols])
    test_pred = pipe.predict(test_df[num_cols + cat_cols])

    metrics = {
        **regression_metrics(val_df['trips_count'].values, val_pred, 'val'),
        **regression_metrics(test_df['trips_count'].values, test_pred, 'test'),
    }
    mlflow.log_metrics(metrics)

    fig_path = FIGURES_DIR / 'p1_candidate_ridge_predictions.png'
    daily = (
        test_df.assign(
            pred_base=test_df['lag_7d'],
            pred_ridge=np.clip(test_pred, 0.0, None),
        )
        .groupby('date')[['trips_count', 'pred_base', 'pred_ridge']]
        .sum()
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily['trips_count'], label='Real (Test)', color='black')
    ax.plot(daily.index, daily['pred_base'], label='Baseline (lag_7d)', color='#ff7f0e', linestyle='--', alpha=0.8)
    ax.plot(daily.index, daily['pred_ridge'], label='Candidato (Ridge)', color='#1f77b4')
    ax.set_title('P1 Comparación en Test: Real vs. Baseline y Ridge Regression')
    ax.set_ylabel('Viajes por día')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(fig_path))
    logger.info(f'P1 Candidato Ridge registrado: {metrics}')


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray | pd.Series, prefix: str) -> dict[str, float]:
    return {
        f'{prefix}_f1_macro': round(float(f1_score(y_true, y_pred, average='macro')), 4),
        f'{prefix}_recall_macro': round(float(recall_score(y_true, y_pred, average='macro')), 4),
        f'{prefix}_precision_macro': round(float(precision_score(y_true, y_pred, average='macro')), 4),
    }


def run_p2(model: str, hparams: dict) -> None:
    """Entrena y registra en la corrida activa el baseline o el candidato de P2 (Clasificación)."""
    df = pd.read_parquet(P2_FEATURES_PATH)
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()

    labels_order = ['deficit', 'balanceado', 'excedente']
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline P2: Misma clase observada hace 7 días (168 horas)
    if model == 'baseline':
        params, tags = common_params_and_tags('P2', 'baseline', 'WeeklyRule_lag168h')
        params['horizon'] = '3_hours'
        params['pressure_threshold'] = 0.30
        mlflow.log_params(params)
        mlflow.set_tags(tags)

        metrics = {
            **classification_metrics(val_df['risk_class'], val_df['risk_lag_168h'], 'val'),
            **classification_metrics(test_df['risk_class'], test_df['risk_lag_168h'], 'test'),
        }
        mlflow.log_metrics(metrics)

        fig_path = FIGURES_DIR / 'p2_baseline_confusion_matrix.png'
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(test_df['risk_class'], test_df['risk_lag_168h'], labels=labels_order)
        ConfusionMatrixDisplay(cm, display_labels=labels_order).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title('P2 Baseline (Regla 7 días) - Matriz de Confusión (Test)')
        plt.tight_layout()
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(fig_path))
        logger.info(f'P2 Baseline registrado: {metrics}')
        return

    # Candidato P2: HistGradientBoostingClassifier
    feature_cols = [
        'hour',
        'day_of_week',
        'is_weekend',
        'departures',
        'arrivals',
        'net_flow',
        'pressure_last_3h',
        'pressure_window_lag_24h',
        'pressure_window_lag_168h',
        'pressure_window_lag_336h',
        'capacity',
    ]

    params, tags = common_params_and_tags('P2', 'candidate', 'HistGradientBoostingClassifier')
    params['horizon'] = '3_hours'
    params['pressure_threshold'] = 0.30
    mlflow.log_params(params)
    mlflow.set_tags(tags)

    clf = HistGradientBoostingClassifier(**hparams, random_state=RANDOM_SEED)
    clf.fit(train_df[feature_cols], train_df['risk_class'])
    val_pred = clf.predict(val_df[feature_cols])
    test_pred = clf.predict(test_df[feature_cols])

    metrics = {
        **classification_metrics(val_df['risk_class'], val_pred, 'val'),
        **classification_metrics(test_df['risk_class'], test_pred, 'test'),
    }
    mlflow.log_metrics(metrics)

    fig_path = FIGURES_DIR / 'p2_candidate_hgb_confusion_matrix.png'
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(test_df['risk_class'], test_pred, labels=labels_order)
    ConfusionMatrixDisplay(cm, display_labels=labels_order).plot(ax=ax, cmap='Greens', colorbar=False)
    ax.set_title('P2 HistGradientBoosting - Matriz de Confusión (Test)')
    plt.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(fig_path))
    logger.info(f'P2 Candidato HGB registrado: {metrics}')


def run_p3(model: str, hparams: dict) -> None:
    """Entrena y registra en la corrida activa el baseline geográfico o el clustering de uso de P3."""
    n_clusters = hparams['n_clusters']
    prof_train = pd.read_parquet(P3_TRAIN_FEATURES_PATH)
    prof_holdout = pd.read_parquet(P3_HOLDOUT_FEATURES_PATH)

    usage_cols = [
        'dep_morning',
        'dep_midday',
        'dep_evening',
        'dep_weekend',
        'median_duration_min',
        'round_trip_pct',
        'avg_daily_trips',
        'dep_arr_ratio',
    ]

    usage_scaler = StandardScaler()
    X_train_usage = usage_scaler.fit_transform(prof_train[usage_cols])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline P3: K-Means Geográfico (latitud y longitud)
    if model == 'baseline':
        geo_scaler = StandardScaler()
        X_train_geo = geo_scaler.fit_transform(prof_train[['latitude', 'longitude']])

        params, tags = common_params_and_tags('P3', 'baseline', 'KMeans_Geographic')
        params['features'] = 'latitude,longitude'
        mlflow.log_params(params)
        mlflow.set_tags(tags)

        km_geo = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        labels_geo = km_geo.fit_predict(X_train_geo)

        # Sin ARI: las coordenadas no cambian entre periodos y la estabilidad saldría ~1 por construcción
        metrics = {
            'train_silhouette_usage': round(float(silhouette_score(X_train_usage, labels_geo)), 4),
            'train_davies_bouldin_usage': round(float(davies_bouldin_score(X_train_usage, labels_geo)), 4),
            'train_silhouette_geo': round(float(silhouette_score(X_train_geo, labels_geo)), 4),
        }
        mlflow.log_metrics(metrics)

        fig_path = FIGURES_DIR / 'p3_baseline_geo_clusters.png'
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in range(n_clusters):
            sub = prof_train[labels_geo == c]
            ax.scatter(sub['longitude'], sub['latitude'], label=f'Cluster Geo {c}', alpha=0.75, s=28)
        ax.set_title(f'P3 Baseline: K-Means Geográfico (k={n_clusters})')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(fig_path))
        logger.info(f'P3 Baseline Geográfico registrado: {metrics}')
        return

    # Candidato P3: K-Means sobre Perfil de Uso
    params, tags = common_params_and_tags('P3', 'candidate', 'KMeans_UsageProfile')
    params['features'] = ','.join(usage_cols)
    mlflow.log_params(params)
    mlflow.set_tags(tags)

    km_usage = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels_usage = km_usage.fit_predict(X_train_usage)

    common = prof_train[['station_id']].assign(c_train=labels_usage).merge(
        prof_holdout, on='station_id', how='inner'
    )
    X_holdout_usage = usage_scaler.transform(common[usage_cols])
    labels_holdout = KMeans(
        n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10
    ).fit_predict(X_holdout_usage)

    metrics = {
        'train_silhouette_usage': round(float(silhouette_score(X_train_usage, labels_usage)), 4),
        'train_davies_bouldin_usage': round(float(davies_bouldin_score(X_train_usage, labels_usage)), 4),
        'holdout_ari_stability': round(float(adjusted_rand_score(common['c_train'], labels_holdout)), 4),
    }
    mlflow.log_metrics(metrics)

    pca_coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X_train_usage)
    fig_path = FIGURES_DIR / 'p3_candidate_usage_clusters_map.png'
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c in range(n_clusters):
        mask = labels_usage == c
        axes[0].scatter(pca_coords[mask, 0], pca_coords[mask, 1], label=f'Cluster {c}', alpha=0.75, s=28)
        axes[1].scatter(
            prof_train.loc[mask, 'longitude'],
            prof_train.loc[mask, 'latitude'],
            label=f'Cluster {c}',
            alpha=0.75,
            s=28,
        )
    axes[0].set_title(f'P3 Perfil de Uso: Proyección PCA 2D (k={n_clusters})')
    axes[0].set_xlabel('PCA 1')
    axes[0].set_ylabel('PCA 2')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title(f'P3 Perfil de Uso: Mapa de Estaciones en AMG (k={n_clusters})')
    axes[1].set_xlabel('Longitud')
    axes[1].set_ylabel('Latitud')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(fig_path))
    logger.info(f'P3 Candidato Perfil de Uso registrado: {metrics}')


RUNNERS = {'p1': run_p1, 'p2': run_p2, 'p3': run_p3}


def parse_hyperparam_args(args: list[str]) -> dict[str, str]:
    """Convertir los -P extra de mlflow run (--alpha 0.5) en {'alpha': '0.5'}."""
    overrides = {}
    tokens = iter(args)
    for token in tokens:
        if not token.startswith('--'):
            raise typer.BadParameter(f'Argumento no válido: {token!r}')
        name, sep, value = token[2:].partition('=')
        if not sep:
            value = next(tokens, None)
            if value is None:
                raise typer.BadParameter(f'Falta el valor de --{name}')
        overrides[name] = value
    return overrides


def resolve_hyperparams(question: str, model: str, overrides: dict[str, str]) -> dict:
    """Combina los hiperparámetros de MODEL_DEFAULTS con los argumentos recibidos."""
    models = MODEL_DEFAULTS[question]
    if model not in models:
        raise typer.BadParameter(
            f'El modelo {model!r} no existe para {question}. Opciones: {", ".join(models)}'
        )

    hparams = dict(models[model])
    for name, value in overrides.items():
        if name not in hparams:
            valid = ', '.join(hparams) or 'ninguno'
            raise typer.BadParameter(
                f'{name!r} no aplica al modelo {model!r} de {question}. '
                f'Hiperparámetros válidos: {valid}'
            )
        try:
            hparams[name] = type(hparams[name])(value)
        except ValueError as error:
            raise typer.BadParameter(f'Valor inválido para {name}: {value!r}') from error
    return hparams


def log_hyperparams(question: str, model: str, hparams: dict, overrides: dict[str, str]) -> None:
    """Registra en MLflow la pregunta, el modelo y los hiperparámetros usados."""
    mlflow.log_params({'question': question, 'model': model, **hparams, **overrides})


def format_models() -> str:
    """Lista los modelos de MODEL_DEFAULTS con sus hiperparámetros por default."""
    lines = []
    for question, models in MODEL_DEFAULTS.items():
        lines.append(question)
        for model, hparams in models.items():
            values = ', '.join(f'{name}={value}' for name, value in hparams.items())
            lines.append(f'  {model}: {values or "(sin hiperparámetros)"}')
    return '\n'.join(lines)


@app.command(context_settings={'allow_extra_args': True, 'ignore_unknown_options': True})
def main(
    ctx: typer.Context,
    question: str = typer.Option('p1', help='Pregunta a ejecutar: p1, p2 o p3'),
    model: str = typer.Option('baseline', help='Modelo definido en MODEL_DEFAULTS de config.py'),
    list_models: bool = typer.Option(
        False, '--list-models', help='Muestra los modelos y sus hiperparámetros por default'
    ),
):
    """Entrena un modelo de P1, P2 o P3 y lo registra como una corrida en MLflow."""
    if list_models:
        typer.echo(format_models())
        raise typer.Exit()

    if question not in RUNNERS:
        raise typer.BadParameter(f'La pregunta debe ser una de: {", ".join(RUNNERS)}')

    overrides = parse_hyperparam_args(ctx.args)
    hparams = resolve_hyperparams(question, model, overrides)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    project_run_id = os.environ.get('MLFLOW_RUN_ID')
    if project_run_id:
        experiment_id = mlflow.get_run(project_run_id).info.experiment_id
        experiment_name = mlflow.get_experiment(experiment_id).name
        if experiment_name != MLFLOW_EXPERIMENT_NAME:
            logger.error(
                f'La corrida se creó en el experimento {experiment_name!r}. '
                f'Ejecuta mlflow run con --experiment-name {MLFLOW_EXPERIMENT_NAME}'
            )
            raise typer.Exit(code=1)
    else:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    run_name = f'{question}-baseline' if model == 'baseline' else f'{question}-candidate-{model}'
    logger.info(
        f'MLflow Tracking URI: {MLFLOW_TRACKING_URI} | Experimento: {MLFLOW_EXPERIMENT_NAME} | '
        f'Corrida: {run_name} | Hiperparámetros: {hparams}'
    )

    # Retomar un run
    with mlflow.start_run(run_name=run_name):
        log_hyperparams(question, model, hparams, overrides)
        RUNNERS[question](model, hparams)

    logger.success(f'Corrida {run_name} registrada en MLflow.')


if __name__ == '__main__':
    app()
