"""Entrenamiento y registro en MLflow de un modelo (baseline o candidato) de P1, P2 o P3.

Cada ejecución genera una sola corrida en MLflow.

Uso:
    python -m mi_bici.modeling.train --question p1 --model ridge --alpha 0.5
"""

import json
import os
import subprocess

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    adjusted_rand_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer

from mi_bici.config import (
    END_MONTH,
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
        'question': question.lower(),
        'role': role,
        'git_commit': get_git_commit(),
    }
    return params, tags


def plain_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cambia los tipos nullable de pandas por float/object para que la firma del modelo sea simple."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_extension_array_dtype(out[col]):
            numeric = pd.api.types.is_numeric_dtype(out[col])
            out[col] = out[col].astype('float64' if numeric else 'object')
    return out


def begin_run(
    question: str,
    model: str,
    hparams: dict,
    model_type: str,
    extra_params: dict,
    train_df: pd.DataFrame,
    target: str | None = None,
) -> None:
    """Registra parámetros, etiquetas y archivos de apoyo (manifiesto, config, datos de entrenamiento)."""
    role = 'baseline' if model == 'baseline' else 'candidate'
    params, tags = common_params_and_tags(question, role, model_type)
    tags.update({'model_family': model, 'stage_of_work': role})
    mlflow.log_params({**params, **extra_params, 'model': model, **hparams})
    mlflow.set_tags(tags)

    mlflow.log_artifact(str(MANIFEST_PATH), 'data')
    mlflow.log_artifact(str(PROJ_ROOT / 'mi_bici' / 'config.py'), 'config')
    mlflow.log_artifact(str(PROJ_ROOT / 'MLproject'), 'config')
    mlflow.log_dict({'question': question, 'model': model, 'hyperparameters': hparams}, 'config/run.json')

    dataset = mlflow.data.from_pandas(train_df, name=f'{question}_train', targets=target)
    mlflow.log_input(dataset, context='training')


def log_sklearn_model(pipe: Pipeline, X: pd.DataFrame) -> None:
    """Guarda el modelo con firma y un ejemplo de entrada."""
    sample = X.head(50)
    signature = infer_signature(sample, pipe.predict(sample))
    mlflow.sklearn.log_model(
        pipe,
        name='model',
        signature=signature,
        input_example=sample,
        serialization_format='cloudpickle',
    )


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


P1_NUM_COLS = [
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
P1_CAT_COLS = ['zone']


def build_p1_model(model: str, hparams: dict) -> Pipeline:
    """Mismo preprocesamiento para todos los regresores de P1."""
    regressors = {
        'ridge': lambda: Ridge(**hparams, random_state=RANDOM_SEED),
        'rf': lambda: RandomForestRegressor(**hparams, n_jobs=-1, random_state=RANDOM_SEED),
        'hgb': lambda: HistGradientBoostingRegressor(**hparams, random_state=RANDOM_SEED),
    }
    prep = ColumnTransformer(
        [
            ('num', StandardScaler(), P1_NUM_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore'), P1_CAT_COLS),
        ]
    )
    return Pipeline([('prep', prep), ('reg', regressors[model]())])


def run_p1(model: str, hparams: dict) -> None:
    """Entrena y registra en la corrida activa el baseline o un candidato de P1 (Regresión)."""
    df = plain_dtypes(pd.read_parquet(P1_FEATURES_PATH))
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    extra = {'horizon': '1_day', 'granularity': 'station_day'}

    # Baseline P1: Seasonal Naive (mismo día de la semana anterior = lag_7d)
    if model == 'baseline':
        begin_run('p1', model, hparams, 'SeasonalNaive_lag7d', extra, train_df, 'trips_count')

        metrics = {
            **regression_metrics(val_df['trips_count'].values, val_df['lag_7d'].values, 'val'),
            **regression_metrics(test_df['trips_count'].values, test_df['lag_7d'].values, 'test'),
        }
        mlflow.log_metrics(metrics)

        daily = test_df.assign(pred=test_df['lag_7d']).groupby('date')[['trips_count', 'pred']].sum()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily.index, daily['trips_count'], label='Real (Test)', color='black')
        ax.plot(daily.index, daily['pred'], label='Seasonal Naive (lag_7d)', color='#ff7f0e', linestyle='--')
        ax.set_title('P1 Baseline: Viajes Diarios Reales vs. Seasonal Naive (Test)')
        ax.set_ylabel('Viajes por día')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        mlflow.log_figure(fig, 'plots/p1_baseline_predictions.png')
        plt.close(fig)
        logger.info(f'P1 Baseline registrado: {metrics}')
        return

    begin_run('p1', model, hparams, model, extra, train_df, 'trips_count')

    features = P1_NUM_COLS + P1_CAT_COLS
    pipe = build_p1_model(model, hparams)
    pipe.fit(train_df[features], train_df['trips_count'])
    val_pred = pipe.predict(val_df[features])
    test_pred = pipe.predict(test_df[features])

    metrics = {
        **regression_metrics(val_df['trips_count'].values, val_pred, 'val'),
        **regression_metrics(test_df['trips_count'].values, test_pred, 'test'),
    }
    mlflow.log_metrics(metrics)
    log_sklearn_model(pipe, train_df[features])

    test_pred = np.clip(test_pred, 0.0, None)
    daily = (
        test_df.assign(pred_base=test_df['lag_7d'], pred_model=test_pred)
        .groupby('date')[['trips_count', 'pred_base', 'pred_model']]
        .sum()
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily['trips_count'], label='Real (Test)', color='black')
    ax.plot(daily.index, daily['pred_base'], label='Baseline (lag_7d)', color='#ff7f0e', linestyle='--', alpha=0.8)
    ax.plot(daily.index, daily['pred_model'], label=f'Candidato ({model})', color='#1f77b4')
    ax.set_title(f'P1 Test: Real vs. Baseline y {model}')
    ax.set_ylabel('Viajes por día')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    mlflow.log_figure(fig, f'plots/p1_{model}_predictions.png')
    plt.close(fig)

    residuals = test_df['trips_count'].values - test_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(test_pred, residuals, s=3, alpha=0.2)
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].set_xlabel('Predicho')
    axes[0].set_ylabel('Residual')
    axes[1].hist(residuals, bins=60, color='#1f77b4')
    axes[1].set_xlabel('Residual (viajes por estación-día)')
    fig.suptitle(f'P1 {model}: residuales en Test')
    plt.tight_layout()
    mlflow.log_figure(fig, f'plots/p1_{model}_residuals.png')
    plt.close(fig)
    logger.info(f'P1 Candidato {model} registrado: {metrics}')


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray | pd.Series, prefix: str) -> dict[str, float]:
    return {
        f'{prefix}_f1_macro': round(float(f1_score(y_true, y_pred, average='macro')), 4),
        f'{prefix}_recall_macro': round(float(recall_score(y_true, y_pred, average='macro')), 4),
        f'{prefix}_precision_macro': round(float(precision_score(y_true, y_pred, average='macro')), 4),
        # La decisión operativa es anticipar la falta de bicis
        f'{prefix}_recall_deficit': round(
            float(recall_score(y_true, y_pred, labels=['deficit'], average=None)[0]), 4
        ),
    }


P2_FEATURES = [
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


def build_p2_model(model: str, hparams: dict):
    """Clasificadores de P2 sobre las mismas características."""
    if model == 'hgb':
        return HistGradientBoostingClassifier(**hparams, random_state=RANDOM_SEED)
    if model == 'hgb_balanced':
        # Pesos por clase para no dejar pasar tantos casos de déficit
        return HistGradientBoostingClassifier(**hparams, class_weight='balanced', random_state=RANDOM_SEED)
    if model == 'rf':
        return RandomForestClassifier(**hparams, n_jobs=-1, random_state=RANDOM_SEED)
    return Pipeline(
        [
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(**hparams, max_iter=500, random_state=RANDOM_SEED)),
        ]
    )


def run_p2(model: str, hparams: dict) -> None:
    """Entrena y registra en la corrida activa el baseline o un candidato de P2 (Clasificación)."""
    df = plain_dtypes(pd.read_parquet(P2_FEATURES_PATH))
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()

    labels_order = ['deficit', 'balanceado', 'excedente']
    extra = {'horizon': '3_hours', 'pressure_threshold': 0.30}

    # Baseline P2: Misma clase observada hace 7 días (168 horas)
    if model == 'baseline':
        begin_run('p2', model, hparams, 'WeeklyRule_lag168h', extra, train_df, 'risk_class')

        metrics = {
            **classification_metrics(val_df['risk_class'], val_df['risk_lag_168h'], 'val'),
            **classification_metrics(test_df['risk_class'], test_df['risk_lag_168h'], 'test'),
        }
        mlflow.log_metrics(metrics)

        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(test_df['risk_class'], test_df['risk_lag_168h'], labels=labels_order)
        ConfusionMatrixDisplay(cm, display_labels=labels_order).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title('P2 Baseline (Regla 7 días) - Matriz de Confusión (Test)')
        plt.tight_layout()
        mlflow.log_figure(fig, 'plots/p2_baseline_confusion_matrix.png')
        plt.close(fig)
        logger.info(f'P2 Baseline registrado: {metrics}')
        return

    begin_run('p2', model, hparams, model, extra, train_df, 'risk_class')

    clf = build_p2_model(model, hparams)
    clf.fit(train_df[P2_FEATURES], train_df['risk_class'])
    val_pred = clf.predict(val_df[P2_FEATURES])
    test_pred = clf.predict(test_df[P2_FEATURES])

    metrics = {
        **classification_metrics(val_df['risk_class'], val_pred, 'val'),
        **classification_metrics(test_df['risk_class'], test_pred, 'test'),
    }
    mlflow.log_metrics(metrics)
    log_sklearn_model(clf, train_df[P2_FEATURES])

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(test_df['risk_class'], test_pred, labels=labels_order)
    ConfusionMatrixDisplay(cm, display_labels=labels_order).plot(ax=ax, cmap='Greens', colorbar=False)
    ax.set_title(f'P2 {model} - Matriz de Confusión (Test)')
    plt.tight_layout()
    mlflow.log_figure(fig, f'plots/p2_{model}_confusion_matrix.png')
    plt.close(fig)

    proba = clf.predict_proba(test_df[P2_FEATURES])
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, label in enumerate(clf.classes_):
        precision, recall, _ = precision_recall_curve(test_df['risk_class'] == label, proba[:, i])
        ax.plot(recall, precision, label=label)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precisión')
    ax.set_title(f'P2 {model} - Curva precisión-recall por clase (Test)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    mlflow.log_figure(fig, f'plots/p2_{model}_precision_recall.png')
    plt.close(fig)
    logger.info(f'P2 Candidato {model} registrado: {metrics}')


P3_USAGE_COLS = [
    'dep_morning',
    'dep_midday',
    'dep_evening',
    'dep_weekend',
    'median_duration_min',
    'round_trip_pct',
    'avg_daily_trips',
    'dep_arr_ratio',
]


def build_p3_model(model: str, hparams: dict) -> Pipeline:
    """Pipelines de clustering sobre el perfil de uso; todos exponen predict."""
    k = hparams['n_clusters']
    if model == 'gmm_usage':
        cluster = GaussianMixture(
            n_components=k, covariance_type=hparams['covariance_type'], random_state=RANDOM_SEED
        )
        return Pipeline([('scale', StandardScaler()), ('cluster', cluster)])
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    if model == 'kmeans_pca':
        pca = PCA(n_components=hparams['n_components'], random_state=RANDOM_SEED)
        return Pipeline([('scale', StandardScaler()), ('pca', pca), ('cluster', kmeans)])
    return Pipeline([('scale', StandardScaler()), ('cluster', kmeans)])


def run_p3(model: str, hparams: dict) -> None:
    """Entrena y registra en la corrida activa el baseline geográfico o un clustering de uso de P3."""
    n_clusters = hparams['n_clusters']
    prof_train = plain_dtypes(pd.read_parquet(P3_TRAIN_FEATURES_PATH))
    prof_holdout = plain_dtypes(pd.read_parquet(P3_HOLDOUT_FEATURES_PATH))

    usage_scaler = StandardScaler()
    X_train_usage = usage_scaler.fit_transform(prof_train[P3_USAGE_COLS])

    # Baseline P3: K-Means Geográfico (latitud y longitud)
    if model == 'baseline':
        geo_scaler = StandardScaler()
        X_train_geo = geo_scaler.fit_transform(prof_train[['latitude', 'longitude']])

        begin_run(
            'p3', model, hparams, 'KMeans_Geographic', {'features': 'latitude,longitude'},
            prof_train[['station_id', 'latitude', 'longitude']],
        )

        km_geo = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        labels_geo = km_geo.fit_predict(X_train_geo)

        # Sin ARI: las coordenadas no cambian entre periodos y la estabilidad saldría ~1 por construcción
        metrics = {
            'train_silhouette_usage': round(float(silhouette_score(X_train_usage, labels_geo)), 4),
            'train_davies_bouldin_usage': round(float(davies_bouldin_score(X_train_usage, labels_geo)), 4),
            'train_silhouette_geo': round(float(silhouette_score(X_train_geo, labels_geo)), 4),
        }
        mlflow.log_metrics(metrics)

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
        mlflow.log_figure(fig, 'plots/p3_baseline_geo_clusters.png')
        plt.close(fig)
        logger.info(f'P3 Baseline Geográfico registrado: {metrics}')
        return

    cols = ['station_id'] + P3_USAGE_COLS
    begin_run(
        'p3', model, hparams, model, {'features': ','.join(P3_USAGE_COLS)}, prof_train[cols]
    )

    pipe = build_p3_model(model, hparams)
    labels_usage = pipe.fit_predict(prof_train[P3_USAGE_COLS])

    # Estabilidad: se reajusta el mismo algoritmo con el perfil de 2026 y se compara con el de entrenamiento
    common = prof_train[['station_id']].assign(c_train=labels_usage).merge(
        prof_holdout, on='station_id', how='inner'
    )
    X_holdout = pipe[:-1].transform(common[P3_USAGE_COLS])
    labels_holdout = clone(pipe[-1]).fit_predict(X_holdout)

    metrics = {
        'train_silhouette_usage': round(float(silhouette_score(X_train_usage, labels_usage)), 4),
        'train_davies_bouldin_usage': round(float(davies_bouldin_score(X_train_usage, labels_usage)), 4),
        'holdout_ari_stability': round(float(adjusted_rand_score(common['c_train'], labels_holdout)), 4),
    }
    mlflow.log_metrics(metrics)
    log_sklearn_model(pipe, prof_train[P3_USAGE_COLS])

    pca_coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X_train_usage)
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
    axes[0].set_title(f'P3 {model}: Proyección PCA 2D (k={n_clusters})')
    axes[0].set_xlabel('PCA 1')
    axes[0].set_ylabel('PCA 2')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title(f'P3 {model}: Mapa de Estaciones en AMG (k={n_clusters})')
    axes[1].set_xlabel('Longitud')
    axes[1].set_ylabel('Latitud')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    mlflow.log_figure(fig, f'plots/p3_{model}_clusters_map.png')
    plt.close(fig)

    # Perfil medio de cada grupo (en desviaciones estándar) para interpretarlos
    profile = pd.DataFrame(X_train_usage, columns=P3_USAGE_COLS).groupby(labels_usage).mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    image = ax.imshow(profile.values, cmap='coolwarm', aspect='auto', vmin=-2, vmax=2)
    ax.set_xticks(range(len(P3_USAGE_COLS)), P3_USAGE_COLS, rotation=45, ha='right')
    ax.set_yticks(range(len(profile)), [f'Cluster {c}' for c in profile.index])
    ax.set_title(f'P3 {model}: perfil medio por grupo (z-score)')
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    mlflow.log_figure(fig, f'plots/p3_{model}_profiles.png')
    plt.close(fig)
    logger.info(f'P3 Candidato {model} registrado: {metrics}')


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
        RUNNERS[question](model, hparams)

    logger.success(f'Corrida {run_name} registrada en MLflow.')


if __name__ == '__main__':
    app()
