"""Entrenamiento y registro en MLflow de baselines y modelos candidatos para P1, P2 y P3.

Uso:
    python -m mi_bici.modeling.train
"""

import json
from pathlib import Path
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
        'question': question,
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


def run_p1_experiments() -> None:
    """Ejecuta y registra en MLflow el baseline y modelo candidato de P1 (Regresión)."""
    df = pd.read_parquet(P1_FEATURES_PATH)
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Baseline P1: Seasonal Naive (mismo día de la semana anterior = lag_7d)
    with mlflow.start_run(run_name='p1-baseline-seasonal-naive', nested=True):
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

    # 2. Candidato P1: Ridge Regression
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

    with mlflow.start_run(run_name='p1-candidate-ridge-regression', nested=True):
        params, tags = common_params_and_tags('P1', 'candidate', 'RidgeRegression')
        params['horizon'] = '1_day'
        params['granularity'] = 'station_day'
        params['alpha'] = 1.0
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
                ('reg', Ridge(alpha=1.0, random_state=RANDOM_SEED)),
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


def run_p2_experiments() -> None:
    """Ejecuta y registra en MLflow el baseline y modelo candidato de P2 (Clasificación)."""
    df = pd.read_parquet(P2_FEATURES_PATH)
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()

    labels_order = ['deficit', 'balanceado', 'excedente']
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Baseline P2: Misma clase observada hace 7 días (168 horas)
    with mlflow.start_run(run_name='p2-baseline-weekly-rule', nested=True):
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

    # 2. Candidato P2: HistGradientBoostingClassifier
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

    with mlflow.start_run(run_name='p2-candidate-hist-gradient-boosting', nested=True):
        params, tags = common_params_and_tags('P2', 'candidate', 'HistGradientBoostingClassifier')
        params['horizon'] = '3_hours'
        params['pressure_threshold'] = 0.30
        params['max_iter'] = 150
        params['learning_rate'] = 0.08
        params['max_depth'] = 6
        mlflow.log_params(params)
        mlflow.set_tags(tags)

        clf = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_depth=6, random_state=RANDOM_SEED
        )
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


def run_p3_experiments(n_clusters: int = 4) -> None:
    """Ejecuta y registra en MLflow el baseline geográfico y el clustering de uso para P3."""
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

    geo_scaler = StandardScaler()
    X_train_geo = geo_scaler.fit_transform(prof_train[['latitude', 'longitude']])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Baseline P3: K-Means Geográfico (latitud y longitud)
    with mlflow.start_run(run_name='p3-baseline-kmeans-geo', nested=True):
        params, tags = common_params_and_tags('P3', 'baseline', 'KMeans_Geographic')
        params['n_clusters'] = n_clusters
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

    # 2. Candidato P3: K-Means sobre Perfil de Uso
    with mlflow.start_run(run_name='p3-candidate-kmeans-usage', nested=True):
        params, tags = common_params_and_tags('P3', 'candidate', 'KMeans_UsageProfile')
        params['n_clusters'] = n_clusters
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


@app.command()
def main(
    question: str = typer.Option("all",help="Pregunta a ejecutar: p1, p2, p3 o all"),
    tracking_uri: str = typer.Option("http://127.0.0.1:5000", help='URI de MLflow Tracking'),
):
    """Ejecuta los experimentos base y candidatos de P1, P2 y P3 registrándolos en MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f'MLflow Tracking URI: {tracking_uri} | Experimento: {MLFLOW_EXPERIMENT_NAME}')

    question = question.lower()
    
    if question not in ["p1", "p2", "p3", "all"]:
        raise typer.BadParameter(
            "La pregunta debe ser p1, p2, p3 o all."
        )

    # MLflow Projects crea una corrida principal automáticamente.
    project_run_id = os.environ.pop("MLFLOW_RUN_ID", None)

    if project_run_id:
        logger.info(f"Corrida principal de MLflow Project: {project_run_id}")

        # Reutilizar la corrida principal únicamente como padre.
        with mlflow.start_run(run_id=project_run_id):
            mlflow.set_tag("execution_type", "mlflow_project")

            if question in ["p1", "all"]:
                run_p1_experiments()

            if question in ["p2", "all"]:
                run_p2_experiments()

            if question in ["p3", "all"]:
                run_p3_experiments()

    else:
        # Permitir también la ejecución directa con Python.
        if question in ["p1", "all"]:
            run_p1_experiments()

        if question in ["p2", "all"]:
            run_p2_experiments()

        if question in ["p3", "all"]:
            run_p3_experiments()
    logger.success(f"Experimentos de {question} finalizados.")


if __name__ == '__main__':
    app()
