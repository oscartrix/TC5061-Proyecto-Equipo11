"""A/B testing entre dos versiones registradas del modelo de P2.

Las dos versiones predicen las mismas filas de un periodo que ninguna vio al entrenar.
Se comparan con un bootstrap por bloques estación-día, una prueba de McNemar y una
simulación de tráfico 50/50, y se decide con una regla fija antes de ver resultados.

Uso:
    python -m mi_bici.monitoring.ab_testing --challenger-alias retrain-jun
"""

import json
import os

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import typer

from mi_bici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    P2_FEATURES_PATH,
    REGISTERED_MODELS,
    REPORTS_DIR,
)
from mi_bici.modeling.evaluation import RISK_LABELS, class_statistics
from mi_bici.modeling.train import P2_FEATURES, plain_dtypes

app = typer.Typer()

MODEL_NAME = REGISTERED_MODELS['p2'][0]
OUTPUT_DIR = REPORTS_DIR / 'ab_testing'
KEY_METRICS = ('f1_macro', 'recall_deficit', 'precision_deficit')
REPORT_METRICS = KEY_METRICS + ('accuracy', 'missed_deficits', 'false_deficit_alerts')
N_BOOTSTRAP = 1000
SEED = 42
# B puede perder hasta 0.01 de recall de déficit, que es el objetivo operativo de P2
RECALL_MARGIN = 0.01


def load_period(start: str, end: str) -> pd.DataFrame:
    """Filas de P2 entre start y end (inclusive), con el bloque estación-día."""
    columns = ['station_id', 'hour_ts', 'risk_class', *P2_FEATURES]
    df = plain_dtypes(pd.read_parquet(P2_FEATURES_PATH, columns=columns))
    hours = pd.to_datetime(df['hour_ts'])
    mask = (hours >= pd.Timestamp(start)) & (hours < pd.Timestamp(end) + pd.Timedelta(days=1))
    df = df[mask].reset_index(drop=True)
    if df.empty:
        raise ValueError(f'No hay datos de P2 entre {start} y {end}.')
    df['block'] = df['station_id'].astype(str) + '_' + hours[mask].dt.date.astype(str).values
    df['month'] = hours[mask].dt.strftime('%Y-%m').values
    return df


def version_info(client: MlflowClient, alias: str) -> dict:
    """Versión, corrida, familia y último día de entrenamiento de un alias."""
    version = client.get_model_version_by_alias(MODEL_NAME, alias)
    params = client.get_run(version.run_id).data.params
    return {
        'alias': alias,
        'version': version.version,
        'run_id': version.run_id,
        'model': params.get('model'),
        'train_end': params['train_end'],
    }


def check_unseen(info: dict, start: str) -> None:
    """Falla si la versión entrenó con datos del periodo de evaluación."""
    if pd.Timestamp(info['train_end']) >= pd.Timestamp(start):
        raise ValueError(
            f'@{info["alias"]} (v{info["version"]}) entrenó hasta {info["train_end"]}; '
            f'no se puede evaluar desde {start}.'
        )


def matrix_metrics(matrix: np.ndarray) -> dict:
    """F1 macro, recall y precisión de déficit a partir de una matriz de confusión."""
    hits = matrix.diagonal()
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(hits, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(hits, support, out=np.zeros(3), where=support > 0)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros(3), where=total > 0)
    return {
        'f1_macro': float(f1.mean()),
        'recall_deficit': float(recall[0]),
        'precision_deficit': float(precision[0]),
    }


def block_matrices(actual, predicted, blocks) -> np.ndarray:
    """Matriz de confusión 3x3 de cada bloque, con forma (n_bloques, 3, 3)."""
    index = {label: i for i, label in enumerate(RISK_LABELS)}
    a = pd.Series(actual).map(index).to_numpy()
    p = pd.Series(predicted).map(index).to_numpy()
    codes, _ = pd.factorize(pd.Series(blocks))
    cells = codes * 9 + a * 3 + p
    counts = np.bincount(cells, minlength=(codes.max() + 1) * 9)
    return counts.reshape(-1, 3, 3)


def paired_bootstrap(actual, pred_a, pred_b, blocks, n: int = N_BOOTSTRAP, seed: int = SEED):
    """IC 95 % de la diferencia B - A remuestreando bloques completos estación-día."""
    mats_a = block_matrices(actual, pred_a, blocks)
    mats_b = block_matrices(actual, pred_b, blocks)
    rng = np.random.default_rng(seed)
    n_blocks = len(mats_a)
    diffs = {metric: [] for metric in KEY_METRICS}
    for _ in range(n):
        weights = np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks)
        a = matrix_metrics(np.tensordot(weights, mats_a, axes=1))
        b = matrix_metrics(np.tensordot(weights, mats_b, axes=1))
        for metric in KEY_METRICS:
            diffs[metric].append(b[metric] - a[metric])
    return {
        metric: {
            'low': float(np.percentile(values, 2.5)),
            'high': float(np.percentile(values, 97.5)),
        }
        for metric, values in diffs.items()
    }


def mcnemar(actual, pred_a, pred_b) -> dict:
    """Prueba exacta de McNemar sobre los aciertos de cada fila."""
    actual = np.asarray(actual)
    right_a = np.asarray(pred_a) == actual
    right_b = np.asarray(pred_b) == actual
    only_a = int((right_a & ~right_b).sum())
    only_b = int((~right_a & right_b).sum())
    pvalue = binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {'only_a_right': only_a, 'only_b_right': only_b, 'pvalue': float(pvalue)}


def simulated_split(df: pd.DataFrame, pred_a, pred_b, seed: int = SEED) -> pd.DataFrame:
    """Reparte los bloques estación-día 50/50 y mide cada versión solo con su tráfico."""
    rng = np.random.default_rng(seed)
    blocks = df['block'].unique()
    to_b = set(blocks[rng.random(len(blocks)) < 0.5])
    in_b = df['block'].isin(to_b).to_numpy()
    rows = []
    for arm, mask, pred in (('A', ~in_b, pred_a), ('B', in_b, pred_b)):
        metrics, _, _ = class_statistics(df['risk_class'][mask], np.asarray(pred)[mask])
        rows.append({'arm': arm, 'n_samples': int(mask.sum()), **metrics})
    return pd.DataFrame(rows)


def decide(diff: dict, ci: dict) -> str:
    """B queda vigente si no pierde recall de déficit y mejora F1 o precisión con IC > 0."""
    keeps_recall = ci['recall_deficit']['low'] >= -RECALL_MARGIN
    improves = ci['f1_macro']['low'] > 0 or ci['precision_deficit']['low'] > 0
    return 'B' if keeps_recall and improves else 'A'


def compare_versions(df: pd.DataFrame, pred_a, pred_b) -> dict:
    """Métricas de A y B, diferencias, IC, McNemar, tráfico simulado y tabla por mes."""
    actual = df['risk_class']
    metrics = {}
    for name, pred in (('A', pred_a), ('B', pred_b)):
        values, _, _ = class_statistics(actual, pred)
        metrics[name] = {key: values[key] for key in REPORT_METRICS}
    diff = {key: metrics['B'][key] - metrics['A'][key] for key in KEY_METRICS}
    ci = paired_bootstrap(actual, pred_a, pred_b, df['block'])

    by_month = []
    for month, group in df.groupby('month'):
        for name, pred in (('A', pred_a), ('B', pred_b)):
            values, _, _ = class_statistics(group['risk_class'], np.asarray(pred)[group.index])
            by_month.append({'month': month, 'version': name, 'n_samples': len(group), **values})

    return {
        'n_samples': len(df),
        'n_blocks': int(df['block'].nunique()),
        'deficit_share': float((actual == 'deficit').mean()),
        'metrics': metrics,
        'diff_b_minus_a': diff,
        'ci95': ci,
        'mcnemar': mcnemar(actual, pred_a, pred_b),
        'simulated_split': simulated_split(df, pred_a, pred_b).to_dict(orient='records'),
        'by_month': pd.DataFrame(by_month),
        'decision': decide(diff, ci),
    }


def plot_comparison(result: dict, labels: dict, path) -> None:
    """Barras de A y B y diferencia B - A con su IC 95 %."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(KEY_METRICS))
    for offset, name in ((-0.2, 'A'), (0.2, 'B')):
        values = [result['metrics'][name][m] for m in KEY_METRICS]
        bars = left.bar(x + offset, values, 0.4, label=labels[name])
        left.bar_label(bars, fmt='%.3f', fontsize=8)
    left.set_xticks(x, KEY_METRICS)
    left.set_ylim(0, 1)
    left.set_title('Métricas en el mismo periodo')
    left.legend(fontsize=8)

    diffs = [result['diff_b_minus_a'][m] for m in KEY_METRICS]
    lows = [result['ci95'][m]['low'] for m in KEY_METRICS]
    highs = [result['ci95'][m]['high'] for m in KEY_METRICS]
    right.errorbar(
        x, diffs, yerr=[np.subtract(diffs, lows), np.subtract(highs, diffs)], fmt='o', capsize=6
    )
    right.axhline(0, color='grey', linewidth=1)
    right.axhline(-RECALL_MARGIN, color='red', linewidth=0.8, linestyle='--')
    right.set_xticks(x, KEY_METRICS)
    right.set_title('Diferencia B - A (IC 95 %, bootstrap por estación-día)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(actual, predictions, labels, name):
    """Genera una matriz de confusión por versión del modelo."""
    paths = []

    for arm in ('A', 'B'):
        y_pred = predictions[arm]

        cm = confusion_matrix(
            actual,
            y_pred,
            labels=list(RISK_LABELS),
        )

        fig, ax = plt.subplots(figsize=(7, 6))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=list(RISK_LABELS),
        )
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)

        ax.set_title(f'Matriz de confusión - {labels[arm]}')
        fig.tight_layout()

        path = OUTPUT_DIR / f'{name}_confusion_{arm}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)

        paths.append(path)

    return paths


def save_outputs(
    result: dict,
    info: dict,
    start: str,
    end: str,
    name: str,
    actual=None,
    predictions=None,
):
    """Escribe el resumen JSON, la tabla por mes y la gráfica en reports/ab_testing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / f'{name}_summary.json'
    month_path = OUTPUT_DIR / f'{name}_by_month.csv'
    plot_path = OUTPUT_DIR / f'{name}_metricas.png'
    summary = {key: value for key, value in result.items() if key != 'by_month'}
    summary.update({'period': {'start': start, 'end': end}, 'versions': info})
    summary['rule'] = (
        f'B si el límite inferior del IC de recall_deficit >= -{RECALL_MARGIN} '
        'y el límite inferior del IC de f1_macro o precision_deficit > 0'
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    result['by_month'].to_csv(month_path, index=False)
    labels = {k: f'{k}: v{v["version"]} ({v["alias"]})' for k, v in info.items()}
    plot_comparison(result, labels, plot_path)

    files = [summary_path, month_path, plot_path]

    if actual is not None and predictions is not None:
        files.extend(plot_confusion_matrices(actual, predictions, labels, name))

    return summary, files


def log_to_mlflow(client: MlflowClient, summary: dict, files: list, name: str) -> str:
    """Corrida de MLflow con las métricas del A/B y etiqueta de la decisión en la versión B."""
    info = summary['versions']
    with mlflow.start_run(run_name=f'p2-{name}') as run:
        mlflow.log_params(
            {
                'question': 'p2',
                'period_start': summary['period']['start'],
                'period_end': summary['period']['end'],
                'version_a': info['A']['version'],
                'version_b': info['B']['version'],
                'n_bootstrap': N_BOOTSTRAP,
                'random_seed': SEED,
                'recall_margin': RECALL_MARGIN,
            }
        )
        mlflow.set_tags({'stage_of_work': 'ab_test', 'decision': summary['decision']})
        for arm in ('A', 'B'):
            mlflow.log_metrics({f'{arm}_{k}': v for k, v in summary['metrics'][arm].items()})
        for metric in KEY_METRICS:
            mlflow.log_metrics(
                {
                    f'diff_{metric}': summary['diff_b_minus_a'][metric],
                    f'diff_{metric}_ci_low': summary['ci95'][metric]['low'],
                    f'diff_{metric}_ci_high': summary['ci95'][metric]['high'],
                }
            )
        mlflow.log_metric('mcnemar_pvalue', summary['mcnemar']['pvalue'])
        for path in files:
            mlflow.log_artifact(str(path), 'ab_testing')
        run_id = run.info.run_id
    client.set_model_version_tag(MODEL_NAME, info['B']['version'], 'ab_test_run_id', run_id)
    client.set_model_version_tag(
        MODEL_NAME, info['B']['version'], 'ab_test_decision', summary['decision']
    )
    return run_id


@app.command()
def main(
    challenger_alias: str = typer.Option('retrain-jun', help='Alias de la versión B'),
    champion_alias: str = typer.Option('champion', help='Alias de la versión A'),
    start: str = typer.Option('2026-07-01', help='Primer día del periodo de evaluación'),
    end: str = typer.Option('2026-09-30', help='Último día del periodo de evaluación'),
    name: str = typer.Option('ab_test', help='Prefijo de los archivos de salida'),
    promote: bool = typer.Option(False, help='Mover @champion a B si gana'),
):
    """Compara dos versiones de P2 sobre el mismo periodo y registra el resultado."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if not os.environ.get('MLFLOW_RUN_ID'):
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(MLFLOW_TRACKING_URI)
    info = {'A': version_info(client, champion_alias), 'B': version_info(client, challenger_alias)}
    for version in info.values():
        check_unseen(version, start)

    df = load_period(start, end)
    preds = {}
    for arm, version in info.items():
        model = mlflow.pyfunc.load_model(f'models:/{MODEL_NAME}/{version["version"]}')
        preds[arm] = np.asarray(model.predict(df[P2_FEATURES]))
    logger.info(
        f'{len(df):,} filas de {start} a {end}; A=v{info["A"]["version"]}, B=v{info["B"]["version"]}'
    )

    result = compare_versions(df, preds['A'], preds['B'])
    summary, files = save_outputs(
        result, info, start, end, name, actual=df['risk_class'], predictions=preds
    )
    run_id = log_to_mlflow(client, summary, files, name)
    for metric in KEY_METRICS:
        ci = summary['ci95'][metric]
        logger.info(
            f'{metric}: A={summary["metrics"]["A"][metric]:.4f} '
            f'B={summary["metrics"]["B"][metric]:.4f} '
            f'B-A={summary["diff_b_minus_a"][metric]:+.4f} [{ci["low"]:+.4f}, {ci["high"]:+.4f}]'
        )
    logger.success(f'Decisión: queda la versión {summary["decision"]} (corrida {run_id})')

    if promote and summary['decision'] == 'B':
        client.set_registered_model_alias(MODEL_NAME, 'champion', info['B']['version'])
        logger.success(f'@champion -> v{info["B"]["version"]}')


if __name__ == '__main__':
    app()
