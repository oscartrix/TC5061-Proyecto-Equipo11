"""Relaciona el drift mensual de P2 con el desempeño del champion en cada mes.

Compara la referencia de entrenamiento contra cada mes de validación y prueba (ene-sep 2026)
y une el resultado con ``by_month.csv``, que genera ``mlflow run . -e evaluate -P question=p2``.
"""

from pathlib import Path

from loguru import logger
import pandas as pd

from mi_bici.config import REPORTS_DIR
from mi_bici.modeling.train import plain_dtypes
from mi_bici.monitoring.evidently_drift import (
    OUTPUT_DIR,
    P2_PREDICTIONS_PATH,
    STATIC_COLUMNS,
    compare,
    sample_rows,
)

BY_MONTH_PATH = REPORTS_DIR / 'evaluation' / 'p2' / 'by_month.csv'
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / 'drift_vs_desempeno_p2.csv'
PERFORMANCE_COLUMNS = [
    'model_f1_macro',
    'model_recall_deficit',
    'model_precision_deficit',
    'model_false_deficit_alerts',
    'n_deficit',
    'n_samples',
]


def monthly_drift(
    frame: pd.DataFrame, sample_size: int = 5000, random_state: int = 42
) -> pd.DataFrame:
    """Score de drift por variable de cada mes de val y test contra la referencia train."""
    reference = sample_rows(frame[frame['split'] == 'train'], sample_size, random_state)
    later = frame[frame['split'].isin(['val', 'test'])]
    months = later['hour_ts'].dt.to_period('M')
    rows = {}
    for month in sorted(months.unique()):
        current = sample_rows(later[months == month], sample_size, random_state)
        comparison = compare(reference, current)
        rows[str(month)] = {
            'split': current['split'].iloc[0],
            'n_drifted': len(comparison['drifted_columns']),
            **{
                column: detail['score']
                for column, detail in comparison['columns'].items()
                if column not in STATIC_COLUMNS
            },
        }
    return pd.DataFrame.from_dict(rows, orient='index').rename_axis('month')


def drift_vs_performance(drift: pd.DataFrame, by_month: pd.DataFrame) -> pd.DataFrame:
    """Une el drift mensual con las métricas del champion del mismo mes."""
    performance = by_month.set_index('month')[PERFORMANCE_COLUMNS]
    return drift.join(performance, how='inner')


def correlations(table: pd.DataFrame) -> pd.DataFrame:
    """Correlación de Pearson entre los scores de drift y las métricas del champion."""
    drift_columns = [
        c for c in table.columns if c not in PERFORMANCE_COLUMNS and c not in ('split',)
    ]
    metrics = ['model_f1_macro', 'model_recall_deficit', 'model_precision_deficit']
    return table[drift_columns + metrics].astype(float).corr().loc[drift_columns, metrics]


def main(
    predictions_path: Path = P2_PREDICTIONS_PATH,
    by_month_path: Path = BY_MONTH_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    for path in (predictions_path, by_month_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f'No se encontró {path}')

    frame = plain_dtypes(pd.read_parquet(predictions_path))
    table = drift_vs_performance(monthly_drift(frame), pd.read_csv(by_month_path))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path)
    logger.info(f'Tabla de drift vs desempeño guardada en: {output_path}')

    columns = ['split', 'n_drifted', 'pressure_last_3h', 'departures', 'risk_class']
    print(table[columns + PERFORMANCE_COLUMNS[:3]].round(3).to_string())
    print()
    print('Correlación entre drift y métricas del champion (Pearson, 9 meses):')
    print(correlations(table).round(2).to_string())
    return table


if __name__ == '__main__':
    main()
