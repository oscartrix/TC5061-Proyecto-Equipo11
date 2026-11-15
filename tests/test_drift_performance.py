"""Pruebas de la relación entre drift mensual y desempeño del champion de P2."""

import numpy as np
import pandas as pd
import pytest

from mi_bici.modeling.train import P2_FEATURES
from mi_bici.monitoring.drift_performance import (
    PERFORMANCE_COLUMNS,
    correlations,
    drift_vs_performance,
    main,
    monthly_drift,
)


def p2_frame(start, periods, split, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({feat: rng.normal(shift, 1, periods) for feat in P2_FEATURES})
    df['hour'] = rng.integers(0, 24, periods)
    df['day_of_week'] = rng.integers(0, 7, periods)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['station_id'] = 1
    df['hour_ts'] = pd.date_range(start, periods=periods, freq='h')
    df['risk_class'] = rng.choice(['deficit', 'balanceado', 'excedente'], periods)
    df['prediction'] = rng.choice(['deficit', 'balanceado', 'excedente'], periods)
    df['split'] = split
    return df


def by_month_table(months):
    return pd.DataFrame(
        {
            'month': months,
            **{column: np.linspace(0.5, 0.6, len(months)) for column in PERFORMANCE_COLUMNS},
        }
    )


def test_monthly_drift_por_mes_de_val_y_test():
    frame = pd.concat(
        [
            p2_frame('2025-01-01', 200, 'train', seed=1),
            p2_frame('2026-01-01', 100, 'val', seed=2),
            p2_frame('2026-07-01', 100, 'test', shift=3.0, seed=3),
        ],
        ignore_index=True,
    )

    drift = monthly_drift(frame)

    assert list(drift.index) == ['2026-01', '2026-07']
    assert list(drift['split']) == ['val', 'test']
    assert 'capacity' not in drift.columns
    assert drift.loc['2026-07', 'departures'] > drift.loc['2026-01', 'departures']
    assert drift['n_drifted'].dtype.kind == 'i'


def test_drift_vs_performance_une_por_mes():
    drift = pd.DataFrame(
        {'split': ['val', 'test'], 'n_drifted': [1, 3], 'departures': [0.05, 0.2]},
        index=pd.Index(['2026-01', '2026-07'], name='month'),
    )

    table = drift_vs_performance(drift, by_month_table(['2026-01', '2026-07', '2026-08']))

    assert list(table.index) == ['2026-01', '2026-07']
    assert set(PERFORMANCE_COLUMNS) <= set(table.columns)


def test_correlations_entre_drift_y_metricas():
    table = pd.DataFrame(
        {
            'split': ['val', 'val', 'test'],
            'n_drifted': [1, 2, 3],
            'departures': [0.1, 0.2, 0.3],
            **{column: [0.6, 0.5, 0.4] for column in PERFORMANCE_COLUMNS},
        }
    )

    result = correlations(table)

    assert list(result.index) == ['n_drifted', 'departures']
    assert result.loc['departures', 'model_f1_macro'] == pytest.approx(-1.0)


def test_main_falla_sin_by_month(tmp_path):
    predictions = tmp_path / 'p2_predictions.parquet'
    p2_frame('2025-01-01', 10, 'train').to_parquet(predictions)

    with pytest.raises(FileNotFoundError, match='by_month'):
        main(predictions, tmp_path / 'by_month.csv', tmp_path / 'salida.csv')
