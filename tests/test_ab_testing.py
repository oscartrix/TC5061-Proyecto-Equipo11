"""Pruebas del A/B testing entre versiones de P2."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mi_bici.monitoring.ab_testing import (
    block_matrices,
    check_unseen,
    compare_versions,
    decide,
    matrix_metrics,
    mcnemar,
    paired_bootstrap,
    simulated_split,
    version_info,
)

LABELS = np.array(['deficit', 'balanceado', 'excedente'])


@pytest.fixture
def period():
    rng = np.random.default_rng(0)
    n = 600
    actual = LABELS[rng.integers(0, 3, n)]
    df = pd.DataFrame(
        {
            'risk_class': actual,
            'block': [f's{i % 10}_d{i // 60}' for i in range(n)],
            'month': np.where(np.arange(n) < 300, '2026-07', '2026-08'),
        }
    )
    pred_a = np.where(rng.random(n) < 0.6, actual, LABELS[rng.integers(0, 3, n)])
    return df, actual, pred_a


def ci(f1=(0.0, 0.0), recall=(0.0, 0.0), precision=(0.0, 0.0)):
    return {
        'f1_macro': dict(zip(('low', 'high'), f1)),
        'recall_deficit': dict(zip(('low', 'high'), recall)),
        'precision_deficit': dict(zip(('low', 'high'), precision)),
    }


def test_matrix_metrics_con_prediccion_perfecta():
    metrics = matrix_metrics(np.diag([5, 3, 2]))

    assert metrics == {'f1_macro': 1.0, 'recall_deficit': 1.0, 'precision_deficit': 1.0}


def test_block_matrices_suma_todas_las_filas(period):
    df, actual, pred_a = period

    mats = block_matrices(actual, pred_a, df['block'])

    assert mats.shape == (df['block'].nunique(), 3, 3)
    assert mats.sum() == len(df)


def test_bootstrap_de_versiones_iguales_da_diferencia_cero(period):
    df, actual, pred_a = period

    result = paired_bootstrap(actual, pred_a, pred_a, df['block'], n=50)

    assert all(v == {'low': 0.0, 'high': 0.0} for v in result.values())


def test_bootstrap_detecta_una_version_mejor(period):
    df, actual, pred_a = period

    result = paired_bootstrap(actual, pred_a, actual, df['block'], n=200)

    assert result['f1_macro']['low'] > 0
    assert result['recall_deficit']['low'] > 0


def test_bootstrap_es_reproducible(period):
    df, actual, pred_a = period
    pred_b = np.roll(pred_a, 1)

    first = paired_bootstrap(actual, pred_a, pred_b, df['block'], n=30)
    second = paired_bootstrap(actual, pred_a, pred_b, df['block'], n=30)

    assert first == second


def test_mcnemar_cuenta_solo_los_desacuerdos():
    actual = np.array(['deficit', 'deficit', 'balanceado', 'excedente'])
    pred_a = np.array(['deficit', 'balanceado', 'balanceado', 'deficit'])
    pred_b = np.array(['deficit', 'deficit', 'excedente', 'deficit'])

    result = mcnemar(actual, pred_a, pred_b)

    assert result['only_a_right'] == 1
    assert result['only_b_right'] == 1
    assert result['pvalue'] == pytest.approx(1.0)


def test_simulated_split_reparte_bloques_completos(period):
    df, actual, pred_a = period

    split = simulated_split(df, pred_a, actual)

    assert split['n_samples'].sum() == len(df)
    assert set(split['arm']) == {'A', 'B'}
    assert split.equals(simulated_split(df, pred_a, actual))


@pytest.mark.parametrize(
    'interval, expected',
    [
        (ci(f1=(0.001, 0.01)), 'B'),
        (ci(precision=(0.002, 0.02)), 'B'),
        (ci(f1=(-0.001, 0.01), recall=(0.01, 0.02)), 'A'),
        (ci(f1=(0.001, 0.01), recall=(-0.02, 0.0)), 'A'),
    ],
)
def test_decide_aplica_la_regla(interval, expected):
    assert decide({}, interval) == expected


def test_check_unseen_rechaza_versiones_que_vieron_el_periodo():
    seen = {'alias': 'retrain', 'version': '4', 'train_end': '2026-08-31'}

    with pytest.raises(ValueError, match='no se puede evaluar'):
        check_unseen(seen, '2026-07-01')
    check_unseen(seen, '2026-09-01')


def test_version_info_lee_train_end_de_la_corrida():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version='3', run_id='r1')
    client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(params={'model': 'hgb_balanced', 'train_end': '2026-06-30'})
    )

    info = version_info(client, 'retrain-jun')

    client.get_model_version_by_alias.assert_called_once_with(
        'mibici-equipo11-p2-riesgo', 'retrain-jun'
    )
    assert info == {
        'alias': 'retrain-jun',
        'version': '3',
        'run_id': 'r1',
        'model': 'hgb_balanced',
        'train_end': '2026-06-30',
    }


def test_compare_versions_reune_metricas_y_meses(period):
    df, actual, pred_a = period

    result = compare_versions(df, pred_a, actual)

    assert result['n_samples'] == len(df)
    assert result['metrics']['B']['f1_macro'] == 1.0
    assert result['diff_b_minus_a']['f1_macro'] > 0
    assert set(result['by_month']['month']) == {'2026-07', '2026-08'}
    assert len(result['by_month']) == 4
