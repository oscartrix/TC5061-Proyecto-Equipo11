"""Pruebas del reentrenamiento de P2 con una ventana nueva."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from typer.testing import CliRunner

from mi_bici.modeling.retrain import (
    app,
    champion_spec,
    label_cutoff,
    training_window,
)


def test_label_cutoff_deja_fuera_las_ultimas_tres_horas():
    assert label_cutoff('2026-08-31') == pd.Timestamp('2026-08-31 20:00')


def test_training_window_usa_hour_ts_y_no_split():
    hours = pd.date_range('2026-08-31 18:00', '2026-09-01 02:00', freq='h')
    df = pd.DataFrame({'hour_ts': hours, 'split': 'test'})

    window = training_window(df, '2026-08-31')

    assert window['hour_ts'].max() == pd.Timestamp('2026-08-31 20:00')
    assert len(window) == 3
    assert (window['split'] == 'test').all()


def test_champion_spec_lee_familia_e_hiperparametros_del_champion():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version='1', run_id='abc')
    client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(
            params={
                'model': 'hgb_balanced',
                'max_iter': '150',
                'learning_rate': '0.08',
                'max_depth': '6',
                'train_end': '2025-12-31',
            }
        )
    )

    version, family, hparams = champion_spec(client)

    client.get_model_version_by_alias.assert_called_once_with(
        'mibici-equipo11-p2-riesgo', 'champion'
    )
    assert version.version == '1'
    assert family == 'hgb_balanced'
    assert hparams == {'max_iter': 150, 'learning_rate': 0.08, 'max_depth': 6}


@pytest.mark.parametrize('alias', ['champion', 'challenger'])
def test_main_rechaza_aliases_de_register(alias):
    result = CliRunner().invoke(app, ['--alias', alias])

    assert result.exit_code != 0
    assert 'register.py' in result.output
