
"""Pruebas automatizadas para el análisis de data drift."""

import numpy as np
import pandas as pd
import pytest

from mi_bici.validation import drift_validation
from mi_bici.validation.drift_validation import (
    apply_drift_policy,
    enforce_drift_policy,
    sample_data,
    validate_drift,
)


def test_sample_data_reproducible():
    """El muestreo debe producir los mismos registros."""

    df = pd.DataFrame({"value": range(6000)})

    sample1 = sample_data(df)
    sample2 = sample_data(df)

    assert len(sample1) == 5000
    pd.testing.assert_frame_equal(sample1, sample2)


def test_validate_drift(tmp_path, monkeypatch):
    """DeepChecks debe devolver resultados válidos."""

    monkeypatch.setattr(drift_validation, 'OUTPUT_DIR', tmp_path)
    rng = np.random.default_rng(42)

    train = pd.DataFrame({
        "trips": rng.normal(100, 15, 100),
        "capacity": rng.integers(10, 50, 100),
    })

    test = pd.DataFrame({
        "trips": rng.normal(110, 15, 100),
        "capacity": rng.integers(10, 50, 100),
    })

    result = validate_drift(
        "TEST",
        train,
        test,
        categorical_features=[],
    )

    assert not result.empty
    assert set(result["feature"]) == {"trips", "capacity"}
    assert result["drift_score"].notna().all()
    assert np.isfinite(result["drift_score"]).all()


@pytest.mark.parametrize('method', ['Kolmogorov-Smirnov', "Cramer's V"])
def test_drift_thresholds_and_invalid_scores(method):
    scores = [0., .1999, .2, .4999, .5, 1., np.nan, np.inf, -.1, 1.1]
    report = pd.DataFrame({
        'dataset': 'P1', 'feature': 'value', 'method': method, 'drift_score': scores,
    })
    result = apply_drift_policy(report)
    assert result['status'].tolist() == [
        'pass', 'pass', 'warning', 'warning', 'error', 'error',
        'error', 'error', 'error', 'error',
    ]
    assert (result['warning_threshold'] == .2).all()
    assert (result['error_threshold'] == .5).all()


@pytest.mark.parametrize('warning,error', [(-.1, .5), (.5, .5), (.6, .5), (0., 1.1), (np.nan, .5)])
def test_invalid_drift_policy_is_rejected(warning, error):
    report = pd.DataFrame({'method': ['Kolmogorov-Smirnov'], 'drift_score': [.3]})
    with pytest.raises(ValueError, match='Umbrales inválidos'):
        apply_drift_policy(report, {'Kolmogorov-Smirnov': {'warning': warning, 'error': error}})


def test_drift_policy_can_be_configured_per_method():
    report = pd.DataFrame({'method': ['Kolmogorov-Smirnov', "Cramer's V"], 'drift_score': [.3, .3]})
    limits = {
        'Kolmogorov-Smirnov': {'warning': .1, 'error': .3},
        "Cramer's V": {'warning': .4, 'error': .6},
    }
    assert apply_drift_policy(report, limits)['status'].tolist() == ['error', 'pass']


def test_unknown_drift_method_is_not_silently_accepted():
    report = pd.DataFrame({'method': ['desconocido'], 'drift_score': [.1]})
    with pytest.raises(ValueError, match='No hay umbrales'):
        apply_drift_policy(report)


def test_empty_drift_report_is_rejected():
    with pytest.raises(ValueError, match='vacío'):
        apply_drift_policy(pd.DataFrame())


def test_warning_does_not_stop_validation(capsys):
    report = apply_drift_policy(pd.DataFrame({
        'dataset': ['P1'], 'feature': ['month'],
        'method': ['Kolmogorov-Smirnov'], 'drift_score': [.3],
    }))
    enforce_drift_policy(report)
    assert 'ADVERTENCIA: P1.month' in capsys.readouterr().out


def test_shifted_distributions_block_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_validation, 'OUTPUT_DIR', tmp_path)
    rng = np.random.default_rng(42)
    report = validate_drift(
        'TEST', pd.DataFrame({'value': rng.normal(0, 1, 100)}),
        pd.DataFrame({'value': rng.normal(100, 1, 100)}), categorical_features=[],
    )
    assert report.iloc[0]['status'] == 'error'
    with pytest.raises(SystemExit, match='TEST.value'):
        enforce_drift_policy(report)
    assert (tmp_path / 'test_drift_report.html').is_file()


@pytest.mark.parametrize('score,blocked', [(.1, False), (.2, False), (.5, True), (np.nan, True)])
def test_drift_command_saves_report_before_applying_gate(
    valid_feature_frames, tmp_path, monkeypatch, score, blocked
):
    paths = {
        'P1': 'P1_FEATURES_PATH', 'P2': 'P2_FEATURES_PATH',
        'P3_train': 'P3_TRAIN_FEATURES_PATH', 'P3_holdout': 'P3_HOLDOUT_FEATURES_PATH',
    }
    for name, attribute in paths.items():
        path = tmp_path / f'{name}.parquet'
        valid_feature_frames[name].to_parquet(path, index=False)
        monkeypatch.setattr(drift_validation, attribute, path)
    output = tmp_path / 'reports'
    monkeypatch.setattr(drift_validation, 'OUTPUT_DIR', output)

    def known_drift(name, *args, **kwargs):
        return apply_drift_policy(pd.DataFrame({
            'dataset': [name], 'feature': ['capacity'],
            'method': ['Kolmogorov-Smirnov'], 'drift_score': [score],
        }))

    monkeypatch.setattr(drift_validation, 'validate_drift', known_drift)
    if blocked:
        with pytest.raises(SystemExit, match='Drift no aceptable'):
            drift_validation.main()
    else:
        drift_validation.main()
    report = pd.read_csv(output / 'drift_summary.csv')
    assert set(report['dataset']) == {'P1', 'P2', 'P3'}
    assert ((report['status'] == 'error').any()) == blocked
