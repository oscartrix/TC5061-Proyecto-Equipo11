"""Pruebas del comando de calidad y sus archivos de entrada requeridos."""

import pandas as pd
import pytest

from mi_bici.validation import run_validation


DATASET_PATHS = {
    'P1': 'P1_FEATURES_PATH',
    'P2': 'P2_FEATURES_PATH',
    'P3_train': 'P3_TRAIN_FEATURES_PATH',
    'P3_holdout': 'P3_HOLDOUT_FEATURES_PATH',
}


@pytest.fixture
def validation_files(tmp_path, monkeypatch):
    paths = {}
    df = pd.DataFrame({'station_id': range(1, 7), 'capacity': range(20, 26)})
    for name, attribute in DATASET_PATHS.items():
        path = tmp_path / f'{name}.parquet'
        df.to_parquet(path, index=False)
        monkeypatch.setattr(run_validation, attribute, path)
        paths[name] = path
    reports_dir = tmp_path / 'reports'
    monkeypatch.setattr(run_validation, 'REPORTS_DIR', reports_dir)
    return paths, reports_dir


@pytest.mark.parametrize(
    'missing',
    [tuple(DATASET_PATHS), *((name,) for name in DATASET_PATHS)],
    ids=['todos', *DATASET_PATHS],
)
def test_missing_datasets_fail_before_validation(validation_files, monkeypatch, missing):
    paths, reports_dir = validation_files
    for name in missing:
        paths[name].unlink()

    def unexpected_validation(*args, **kwargs):
        pytest.fail('No validar datos si falta un archivo requerido')

    monkeypatch.setattr(run_validation, 'validate_data_quality', unexpected_validation)

    with pytest.raises(SystemExit) as error:
        run_validation.main()

    assert error.value.code != 0
    message = str(error.value)
    for name in missing:
        assert name in message
        assert str(paths[name]) in message
    assert 'ingesta' in message
    assert 'generación de características' in message
    assert not reports_dir.exists()


def test_dataset_path_must_be_a_file(validation_files):
    paths, reports_dir = validation_files
    paths['P1'].unlink()
    paths['P1'].mkdir()

    with pytest.raises(SystemExit, match=str(paths['P1'])):
        run_validation.main()

    assert not reports_dir.exists()


def test_all_required_datasets_produce_quality_report(validation_files):
    _, reports_dir = validation_files

    run_validation.main()

    report = pd.read_csv(reports_dir / 'quality_summary.csv')
    assert set(report['dataset']) == set(DATASET_PATHS)
    assert (report['total_rows'] == 6).all()
    assert (report['missing_count'] == 0).all()
    assert (report['duplicate_rows'] == 0).all()
