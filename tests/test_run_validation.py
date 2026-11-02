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
def validation_files(tmp_path, monkeypatch, valid_feature_frames):
    paths = {}
    for name, attribute in DATASET_PATHS.items():
        path = tmp_path / f'{name}.parquet'
        valid_feature_frames[name].to_parquet(path, index=False)
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
    assert (report['status'] == 'pass').all()
    assert pd.read_csv(reports_dir / 'quality_errors.csv').empty


@pytest.mark.parametrize('defect', ['column', 'type', 'range', 'empty'])
def test_contract_errors_stop_command_and_are_reported(validation_files, defect):
    paths, reports_dir = validation_files
    df = pd.read_parquet(paths['P1'])
    if defect == 'column':
        df = df.drop(columns='capacity')
    elif defect == 'type':
        df['capacity'] = df['capacity'].astype(str)
    elif defect == 'range':
        df.loc[0, 'capacity'] = 0
    else:
        df = df.iloc[:0]
    df.to_parquet(paths['P1'], index=False)

    with pytest.raises(SystemExit, match='Problemas de calidad'):
        run_validation.main()

    errors = pd.read_csv(reports_dir / 'quality_errors.csv')
    assert set(errors['dataset']) == {'P1'}
    summary = pd.read_csv(reports_dir / 'quality_summary.csv')
    assert (summary.loc[summary.dataset == 'P1', 'status'] == 'error').all()


def test_contract_checks_rows_outside_deepchecks_sample(validation_files):
    paths, reports_dir = validation_files
    row = pd.read_parquet(paths['P1']).iloc[[0]]
    df = pd.concat([row] * 5001, ignore_index=True)
    df['station_id'] = range(1, len(df) + 1)
    excluded = df.index.difference(df.sample(n=5000, random_state=42).index)[0]
    df.loc[excluded, 'trips_count'] = -1
    df.to_parquet(paths['P1'], index=False)

    with pytest.raises(SystemExit, match='trips_count'):
        run_validation.main()

    assert 'trips_count' in pd.read_csv(reports_dir / 'quality_errors.csv')['error'].str.cat()


def test_failed_deepchecks_condition_stops_command(validation_files, monkeypatch):
    _, reports_dir = validation_files
    monkeypatch.setattr(run_validation, 'validate_data_quality', lambda df: {
        'errors': ['DeepChecks mixed_nulls: condición de aceptación incumplida']
    })

    with pytest.raises(SystemExit, match='DeepChecks mixed_nulls'):
        run_validation.main()

    assert len(pd.read_csv(reports_dir / 'quality_errors.csv')) == 4
