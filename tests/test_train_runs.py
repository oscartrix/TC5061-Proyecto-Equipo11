import mlflow
import numpy as np
import pandas as pd
import pytest

from mi_bici.config import MODEL_DEFAULTS
from mi_bici.modeling import train


def synthetic_p1(n=300):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            'station_id': rng.integers(1, 20, n),
            'date': pd.date_range('2025-01-01', periods=n, freq='D'),
            'trips_count': rng.integers(0, 40, n),
            'capacity': 20,
            'zone': rng.choice(['centro', 'norte'], n),
            'month': 1,
            'is_weekend': rng.integers(0, 2, n),
            'is_holiday': 0,
            'day_of_week': rng.integers(0, 7, n),
            'split': ['train'] * 200 + ['val'] * 50 + ['test'] * 50,
        }
    )
    for col in ['lag_1d', 'lag_7d', 'lag_14d', 'lag_21d', 'rolling_mean_7d']:
        df[col] = rng.random(n) * 30
    return df


def synthetic_p2(n=300):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({col: rng.random(n) for col in train.P2_FEATURES})
    df['risk_class'] = rng.choice(['deficit', 'balanceado', 'excedente'], n)
    df['risk_lag_168h'] = rng.choice(['deficit', 'balanceado', 'excedente'], n)
    df['split'] = ['train'] * 200 + ['val'] * 50 + ['test'] * 50
    return df


def synthetic_p3(n=60):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({col: rng.random(n) for col in train.P3_USAGE_COLS})
    df['station_id'] = np.arange(n)
    df['latitude'] = 20.6 + rng.random(n) / 10
    df['longitude'] = -103.3 + rng.random(n) / 10
    return df


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    for name, builder in [('P1_FEATURES_PATH', synthetic_p1), ('P2_FEATURES_PATH', synthetic_p2)]:
        path = tmp_path / f'{name}.parquet'
        builder().to_parquet(path)
        monkeypatch.setattr(train, name, path)
    for name in ['P3_TRAIN_FEATURES_PATH', 'P3_HOLDOUT_FEATURES_PATH']:
        path = tmp_path / f'{name}.parquet'
        synthetic_p3().to_parquet(path)
        monkeypatch.setattr(train, name, path)

    mlflow.set_tracking_uri(f'sqlite:///{tmp_path / "test.db"}')
    experiment_id = mlflow.create_experiment('test', artifact_location=(tmp_path / 'art').as_uri())
    mlflow.set_experiment(experiment_id=experiment_id)
    return mlflow.MlflowClient()


CASES = [
    (question, model)
    for question, models in MODEL_DEFAULTS.items()
    for model in models
]


@pytest.mark.parametrize(('question', 'model'), CASES)
def test_run_logs_tags_and_artifacts(tracking, question, model):
    hparams = MODEL_DEFAULTS[question][model]
    with mlflow.start_run() as run:
        train.RUNNERS[question](model, hparams)

    data = tracking.get_run(run.info.run_id).data
    stage = 'baseline' if model == 'baseline' else 'candidate'
    assert data.tags['stage_of_work'] == stage
    assert data.tags['model_family'] == model
    assert data.tags['question'] == question

    artifacts = {a.path for a in tracking.list_artifacts(run.info.run_id)}
    assert {'plots', 'data', 'config'} <= artifacts

    models = tracking.search_logged_models(
        experiment_ids=[run.info.experiment_id], filter_string=f"source_run_id='{run.info.run_id}'"
    )
    assert len(models) == (0 if model == 'baseline' else 1)
    if models:
        info = mlflow.models.get_model_info(f'models:/{models[0].model_id}')
        assert info.signature is not None
