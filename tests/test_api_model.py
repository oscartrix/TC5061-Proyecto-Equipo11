"""Pruebas de la carga del modelo desde un registry temporal."""

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mi_bici.api import model as served
from mi_bici.api.model import (
    DEFAULT_MODEL_ALIAS,
    DEFAULT_MODEL_NAME,
    ModelLoadError,
    ModelSettings,
    align_features,
    load_served_model,
)
from mi_bici.api.schemas import EXAMPLE_FEATURES, P1Features
from mi_bici.modeling.train import P1_CAT_COLS, P1_NUM_COLS

COLUMNS = P1_NUM_COLS + P1_CAT_COLS
NAME = 'modelo-prueba'
NO_SIGNATURE_NAME = 'modelo-sin-firma'


def training_frame() -> pd.DataFrame:
    """Mismos tipos que la tabla real de P1. capacity double y calendario int32."""
    rows = [
        P1Features.model_validate({
            **EXAMPLE_FEATURES,
            'lag_1d': float(i),
            'capacity': 10 + i,
            'zone': ['GDL', 'ZPN', 'TLQ'][i % 3],
        }).model_dump()
        for i in range(12)
    ]
    return pd.DataFrame(rows)[COLUMNS].astype({
        'capacity': 'float64',
        'day_of_week': 'int32',
        'month': 'int32',
    })


def fit_pipeline(x: pd.DataFrame, slope: float) -> Pipeline:
    prep = ColumnTransformer([
        ('num', 'passthrough', P1_NUM_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), P1_CAT_COLS),
    ])
    return Pipeline([('prep', prep), ('reg', LinearRegression())]).fit(x, slope * x['lag_1d'])


@pytest.fixture(scope='module')
def registry(tmp_path_factory):
    root = tmp_path_factory.mktemp('api-registry')
    uri = f'sqlite:///{root / "mlflow.db"}'
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment(
        'api', artifact_location=(root / 'artifacts').as_uri()
    )
    client = mlflow.MlflowClient()
    x = training_frame()
    pipelines = {}
    run_ids = {}
    try:
        for version, slope in {'1': 2.0, '2': 3.0}.items():
            pipe = fit_pipeline(x, slope)
            with mlflow.start_run(experiment_id=experiment_id) as run:
                info = mlflow.sklearn.log_model(
                    pipe,
                    name='model',
                    signature=infer_signature(x, pipe.predict(x)),
                    serialization_format='cloudpickle',
                )
            mlflow.register_model(info.model_uri, NAME)
            pipelines[version] = pipe
            run_ids[version] = run.info.run_id
        client.set_registered_model_alias(NAME, 'champion', '1')
        client.set_registered_model_alias(NAME, 'challenger', '2')

        with mlflow.start_run(experiment_id=experiment_id):
            info = mlflow.sklearn.log_model(
                pipelines['1'], name='model', serialization_format='cloudpickle'
            )
        mlflow.register_model(info.model_uri, NO_SIGNATURE_NAME)
        client.set_registered_model_alias(NO_SIGNATURE_NAME, 'champion', '1')
        yield uri, pipelines, run_ids
    finally:
        mlflow.set_tracking_uri(previous_uri)


@pytest.fixture
def tracking(registry, monkeypatch):
    uri, pipelines, run_ids = registry
    monkeypatch.setattr(served, 'MLFLOW_TRACKING_URI', uri)
    return pipelines, run_ids


def request_frame() -> pd.DataFrame:
    """Columnas como las genera el request."""
    row = P1Features.model_validate(EXAMPLE_FEATURES).model_dump()
    return pd.DataFrame([row])[list(reversed(COLUMNS))]


def test_settings_defaults(monkeypatch):
    for variable in ('MODEL_NAME', 'MODEL_ALIAS', 'MODEL_VERSION'):
        monkeypatch.delenv(variable, raising=False)
    settings = ModelSettings.from_env()
    assert settings == ModelSettings(DEFAULT_MODEL_NAME, DEFAULT_MODEL_ALIAS, None)
    assert settings.reference == 'mibici-equipo11-p1-demanda@champion'


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv('MODEL_NAME', NAME)
    monkeypatch.setenv('MODEL_ALIAS', 'challenger')
    monkeypatch.setenv('MODEL_VERSION', '2')
    settings = ModelSettings.from_env()
    assert settings == ModelSettings(NAME, 'challenger', '2')
    assert settings.reference == f'{NAME}/2'


def test_empty_env_variables_use_defaults(monkeypatch):
    for variable in ('MODEL_NAME', 'MODEL_ALIAS', 'MODEL_VERSION'):
        monkeypatch.setenv(variable, '')
    assert ModelSettings.from_env() == ModelSettings()


@pytest.mark.parametrize('settings,version', [
    (ModelSettings(NAME, 'champion'), '1'),
    (ModelSettings(NAME, 'challenger'), '2'),
    (ModelSettings(NAME, version='2'), '2'),
    (ModelSettings(NAME, 'champion', '2'), '2'),
])
def test_load_resolves_concrete_version(tracking, settings, version):
    pipelines, run_ids = tracking
    loaded = load_served_model(settings)

    assert (loaded.name, loaded.version, loaded.run_id) == (NAME, version, run_ids[version])
    features = request_frame()
    expected = pipelines[version].predict(features[COLUMNS])
    np.testing.assert_allclose(loaded.predict(features), expected)


def test_load_uses_env_settings_by_default(tracking, monkeypatch):
    monkeypatch.setenv('MODEL_NAME', NAME)
    monkeypatch.delenv('MODEL_ALIAS', raising=False)
    monkeypatch.setenv('MODEL_VERSION', '2')
    assert load_served_model().version == '2'


def test_request_types_need_alignment_and_are_aligned(tracking):
    loaded = load_served_model(ModelSettings(NAME))
    features = request_frame()

    with pytest.raises(MlflowException):
        loaded.model.predict(features)

    aligned = align_features(features, loaded.model)
    assert aligned['day_of_week'].dtype == 'int32'
    assert aligned['capacity'].dtype == 'float64'
    assert features['day_of_week'].dtype == 'int64'
    assert loaded.predict(features).shape == (1,)


def test_predict_requires_model_columns(tracking):
    loaded = load_served_model(ModelSettings(NAME))
    with pytest.raises(ValueError, match='capacity'):
        loaded.predict(request_frame().drop(columns='capacity'))


@pytest.mark.parametrize('settings', [
    ModelSettings('modelo-inexistente'),
    ModelSettings(NAME, 'alias-inexistente'),
    ModelSettings(NAME, version='99'),
    ModelSettings(NAME, version='abc'),
])
def test_missing_model_raises_clear_error(tracking, settings):
    with pytest.raises(ModelLoadError, match='No se pudo cargar') as error:
        load_served_model(settings)
    assert settings.reference in str(error.value)


def test_model_without_signature_is_rejected(tracking):
    with pytest.raises(ModelLoadError, match='no tiene una firma'):
        load_served_model(ModelSettings(NO_SIGNATURE_NAME))


def test_unreachable_server_raises_clear_error(monkeypatch):
    uri = 'http://127.0.0.1:1'
    monkeypatch.setattr(served, 'MLFLOW_TRACKING_URI', uri)
    monkeypatch.setenv('MLFLOW_HTTP_REQUEST_MAX_RETRIES', '0')
    monkeypatch.setenv('MLFLOW_HTTP_REQUEST_TIMEOUT', '2')
    with pytest.raises(ModelLoadError, match=uri):
        load_served_model(ModelSettings(NAME))
