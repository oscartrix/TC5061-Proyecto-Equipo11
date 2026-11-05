"""Pruebas de los endpoints de la API."""

import json

from fastapi.testclient import TestClient
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mi_bici.api import main
from mi_bici.api.model import ModelLoadError, ServedModel
from mi_bici.api.schemas import EXAMPLE_FEATURES, MAX_INSTANCES
from mi_bici.modeling.train import P1_CAT_COLS, P1_NUM_COLS

COLUMNS = P1_NUM_COLS + P1_CAT_COLS
PREDICT_URL = '/p1/predict'


def features(**changes) -> dict:
    return {**EXAMPLE_FEATURES, **changes}


def request_body(*rows) -> dict:
    return {'instances': list(rows) or [features()]}


@pytest.fixture(scope='module')
def pipeline_and_model(tmp_path_factory):
    """Modelo real de MLflow, con los mismos tipos que la tabla de P1."""
    x = pd.DataFrame([
        features(lag_1d=float(i), zone=['GDL', 'ZPN', 'TLQ'][i % 3])
        for i in range(12)
    ])[COLUMNS].astype({'capacity': 'float64', 'day_of_week': 'int32', 'month': 'int32'})
    prep = ColumnTransformer([
        ('num', 'passthrough', P1_NUM_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), P1_CAT_COLS),
    ])
    # La pendiente de -1 permite obtener predicciones negativas.
    pipeline = Pipeline([('prep', prep), ('reg', LinearRegression())]).fit(x, 5 - x['lag_1d'])
    path = tmp_path_factory.mktemp('api-model') / 'model'
    mlflow.sklearn.save_model(
        pipeline,
        path,
        signature=infer_signature(x, pipeline.predict(x)),
        serialization_format='cloudpickle',
    )
    return pipeline, mlflow.pyfunc.load_model(str(path))


@pytest.fixture
def served_model(pipeline_and_model):
    return ServedModel('modelo-prueba', '7', 'run-123', pipeline_and_model[1])


@pytest.fixture
def client(served_model, monkeypatch):
    monkeypatch.setattr(main, 'load_served_model', lambda: served_model)
    with TestClient(main.app) as test_client:
        yield test_client


class FakeServedModel:
    """Modelo que devuelve valores fijos o falla al predecir."""

    name = 'modelo-prueba'
    version = '7'
    run_id = None

    def __init__(self, result):
        self.result = result

    def predict(self, frame):
        if isinstance(self.result, Exception):
            raise self.result
        return np.asarray(self.result)


def client_with(monkeypatch, result) -> TestClient:
    monkeypatch.setattr(main, 'load_served_model', lambda: FakeServedModel(result))
    # Permite revisar la respuesta 500 en lugar de recibir la excepción en la prueba.
    return TestClient(main.app, raise_server_exceptions=False)


def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_model_info(client):
    response = client.get('/model-info')
    assert response.status_code == 200
    assert response.json() == {
        'model_name': 'modelo-prueba',
        'model_version': '7',
        'run_id': 'run-123',
    }


def test_predict_returns_model_prediction(client, pipeline_and_model):
    pipeline, _ = pipeline_and_model
    response = client.post(PREDICT_URL, json=request_body(features(lag_1d=1.0)))
    assert response.status_code == 200
    body = response.json()
    assert (body['model_name'], body['model_version']) == ('modelo-prueba', '7')
    frame = pd.DataFrame([features(lag_1d=1.0)])[COLUMNS]
    np.testing.assert_allclose(body['predictions'], pipeline.predict(frame))


def test_predict_keeps_row_order(client, pipeline_and_model):
    pipeline, _ = pipeline_and_model
    rows = [features(lag_1d=float(i)) for i in (3, 0, 2, 1)]
    response = client.post(PREDICT_URL, json=request_body(*rows))
    assert response.status_code == 200
    expected = pipeline.predict(pd.DataFrame(rows)[COLUMNS])
    assert len(set(expected.round(6))) == len(rows)
    np.testing.assert_allclose(response.json()['predictions'], expected)


def test_predict_limits_negative_values_to_zero(client, pipeline_and_model):
    pipeline, _ = pipeline_and_model
    row = features(lag_1d=50.0)
    assert pipeline.predict(pd.DataFrame([row])[COLUMNS])[0] < 0
    response = client.post(PREDICT_URL, json=request_body(row))
    assert response.status_code == 200
    assert response.json()['predictions'] == [0.0]


def test_predict_accepts_maximum_size(client):
    response = client.post(PREDICT_URL, json=request_body(*[features()] * MAX_INSTANCES))
    assert response.status_code == 200
    assert len(response.json()['predictions']) == MAX_INSTANCES


@pytest.mark.parametrize('body,location', [
    ({'instances': []}, ['body', 'instances']),
    ({}, ['body', 'instances']),
    ({'instances': [features(month=13)]}, ['body', 'instances', 0, 'month']),
    ({'instances': [features(capacity='19')]}, ['body', 'instances', 0, 'capacity']),
    ({'instances': [features(zone='CDMX')]}, ['body', 'instances', 0, 'zone']),
    ({'instances': [features(station_id=1)]}, ['body', 'instances', 0, 'station_id']),
    ({'instances': [features(is_weekend=1)]}, ['body', 'instances', 0]),
    ({'instances': [features()], 'extra': 1}, ['body', 'extra']),
])
def test_invalid_body_returns_422(client, body, location):
    response = client.post(PREDICT_URL, json=body)
    assert response.status_code == 422
    assert response.json()['detail'][0]['loc'] == location


def test_too_many_rows_return_422(client):
    body = request_body(*[features()] * (MAX_INSTANCES + 1))
    assert client.post(PREDICT_URL, json=body).status_code == 422


def test_malformed_json_returns_422(client):
    response = client.post(
        PREDICT_URL, content='{"instances": [', headers={'Content-Type': 'application/json'}
    )
    assert response.status_code == 422


@pytest.mark.parametrize('literal', ['NaN', 'Infinity', '-Infinity'])
def test_non_finite_values_return_422(client, literal):
    # Python lee estos literales aunque no son JSON estándar.
    row = json.dumps(features()).replace('"lag_1d": 12.0', f'"lag_1d": {literal}')
    response = client.post(
        PREDICT_URL,
        content=f'{{"instances": [{row}]}}',
        headers={'Content-Type': 'application/json'},
    )
    assert response.status_code == 422
    error = response.json()['detail'][0]
    assert error['loc'] == ['body', 'instances', 0, 'lag_1d']
    assert 'input' not in error


def test_model_failure_returns_500_without_details(monkeypatch):
    with client_with(monkeypatch, ValueError('detalle-interno')) as client:
        response = client.post(PREDICT_URL, json=request_body())
    assert response.status_code == 500
    assert response.json() == {'detail': 'Error interno del servidor'}
    assert 'detalle-interno' not in response.text


@pytest.mark.parametrize('result', [[float('nan')], [float('inf')]])
def test_non_finite_prediction_returns_500(monkeypatch, result):
    with client_with(monkeypatch, result) as client:
        response = client.post(PREDICT_URL, json=request_body())
    assert response.status_code == 500
    assert response.json() == {'detail': 'Error interno del servidor'}


def test_service_does_not_start_without_model(monkeypatch):
    def fail():
        raise ModelLoadError('sin modelo')

    monkeypatch.setattr(main, 'load_served_model', fail)
    with pytest.raises(ModelLoadError, match='sin modelo'), TestClient(main.app):
        pass


def test_openapi_publishes_example_and_routes(client):
    schema = client.get('/openapi.json').json()
    assert {'/health', '/model-info', PREDICT_URL} <= set(schema['paths'])
    example = schema['components']['schemas']['PredictionRequest']['examples'][0]
    assert example == {'instances': [EXAMPLE_FEATURES]}
