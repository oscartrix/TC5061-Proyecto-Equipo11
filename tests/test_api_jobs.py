"""Pruebas de las predicciones encoladas con RQ."""

from fakeredis import FakeRedis
from fastapi.testclient import TestClient
import numpy as np
import pytest
from rq import Queue

from mi_bici.api import jobs, main
from mi_bici.api.schemas import EXAMPLE_FEATURES


class FixedModel:
    name = 'modelo-prueba'
    version = '3'
    run_id = None
    backend = 'onnx'

    def predict(self, frame):
        return np.array([2.5, -1.0][: len(frame)])


@pytest.fixture
def queue():
    # is_async=False ejecuta el trabajo al encolarlo, sin worker
    return Queue(jobs.QUEUE_NAME, connection=FakeRedis(), is_async=False)


@pytest.fixture
def client(queue, monkeypatch):
    monkeypatch.setattr(main, 'load_served_model', FixedModel)
    monkeypatch.setattr(main, 'get_queue', lambda: queue)
    monkeypatch.setattr(jobs, '_served_model', FixedModel())
    with TestClient(main.app) as test_client:
        yield test_client


def test_enqueue_and_read_result(client):
    body = {'instances': [EXAMPLE_FEATURES, EXAMPLE_FEATURES]}
    response = client.post('/p1/predict-async', json=body)
    assert response.status_code == 202
    job_id = response.json()['job_id']

    job = client.get(f'/jobs/{job_id}').json()
    assert job['status'] == 'finished'
    # Igual que /p1/predict, los negativos quedan en cero
    assert job['result'] == {
        'model_name': 'modelo-prueba',
        'model_version': '3',
        'predictions': [2.5, 0.0],
    }


def test_invalid_rows_are_not_enqueued(client, queue):
    response = client.post('/p1/predict-async', json={'instances': [{'month': 13}]})
    assert response.status_code == 422
    assert queue.count == 0


def test_unknown_job_returns_404(client):
    assert client.get('/jobs/no-existe').status_code == 404


def test_without_redis_returns_503(monkeypatch):
    monkeypatch.setattr(main, 'load_served_model', FixedModel)
    monkeypatch.delenv('REDIS_URL', raising=False)
    with TestClient(main.app) as test_client:
        body = {'instances': [EXAMPLE_FEATURES]}
        assert test_client.post('/p1/predict-async', json=body).status_code == 503
        assert test_client.get('/jobs/abc').status_code == 503


def test_worker_loads_model_once(monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, '_served_model', None)
    monkeypatch.setattr(jobs, 'load_served_model', lambda: calls.append(1) or FixedModel())
    jobs.predict_batch([EXAMPLE_FEATURES])
    jobs.predict_batch([EXAMPLE_FEATURES])
    assert len(calls) == 1
