"""Predicciones por lote en segundo plano con Redis y RQ."""

import os

from redis import Redis
from rq import Queue

from mi_bici.api.model import load_served_model, predict_rows

QUEUE_NAME = 'predicciones'
# Los resultados se borran de Redis después de una hora
RESULT_TTL = 3600

_served_model = None


def get_queue() -> Queue | None:
    """Cola de predicciones; None si no hay REDIS_URL."""
    url = os.getenv('REDIS_URL')
    return Queue(QUEUE_NAME, connection=Redis.from_url(url)) if url else None


def predict_batch(rows: list[dict]) -> dict:
    """Tarea del worker. Con SimpleWorker el modelo se carga una sola vez."""
    global _served_model
    if _served_model is None:
        _served_model = load_served_model()
    return {
        'model_name': _served_model.name,
        'model_version': _served_model.version,
        'predictions': predict_rows(_served_model, rows),
    }
