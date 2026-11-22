"""Pruebas de carga con Locust para la API de demanda MiBici.

Uso:
    # Modo con interfaz Web:
    locust -f scripts/locustfile.py --host http://localhost:8000

    # Modo headless con usuarios fijos:
    locust -f scripts/locustfile.py --headless -u 20 -r 5 -t 30s \
        --host http://localhost:8000 --html reports/locust_report.html

    # Carga por escalones (20 a 400 usuarios, 60 s cada uno):
    LOCUST_STEPS=20,50,100,200,300,400 LOCUST_STEP_SECONDS=60 \
        locust -f scripts/locustfile.py --headless --host http://localhost:8000

Con LOCUST_ASYNC=1 se agrega la tarea que encola lotes en Redis (solo con docker compose).
"""

import os
import random

from locust import HttpUser, LoadTestShape, between, task

from mi_bici.api.schemas import EXAMPLE_FEATURES

ZONES = ['GDL', 'ZPN', 'TLQ']


def random_row() -> dict:
    """Fila válida con rezagos distintos, para no repetir siempre la misma entrada."""
    lags = [round(random.uniform(0, 60), 1) for _ in range(4)]
    day = random.randint(0, 6)
    return {
        **EXAMPLE_FEATURES,
        'lag_1d': lags[0],
        'lag_7d': lags[1],
        'lag_14d': lags[2],
        'lag_21d': lags[3],
        'rolling_mean_7d': round(sum(lags) / 4, 3),
        'capacity': random.randint(10, 40),
        'day_of_week': day,
        'is_weekend': int(day >= 5),
        'month': random.randint(1, 12),
        'zone': random.choice(ZONES),
    }


class MiBiciApiUser(HttpUser):
    # Un usuario espera entre 0.5 y 2 s entre peticiones
    wait_time = between(0.5, 2.0)

    @task(3)
    def health(self) -> None:
        self.client.get('/health', name='/health')

    @task(1)
    def model_info(self) -> None:
        self.client.get('/model-info', name='/model-info')

    @task(6)
    def predict_one(self) -> None:
        payload = {'instances': [random_row()]}
        self.client.post('/p1/predict', json=payload, name='/p1/predict (1 instancia)')

    @task(2)
    def predict_batch(self) -> None:
        payload = {'instances': [random_row() for _ in range(10)]}
        self.client.post('/p1/predict', json=payload, name='/p1/predict (batch 10)')

    @task(1)
    def predict_async(self) -> None:
        if os.getenv('LOCUST_ASYNC') != '1':
            return
        payload = {'instances': [random_row() for _ in range(50)]}
        self.client.post('/p1/predict-async', json=payload, name='/p1/predict-async (50)')


if os.getenv('LOCUST_STEPS'):

    class StepLoad(LoadTestShape):
        """Sube los usuarios por escalones para encontrar dónde se satura la API."""

        steps = tuple(int(users) for users in os.environ['LOCUST_STEPS'].split(','))
        seconds = int(os.getenv('LOCUST_STEP_SECONDS', '60'))

        def tick(self):
            step = int(self.get_run_time() // self.seconds)
            if step >= len(self.steps):
                return None
            return self.steps[step], 20
