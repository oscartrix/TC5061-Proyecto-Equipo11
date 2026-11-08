"""Pruebas de carga con Locust para la API de demanda MiBici.

Uso:
    # Modo con interfaz Web:
    locust -f scripts/locustfile.py --host http://localhost:8000

    # Modo headless generando reporte HTML con gráficas:
    locust -f scripts/locustfile.py --headless -u 20 -r 5 -t 30s \
        --host http://localhost:8000 \
        --html reports/locust_report.html
"""

from locust import HttpUser, between, task

from mi_bici.api.schemas import EXAMPLE_FEATURES


class MiBiciApiUser(HttpUser):
    # Tiempo de espera entre tareas por usuario (entre 0.5 y 2.0 segundos)
    wait_time = between(0.5, 2.0)

    @task(3)
    def test_health(self) -> None:
        """Verifica el estado del servicio."""
        self.client.get('/health', name='/health')

    @task(1)
    def test_model_info(self) -> None:
        """Consulta la versión y metadatos del modelo cargado."""
        self.client.get('/model-info', name='/model-info')

    @task(6)
    def test_predict_single_instance(self) -> None:
        """Envía una predicción con 1 instancia."""
        payload = {'instances': [EXAMPLE_FEATURES]}
        self.client.post('/p1/predict', json=payload, name='/p1/predict (1 instancia)')

    @task(2)
    def test_predict_batch_instances(self) -> None:
        """Envía una predicción con un lote de 10 instancias."""
        payload = {'instances': [EXAMPLE_FEATURES] * 10}
        self.client.post('/p1/predict', json=payload, name='/p1/predict (batch 10)')
