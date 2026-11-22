"""API REST de demanda diaria por estación (P1)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from rq.exceptions import NoSuchJobError
from rq.job import Job

from mi_bici.api.jobs import RESULT_TTL, get_queue, predict_batch
from mi_bici.api.model import ServedModel, load_served_model, predict_rows
from mi_bici.api.schemas import (
    HealthResponse,
    JobResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Si el modelo no carga, el servicio no arranca.
    app.state.served_model = load_served_model()
    app.state.queue = get_queue()
    yield


app = FastAPI(
    title='API de demanda MiBici',
    description='Predice los viajes de salida por estación para el día siguiente.',
    version='0.1.0',
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    # No se repite el valor recibido: NaN e Infinity no se pueden enviar en JSON.
    details = [
        {'loc': item['loc'], 'msg': item['msg'], 'type': item['type']} for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={'detail': details})


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    # El detalle queda en los logs del servidor y no se envía al cliente.
    return JSONResponse(status_code=500, content={'detail': 'Error interno del servidor'})


def get_served_model(request: Request) -> ServedModel:
    return request.app.state.served_model


@app.get('/health', summary='Revisar que el servicio esté activo')
def health() -> HealthResponse:
    return HealthResponse()


@app.get('/model-info', summary='Consultar el modelo cargado')
def model_info(request: Request) -> ModelInfoResponse:
    served_model = get_served_model(request)
    return ModelInfoResponse(
        model_name=served_model.name,
        model_version=served_model.version,
        run_id=served_model.run_id,
        backend=served_model.backend,
    )


@app.post('/p1/predict', summary='Predecir viajes por estación y día')
def predict(body: PredictionRequest, request: Request) -> PredictionResponse:
    served_model = get_served_model(request)
    rows = [row.model_dump() for row in body.instances]
    return PredictionResponse(
        model_name=served_model.name,
        model_version=served_model.version,
        predictions=predict_rows(served_model, rows),
    )


def get_queue_or_503(request: Request):
    queue = request.app.state.queue
    if queue is None:
        raise HTTPException(status_code=503, detail='La cola de predicciones no está configurada')
    return queue


@app.post('/p1/predict-async', status_code=202, summary='Encolar un lote de predicciones')
def predict_async(body: PredictionRequest, request: Request) -> JobResponse:
    queue = get_queue_or_503(request)
    rows = [row.model_dump() for row in body.instances]
    job = queue.enqueue(predict_batch, rows, result_ttl=RESULT_TTL)
    return JobResponse(job_id=job.id, status=job.get_status())


@app.get('/jobs/{job_id}', summary='Consultar una predicción encolada')
def job_status(job_id: str, request: Request) -> JobResponse:
    queue = get_queue_or_503(request)
    try:
        job = Job.fetch(job_id, connection=queue.connection)
    except NoSuchJobError:
        raise HTTPException(status_code=404, detail='No existe el trabajo') from None
    status = job.get_status()
    result = job.return_value() if status == 'finished' else None
    return JobResponse(job_id=job.id, status=status, result=result)
