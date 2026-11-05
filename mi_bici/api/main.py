"""API REST de demanda diaria por estación (P1)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd

from mi_bici.api.model import ServedModel, load_served_model
from mi_bici.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Si el modelo no carga, el servicio no arranca.
    app.state.served_model = load_served_model()
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
    )


@app.post('/p1/predict', summary='Predecir viajes por estación y día')
def predict(body: PredictionRequest, request: Request) -> PredictionResponse:
    served_model = get_served_model(request)
    features = pd.DataFrame([row.model_dump() for row in body.instances])
    # No puede haber viajes negativos; el entrenamiento los limita igual al medir el error.
    predictions = np.clip(served_model.predict(features), 0.0, None)
    return PredictionResponse(
        model_name=served_model.name,
        model_version=served_model.version,
        predictions=predictions.tolist(),
    )
