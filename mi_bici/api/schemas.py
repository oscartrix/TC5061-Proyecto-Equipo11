"""Esquemas de entrada y salida de la API de demanda diaria por estación (P1)."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_INSTANCES = 500

EXAMPLE_FEATURES = {
    'lag_1d': 12.0,
    'lag_7d': 11.0,
    'lag_14d': 3.0,
    'lag_21d': 4.0,
    'rolling_mean_7d': 11.857143,
    'capacity': 19,
    'day_of_week': 2,
    'month': 9,
    'is_weekend': 0,
    'is_holiday': 0,
    'zone': 'ZPN',
}

TripCount = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class P1Features(BaseModel):
    """Variables de una estación y un día."""

    # strict evita conversiones silenciosas, como "12" a 12 o true a 1.
    model_config = ConfigDict(
        extra='forbid',
        strict=True,
        json_schema_extra={'examples': [EXAMPLE_FEATURES]},
    )

    lag_1d: TripCount = Field(description='Viajes de la estación hace 1 día')
    lag_7d: TripCount = Field(description='Viajes de la estación hace 7 días')
    lag_14d: TripCount = Field(description='Viajes de la estación hace 14 días')
    lag_21d: TripCount = Field(description='Viajes de la estación hace 21 días')
    rolling_mean_7d: TripCount = Field(description='Media de viajes de los 7 días previos')
    capacity: int = Field(ge=1, description='Anclajes de la estación')
    day_of_week: int = Field(ge=0, le=6, description='Día de la semana: 0 es lunes, 6 es domingo')
    month: int = Field(ge=1, le=12, description='Mes del año')
    is_weekend: int = Field(ge=0, le=1, description='1 si el día es sábado o domingo')
    is_holiday: int = Field(ge=0, le=1, description='1 si el día es festivo en México')
    zone: Literal['GDL', 'ZPN', 'TLQ'] = Field(description='Zona de la estación')

    @model_validator(mode='after')
    def check_weekend_matches_day(self) -> Self:
        if self.is_weekend != int(self.day_of_week >= 5):
            raise ValueError(
                f'is_weekend={self.is_weekend} no corresponde a day_of_week={self.day_of_week}'
            )
        return self


class PredictionRequest(BaseModel):
    """Una o varias filas a predecir."""

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={'examples': [{'instances': [EXAMPLE_FEATURES]}]},
    )

    instances: list[P1Features] = Field(min_length=1, max_length=MAX_INSTANCES)


class PredictionResponse(BaseModel):
    """Predicciones y versión concreta del modelo que las generó."""

    model_config = ConfigDict(extra='forbid')

    model_name: str = Field(min_length=1, description='Nombre en el Model Registry')
    model_version: str = Field(min_length=1, description='Versión en el Model Registry')
    predictions: list[TripCount] = Field(
        min_length=1,
        max_length=MAX_INSTANCES,
        description='Viajes esperados por fila, en el orden de la petición',
    )
