"""Pruebas de los esquemas de la API de P1."""

import json

from pydantic import ValidationError
import pytest

from mi_bici.api.schemas import (
    EXAMPLE_FEATURES,
    MAX_INSTANCES,
    P1Features,
    PredictionRequest,
    PredictionResponse,
)
from mi_bici.modeling.train import P1_CAT_COLS, P1_NUM_COLS


def features(**changes) -> dict:
    return {**EXAMPLE_FEATURES, **changes}


def response(**changes) -> dict:
    return {'model_name': 'modelo', 'model_version': '1', 'predictions': [8.5], **changes}


def test_schema_has_the_columns_used_to_train():
    assert set(P1Features.model_fields) == set(P1_NUM_COLS + P1_CAT_COLS)


def test_example_is_valid():
    assert P1Features.model_validate(EXAMPLE_FEATURES).model_dump() == EXAMPLE_FEATURES


def test_json_request_is_accepted_and_keeps_order():
    payload = {'instances': [features(month=1), features(month=2)]}
    request = PredictionRequest.model_validate_json(json.dumps(payload))
    assert [row.month for row in request.instances] == [1, 2]


def test_integer_is_accepted_for_trip_counts():
    assert P1Features.model_validate(features(lag_1d=12)).lag_1d == 12.0


@pytest.mark.parametrize('day,is_weekend', [(0, 0), (4, 0), (5, 1), (6, 1)])
def test_weekend_consistent_with_day_is_accepted(day, is_weekend):
    P1Features.model_validate(features(day_of_week=day, is_weekend=is_weekend))


@pytest.mark.parametrize('day,is_weekend', [(4, 1), (5, 0), (6, 0), (0, 1)])
def test_weekend_inconsistent_with_day_is_rejected(day, is_weekend):
    with pytest.raises(ValidationError, match='no corresponde'):
        P1Features.model_validate(features(day_of_week=day, is_weekend=is_weekend))


@pytest.mark.parametrize('field,value', [
    ('lag_1d', -1.0),
    ('lag_7d', -0.5),
    ('lag_14d', -1.0),
    ('lag_21d', -1.0),
    ('rolling_mean_7d', -0.1),
    ('capacity', 0),
    ('day_of_week', -1),
    ('day_of_week', 7),
    ('month', 0),
    ('month', 13),
    ('is_weekend', 2),
    ('is_holiday', -1),
    ('is_holiday', 2),
    ('zone', 'CDMX'),
    ('zone', 'gdl'),
])
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValidationError) as error:
        P1Features.model_validate(features(**{field: value}))
    assert error.value.errors()[0]['loc'] == (field,)


@pytest.mark.parametrize('field,value', [
    ('lag_1d', '12'),
    ('lag_1d', True),
    ('lag_1d', None),
    ('capacity', '19'),
    ('capacity', 19.5),
    ('month', 9.0),
    ('is_weekend', True),
    ('is_holiday', False),
    ('zone', 1),
])
def test_incorrect_types_are_not_converted(field, value):
    with pytest.raises(ValidationError) as error:
        P1Features.model_validate(features(**{field: value}))
    assert error.value.errors()[0]['loc'] == (field,)


@pytest.mark.parametrize('literal', ['NaN', 'Infinity', '-Infinity'])
def test_non_finite_values_in_json_are_rejected(literal):
    # json.loads acepta estos literales, así que deben rechazarse también al validar.
    payload = json.loads(f'{{"lag_1d": {literal}}}')
    with pytest.raises(ValidationError) as error:
        P1Features.model_validate(features(**payload))
    assert error.value.errors()[0]['loc'] == ('lag_1d',)


@pytest.mark.parametrize('field', EXAMPLE_FEATURES)
def test_missing_field_is_rejected(field):
    payload = features()
    del payload[field]
    with pytest.raises(ValidationError) as error:
        P1Features.model_validate(payload)
    assert error.value.errors()[0]['type'] == 'missing'


@pytest.mark.parametrize('extra', ['station_id', 'date', 'trips_count', 'split'])
def test_columns_outside_the_model_are_rejected(extra):
    with pytest.raises(ValidationError) as error:
        P1Features.model_validate(features(**{extra: 1}))
    assert error.value.errors()[0]['type'] == 'extra_forbidden'


def test_request_limits():
    PredictionRequest.model_validate({'instances': [features()] * MAX_INSTANCES})
    for size in (0, MAX_INSTANCES + 1):
        with pytest.raises(ValidationError):
            PredictionRequest.model_validate({'instances': [features()] * size})


def test_request_rejects_extra_fields():
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate({'instances': [features()], 'extra': 1})


def test_request_points_to_the_invalid_row():
    payload = {'instances': [features(), features(month=13)]}
    with pytest.raises(ValidationError) as error:
        PredictionRequest.model_validate(payload)
    assert error.value.errors()[0]['loc'] == ('instances', 1, 'month')


def test_valid_response():
    assert PredictionResponse.model_validate(response()).predictions == [8.5]


@pytest.mark.parametrize('changes', [
    {'predictions': [-0.1]},
    {'predictions': [float('nan')]},
    {'predictions': [float('inf')]},
    {'predictions': []},
    {'predictions': [1.0] * (MAX_INSTANCES + 1)},
    {'model_name': ''},
    {'model_version': ''},
    {'extra': 1},
])
def test_invalid_response_is_rejected(changes):
    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(response(**changes))


def test_schema_example_is_published():
    schema = PredictionRequest.model_json_schema()
    assert schema['examples'] == [{'instances': [EXAMPLE_FEATURES]}]
