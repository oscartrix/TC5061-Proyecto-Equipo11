"""Aceptación y rechazo de datos completos según los contratos del proyecto."""

import numpy as np
import pandas as pd
import pytest

from mi_bici.validation.contracts import validate_feature_contract


@pytest.mark.parametrize('name', ['P1', 'P2', 'P3_train', 'P3_holdout'])
def test_valid_feature_contract(valid_feature_frames, name):
    assert validate_feature_contract(name, valid_feature_frames[name]) == []


@pytest.mark.parametrize('name,column,dtype', [
    ('P1', 'capacity', 'Int32'), ('P1', 'zone', 'string'),
    ('P2', 'pressure_ratio', 'Float64'), ('P3_train', 'is_active', 'boolean'),
])
def test_nullable_pandas_dtypes_are_accepted(valid_feature_frames, name, column, dtype):
    df = valid_feature_frames[name]
    df[column] = df[column].astype(dtype)
    assert validate_feature_contract(name, df) == []


@pytest.mark.parametrize('name,column,value,reason', [
    ('P1', 'capacity', 0, 'fuera de rango'),
    ('P1', 'trips_count', -1, 'fuera de rango'),
    ('P1', 'month', 13, 'fuera de rango'),
    ('P1', 'day_of_week', 7, 'fuera de rango'),
    ('P1', 'is_weekend', 2, 'catálogo'),
    ('P1', 'zone', '  ', 'texto vacío'),
    ('P1', 'date', pd.Timestamp('2023-12-31'), 'fuera de rango'),
    ('P2', 'hour', 24, 'fuera de rango'),
    ('P2', 'risk_class', 'desconocida', 'catálogo'),
    ('P2', 'pressure_ratio', np.inf, 'no finitos'),
    ('P2', 'pressure_last_3h', np.nan, 'faltantes'),
    ('P3_train', 'latitude', 91., 'fuera de rango'),
    ('P3_train', 'longitude', -181., 'fuera de rango'),
    ('P3_train', 'dep_morning', 1.01, 'fuera de rango'),
    ('P3_holdout', 'median_duration_min', 181., 'fuera de rango'),
    ('P3_holdout', 'is_active', False, 'catálogo'),
])
def test_invalid_values_are_rejected(valid_feature_frames, name, column, value, reason):
    df = valid_feature_frames[name]
    df.loc[0, column] = value
    errors = validate_feature_contract(name, df)
    assert any(column in error and reason in error for error in errors)


@pytest.mark.parametrize('name,column,value', [
    ('P1', 'capacity', 1), ('P1', 'trips_count', 0),
    ('P1', 'month', 1), ('P1', 'month', 12),
    ('P2', 'hour', 0), ('P2', 'hour', 23),
    ('P3_train', 'latitude', -90.), ('P3_train', 'latitude', 90.),
    ('P3_train', 'longitude', -180.), ('P3_train', 'longitude', 180.),
    ('P3_holdout', 'dep_morning', 0.), ('P3_holdout', 'dep_morning', 1.),
    ('P3_holdout', 'median_duration_min', 1.),
    ('P3_holdout', 'median_duration_min', 180.),
])
def test_range_boundaries_are_inclusive(valid_feature_frames, name, column, value):
    df = valid_feature_frames[name]
    df.loc[0, column] = value
    assert validate_feature_contract(name, df) == []


@pytest.mark.parametrize('name,column,dtype', [
    ('P1', 'capacity', str), ('P1', 'date', str),
    ('P2', 'hour', float), ('P3_train', 'is_active', str),
])
def test_wrong_column_type_is_rejected(valid_feature_frames, name, column, dtype):
    df = valid_feature_frames[name]
    df[column] = df[column].astype(dtype)
    assert any(column in error and 'se requiere' in error
               for error in validate_feature_contract(name, df))


def test_missing_and_unexpected_columns_are_reported(valid_feature_frames):
    df = valid_feature_frames['P1'].drop(columns='capacity').assign(user_id=123)
    errors = validate_feature_contract('P1', df)
    assert any('faltan columnas' in error and 'capacity' in error for error in errors)
    assert any('no permitidas' in error and 'user_id' in error for error in errors)


@pytest.mark.parametrize('name,changed_column', [
    ('P1', 'trips_count'), ('P2', 'departures'), ('P3_train', 'total_deps'),
])
def test_duplicate_keys_with_different_values_are_rejected(
    valid_feature_frames, name, changed_column
):
    df = valid_feature_frames[name]
    duplicate = df.iloc[[0]].copy()
    duplicate[changed_column] += 1
    df = pd.concat([df, duplicate], ignore_index=True)
    assert not df.duplicated().any()
    assert any('claves duplicadas' in error for error in validate_feature_contract(name, df))
