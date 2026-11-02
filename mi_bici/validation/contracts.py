"""Contratos de aceptación de los archivos de características completos."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api import types

from mi_bici.config import END_MONTH, START_MONTH


@dataclass(frozen=True)
class ColumnRule:
    dtype: str
    minimum: float | str | None = None
    maximum: float | str | None = None
    values: tuple | None = None


STATION_ID = ColumnRule('integer', minimum=1)
CAPACITY = ColumnRule('integer', minimum=1)
NONNEGATIVE = ColumnRule('number', minimum=0)
FRACTION = ColumnRule('number', minimum=0, maximum=1)
TEXT = ColumnRule('string')
FLOW = ColumnRule('number')
DATE = ColumnRule(
    'datetime', minimum=f'{START_MONTH}-01', maximum=str(pd.Period(END_MONTH, freq='M').end_time)
)
RISK = ColumnRule('string', values=('deficit', 'balanceado', 'excedente'))
CALENDAR = {
    'day_of_week': ColumnRule('integer', minimum=0, maximum=6),
    'is_weekend': ColumnRule('integer', values=(0, 1)),
    'split': ColumnRule('string', values=('train', 'val', 'test')),
}
PROFILE = {
    'station_id': STATION_ID,
    'name': TEXT,
    'latitude': ColumnRule('number', minimum=-90, maximum=90),
    'longitude': ColumnRule('number', minimum=-180, maximum=180),
    'capacity': CAPACITY,
    'zone': TEXT,
    'is_ex': ColumnRule('boolean', values=(False,)),
    'is_active': ColumnRule('boolean', values=(True,)),
    'total_deps': ColumnRule('integer', minimum=1),
    'total_arrs': NONNEGATIVE,
    'dep_morning': FRACTION,
    'dep_midday': FRACTION,
    'dep_evening': FRACTION,
    'dep_weekend': FRACTION,
    'median_duration_min': ColumnRule('number', minimum=1, maximum=180),
    'round_trip_pct': FRACTION,
    'avg_daily_trips': NONNEGATIVE,
    'dep_arr_ratio': NONNEGATIVE,
}
FEATURE_CONTRACTS = {
    'P1': {
        'station_id': STATION_ID,
        'date': DATE,
        'trips_count': NONNEGATIVE,
        'capacity': CAPACITY,
        'zone': TEXT,
        **{name: NONNEGATIVE for name in (
            'lag_1d', 'lag_7d', 'lag_14d', 'lag_21d', 'rolling_mean_7d'
        )},
        **CALENDAR,
        'month': ColumnRule('integer', minimum=1, maximum=12),
        'is_holiday': ColumnRule('integer', values=(0, 1)),
    },
    'P2': {
        'station_id': STATION_ID,
        'hour_ts': DATE,
        'departures': NONNEGATIVE,
        'arrivals': NONNEGATIVE,
        'capacity': CAPACITY,
        'zone': TEXT,
        **{name: FLOW for name in (
            'net_flow', 'future_3h_net_flow', 'pressure_ratio', 'net_flow_last_3h',
            'pressure_last_3h', 'pressure_window_lag_24h',
            'pressure_window_lag_168h', 'pressure_window_lag_336h',
        )},
        'risk_class': RISK,
        'risk_lag_168h': RISK,
        'hour': ColumnRule('integer', minimum=0, maximum=23),
        **CALENDAR,
    },
    'P3_train': PROFILE,
    'P3_holdout': PROFILE,
}
FEATURE_KEYS = {
    'P1': ('station_id', 'date'),
    'P2': ('station_id', 'hour_ts'),
    'P3_train': ('station_id',),
    'P3_holdout': ('station_id',),
}


def matches_dtype(series: pd.Series, kind: str) -> bool:
    """Acepta tipos nativos y nullable de pandas sin convertir los datos."""
    if kind == 'integer':
        return types.is_integer_dtype(series.dtype)
    if kind == 'number':
        return types.is_integer_dtype(series.dtype) or types.is_float_dtype(series.dtype)
    if kind == 'boolean':
        return types.is_bool_dtype(series.dtype)
    if kind == 'datetime':
        return types.is_datetime64_dtype(series.dtype)
    if kind == 'string':
        return types.infer_dtype(series.dropna()) in {'string', 'unicode', 'empty'}
    raise ValueError(f'Tipo de contrato desconocido: {kind}')


def validate_feature_contract(name: str, df: pd.DataFrame) -> list[str]:
    """Devuelve todos los incumplimientos del contrato sobre el dataset completo."""
    contract = FEATURE_CONTRACTS[name]
    errors = []
    if df.empty:
        errors.append(f'{name}: sin datos')
    if df.columns.has_duplicates:
        return [f'{name}: nombres de columnas duplicados']

    missing = sorted(set(contract) - set(df.columns))
    extra = sorted(set(df.columns) - set(contract))
    if missing:
        errors.append(f'{name}: faltan columnas requeridas: {", ".join(missing)}')
    if extra:
        errors.append(f'{name}: columnas no permitidas: {", ".join(extra)}')

    for column, rule in contract.items():
        if column not in df:
            continue
        series = df[column]
        prefix = f'{name}.{column}'
        nulls = int(series.isna().sum())
        if nulls:
            errors.append(f'{prefix}: {nulls} valores faltantes')
        if not matches_dtype(series, rule.dtype):
            errors.append(f'{prefix}: tipo {series.dtype}; se requiere {rule.dtype}')
            continue

        values = series.dropna()
        if rule.dtype in {'integer', 'number'}:
            invalid = int((~np.isfinite(values.to_numpy(dtype=float))).sum())
            if invalid:
                errors.append(f'{prefix}: {invalid} valores no finitos')
        if rule.dtype == 'string' and values.str.strip().eq('').any():
            errors.append(f'{prefix}: contiene texto vacío')
        for bound, limit in [('minimum', rule.minimum), ('maximum', rule.maximum)]:
            if limit is None:
                continue
            limit = pd.Timestamp(limit) if rule.dtype == 'datetime' else limit
            outside = values < limit if bound == 'minimum' else values > limit
            if outside.any():
                operator = '>=' if bound == 'minimum' else '<='
                errors.append(
                    f'{prefix}: {int(outside.sum())} valores fuera de rango ({operator} {limit})'
                )
        if rule.values is not None and (~values.isin(rule.values)).any():
            errors.append(f'{prefix}: valores fuera del catálogo {rule.values}')

    keys = FEATURE_KEYS[name]
    if all(key in df for key in keys):
        duplicates = int(df.duplicated(subset=list(keys)).sum())
        if duplicates:
            errors.append(f'{name}: {duplicates} claves duplicadas en {", ".join(keys)}')
    return errors
