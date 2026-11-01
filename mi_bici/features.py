"""Módulo de ingeniería de características para P1, P2 y P3.

Uso:
    python -m mi_bici.features
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mi_bici.config import (
    P1_FEATURES_PATH,
    P2_FEATURES_PATH,
    P3_HOLDOUT_FEATURES_PATH,
    P3_TRAIN_FEATURES_PATH,
    STATIONS_PARQUET_PATH,
    TRAIN_END_DATE,
    TRIPS_PARQUET_PATH,
    VAL_END_DATE,
    VAL_START_DATE,
)

app = typer.Typer()

# Días de descanso obligatorio en México (LFT art. 74). Tres caen en lunes y cambian cada año.
FIXED_HOLIDAYS_MX = {(1, 1), (5, 1), (9, 16), (12, 25)}
MONDAY_HOLIDAYS_MX = {2: 1, 3: 3, 11: 3}  # mes: n-ésimo lunes


def is_mx_holiday(day: pd.Timestamp) -> bool:
    """Indica si la fecha es día de descanso obligatorio en México."""
    if (day.month, day.day) in FIXED_HOLIDAYS_MX:
        return True
    if day.month == 10 and day.day == 1 and (day.year - 2024) % 6 == 0:
        return True  # transmisión del Poder Ejecutivo
    nth = MONDAY_HOLIDAYS_MX.get(day.month)
    return nth is not None and day.dayofweek == 0 and (day.day - 1) // 7 + 1 == nth


def assign_temporal_split(dates: pd.Series) -> pd.Series:
    """Asigna el conjunto temporal (train, val, test) según la fecha sin filtrar información futura.

    Pertinencia (P1 y P2): Garantiza que el entrenamiento use solo 2024-2025,
    la validación ene-jun 2026 y la prueba jul-sep 2026.
    """
    day_str = dates.dt.strftime('%Y-%m-%d')
    split = pd.Series('test', index=dates.index, dtype='string')
    split.loc[day_str <= TRAIN_END_DATE] = 'train'
    split.loc[(day_str >= VAL_START_DATE) & (day_str <= VAL_END_DATE)] = 'val'
    return split


def build_p1_daily_features(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla de características diarias por estación para P1 (Regresión).

    Qué hace:
        Agrega los viajes iniciados por estación y día sobre la malla completa de estaciones
        activas, añade rezagos pasados (1, 7, 14 y 21 días), promedio móvil de 7 días previos
        y variables de calendario (día de la semana, mes, fin de semana, festivo), además de
        la capacidad y zona de cada estación.

    Por qué es pertinente para P1:
        La demanda diaria en sistemas de bicicleta pública tiene fuerte autocorrelación semanal
        (lag_7d, lag_14d, lag_21d) y tendencia reciente (lag_1d, rolling_mean_7d). Todos los lags
        se calculan con shift positivo (>=1 día) para evitar fuga de información del futuro.
    """
    df = trips[['origin_id', 'start_time']].copy()
    df['date'] = df['start_time'].dt.floor('D')

    daily_counts = (
        df.groupby(['origin_id', 'date'], as_index=False)
        .size()
        .rename(columns={'origin_id': 'station_id', 'size': 'trips_count'})
    )

    active_stations = stations.loc[
        stations['is_active'], ['station_id', 'capacity', 'zone']
    ].copy()
    all_dates = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
    first_day = daily_counts.groupby('station_id')['date'].min()

    grid = (
        first_day
        .apply(lambda d: pd.Series(all_dates[all_dates >= d]))
        .stack()
        .reset_index(level=1, drop=True)
        .reset_index()
        .rename(columns={0: 'date'})
    )

    p1 = grid.merge(daily_counts, on=['station_id', 'date'], how='left')
    p1['trips_count'] = p1['trips_count'].fillna(0.0).astype(float)
    p1 = p1.merge(active_stations, on='station_id', how='inner')
    p1 = p1.sort_values(['station_id', 'date']).reset_index(drop=True)

    grp = p1.groupby('station_id')['trips_count']
    p1['lag_1d'] = grp.shift(1)
    p1['lag_7d'] = grp.shift(7)
    p1['lag_14d'] = grp.shift(14)
    p1['lag_21d'] = grp.shift(21)
    p1['rolling_mean_7d'] = grp.transform(lambda s: s.shift(1).rolling(7).mean())

    p1['day_of_week'] = p1['date'].dt.dayofweek
    p1['month'] = p1['date'].dt.month
    p1['is_weekend'] = (p1['day_of_week'] >= 5).astype(int)
    p1['is_holiday'] = p1['date'].map(is_mx_holiday).astype(int)
    p1['split'] = assign_temporal_split(p1['date'])

    p1 = p1.dropna(
        subset=['lag_1d', 'lag_7d', 'lag_14d', 'lag_21d', 'rolling_mean_7d']
    ).reset_index(drop=True)
    return p1


def build_p2_hourly_features(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
    top_n_stations: int = 50,
    pressure_threshold: float = 0.30,
) -> pd.DataFrame:
    """Construye la tabla horaria de flujo neto y riesgo para P2 (Clasificación).

    Qué hace:
        Calcula las salidas (departures), llegadas (arrivals) y flujo neto horario
        (arrivals - departures) para las estaciones activas de mayor demanda (seleccionadas
        exclusivamente con datos de Train). Aproxima el riesgo en las próximas 3 horas
        comparando el flujo neto futuro acumulado de 3h contra la capacidad de la estación:
        - 'deficit': si future_3h_net_flow / capacity <= -pressure_threshold
        - 'excedente': si future_3h_net_flow / capacity >= +pressure_threshold
        - 'balanceado': en caso contrario.

    Por qué es pertinente para P2:
        Dado que MiBici no publica el inventario en tiempo real, una pérdida o ganancia neta
        acumulada superior al 30% de los anclajes totales en 3 horas identifica las estaciones
        con alta probabilidad de vaciarse o saturarse. Las variables predictoras usan solo
        datos hasta la hora t (flujo actual y rezagos de 3h, 24h y 168h).
    """
    active = stations.loc[
        stations['is_active'] & (stations['capacity'] > 0),
        ['station_id', 'capacity', 'zone'],
    ].copy()

    # Seleccionar estaciones usando únicamente el periodo de entrenamiento para evitar fuga
    train_trips = trips[trips['start_time'].dt.strftime('%Y-%m-%d') <= TRAIN_END_DATE]
    top_ids = (
        train_trips[train_trips['origin_id'].isin(active['station_id'])]
        .groupby('origin_id')
        .size()
        .nlargest(top_n_stations)
        .index
    )

    sub = trips[trips['origin_id'].isin(top_ids) | trips['destination_id'].isin(top_ids)].copy()
    sub['dep_hour'] = sub['start_time'].dt.floor('h')
    sub['arr_hour'] = sub['end_time'].dt.floor('h')

    deps = (
        sub[sub['origin_id'].isin(top_ids)]
        .groupby(['origin_id', 'dep_hour'], as_index=False)
        .size()
        .rename(columns={'origin_id': 'station_id', 'dep_hour': 'hour_ts', 'size': 'departures'})
    )
    arrs = (
        sub[sub['destination_id'].isin(top_ids)]
        .groupby(['destination_id', 'arr_hour'], as_index=False)
        .size()
        .rename(
            columns={'destination_id': 'station_id', 'arr_hour': 'hour_ts', 'size': 'arrivals'}
        )
    )

    all_hours = pd.date_range(
        trips['start_time'].min().floor('h'),
        trips['start_time'].max().floor('h'),
        freq='h',
    )
    grid = (
        pd.MultiIndex.from_product([top_ids, all_hours], names=['station_id', 'hour_ts'])
        .to_frame()
        .reset_index(drop=True)
    )

    p2 = (
        grid.merge(deps, on=['station_id', 'hour_ts'], how='left')
        .merge(arrs, on=['station_id', 'hour_ts'], how='left')
        .merge(active, on='station_id', how='left')
    )
    p2['departures'] = p2['departures'].fillna(0.0)
    p2['arrivals'] = p2['arrivals'].fillna(0.0)
    p2['net_flow'] = p2['arrivals'] - p2['departures']
    p2 = p2.sort_values(['station_id', 'hour_ts']).reset_index(drop=True)

    grp_flow = p2.groupby('station_id')['net_flow']
    p2['future_3h_net_flow'] = grp_flow.shift(-1) + grp_flow.shift(-2) + grp_flow.shift(-3)
    p2['pressure_ratio'] = p2['future_3h_net_flow'] / p2['capacity']

    p2['risk_class'] = 'balanceado'
    p2.loc[p2['pressure_ratio'] <= -pressure_threshold, 'risk_class'] = 'deficit'
    p2.loc[p2['pressure_ratio'] >= pressure_threshold, 'risk_class'] = 'excedente'

    # Variables predictoras estrictamente pasadas (hasta hora t)
    grp_fut3h = p2.groupby('station_id')['future_3h_net_flow']
    p2['net_flow_last_3h'] = grp_flow.rolling(3).sum().reset_index(level=0, drop=True)
    p2['pressure_last_3h'] = p2['net_flow_last_3h'] / p2['capacity']
    p2['pressure_window_lag_24h'] = grp_fut3h.shift(24) / p2['capacity']
    p2['pressure_window_lag_168h'] = grp_fut3h.shift(168) / p2['capacity']
    p2['pressure_window_lag_336h'] = grp_fut3h.shift(336) / p2['capacity']
    p2['risk_lag_168h'] = p2.groupby('station_id')['risk_class'].shift(168)

    p2['hour'] = p2['hour_ts'].dt.hour
    p2['day_of_week'] = p2['hour_ts'].dt.dayofweek
    p2['is_weekend'] = (p2['day_of_week'] >= 5).astype(int)
    p2['split'] = assign_temporal_split(p2['hour_ts'])

    p2 = p2.dropna(
        subset=[
            'future_3h_net_flow',
            'pressure_last_3h',
            'pressure_window_lag_24h',
            'pressure_window_lag_168h',
            'pressure_window_lag_336h',
            'risk_lag_168h',
        ]
    ).reset_index(drop=True)
    return p2


def build_p3_station_profiles(trips_subset: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Construye el perfil operativo de uso por estación para P3 (Clustering).

    Qué hace:
        Resume para cada estación activa la proporción de viajes iniciados en franjas
        horarias entre semana (mañana 6-10h, mediodía 11-15h, tarde 16-20h) y en fin de semana,
        la duración mediana de viaje en minutos, el porcentaje de viajes redondos
        (origen == destino), el volumen diario promedio y la razón salidas/llegadas.

    Por qué es pertinente para P3:
        Permite agrupar estaciones según su vocación real en la ciudad (laboral/receptora,
        residencial/emisora, recreativa o periférica) para apoyar decisiones de expansión
        y densificación urbana.
    """
    df = trips_subset.copy()
    df['duration_min'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60.0
    df = df[(df['duration_min'] >= 1.0) & (df['duration_min'] <= 180.0)].copy()

    df['hour'] = df['start_time'].dt.hour
    df['is_weekend'] = df['start_time'].dt.dayofweek >= 5
    df['is_round_trip'] = (df['origin_id'] == df['destination_id']).astype(int)

    n_days = max(
        (df['start_time'].max().floor('D') - df['start_time'].min().floor('D')).days + 1, 1
    )

    df['dep_morning'] = ((~df['is_weekend']) & df['hour'].between(6, 10)).astype(int)
    df['dep_midday'] = ((~df['is_weekend']) & df['hour'].between(11, 15)).astype(int)
    df['dep_evening'] = ((~df['is_weekend']) & df['hour'].between(16, 20)).astype(int)
    df['dep_weekend'] = df['is_weekend'].astype(int)

    dep_agg = (
        df.groupby('origin_id')
        .agg(
            total_deps=('trip_id', 'count'),
            dep_morning=('dep_morning', 'mean'),
            dep_midday=('dep_midday', 'mean'),
            dep_evening=('dep_evening', 'mean'),
            dep_weekend=('dep_weekend', 'mean'),
            median_duration_min=('duration_min', 'median'),
            round_trip_pct=('is_round_trip', 'mean'),
        )
        .reset_index()
        .rename(columns={'origin_id': 'station_id'})
    )

    arr_agg = (
        df.groupby('destination_id')
        .size()
        .reset_index(name='total_arrs')
        .rename(columns={'destination_id': 'station_id'})
    )

    active = stations[stations['is_active']].copy()
    prof = active.merge(dep_agg, on='station_id', how='inner').merge(
        arr_agg, on='station_id', how='left'
    )
    prof['total_arrs'] = prof['total_arrs'].fillna(0)
    prof['avg_daily_trips'] = prof['total_deps'] / n_days
    prof['dep_arr_ratio'] = prof['total_deps'] / np.maximum(prof['total_arrs'], 1)
    return prof


@app.command()
def main(
    trips_path: Path = typer.Option(TRIPS_PARQUET_PATH, help='Ruta a trips.parquet'),
    stations_path: Path = typer.Option(STATIONS_PARQUET_PATH, help='Ruta a stations.parquet'),
):
    """Genera las tablas de características de P1, P2 y P3 en data/processed/."""
    logger.info('Cargando datos limpios desde Parquet...')
    trips = pd.read_parquet(trips_path)
    stations = pd.read_parquet(stations_path)

    logger.info('Construyendo características para P1 (Regresión diaria)...')
    p1 = build_p1_daily_features(trips, stations)
    p1.to_parquet(P1_FEATURES_PATH, index=False)
    logger.success(f'P1 guardado en {P1_FEATURES_PATH} ({len(p1):,} filas)')

    logger.info('Construyendo características para P2 (Clasificación horaria)...')
    p2 = build_p2_hourly_features(trips, stations)
    p2.to_parquet(P2_FEATURES_PATH, index=False)
    logger.success(f'P2 guardado en {P2_FEATURES_PATH} ({len(p2):,} filas)')

    logger.info('Construyendo perfiles de estación para P3 (Clustering)...')
    train_trips = trips[trips['start_time'].dt.strftime('%Y-%m-%d') <= TRAIN_END_DATE]
    # Perfil de 2026 (validación + prueba) para medir la estabilidad de los grupos
    holdout_trips = trips[trips['start_time'].dt.strftime('%Y-%m-%d') >= VAL_START_DATE]

    p3_train = build_p3_station_profiles(train_trips, stations)
    p3_holdout = build_p3_station_profiles(holdout_trips, stations)
    p3_train.to_parquet(P3_TRAIN_FEATURES_PATH, index=False)
    p3_holdout.to_parquet(P3_HOLDOUT_FEATURES_PATH, index=False)
    logger.success(
        f'P3 guardado en {P3_TRAIN_FEATURES_PATH} ({len(p3_train)} estaciones) '
        f'y {P3_HOLDOUT_FEATURES_PATH} ({len(p3_holdout)} estaciones)'
    )


if __name__ == '__main__':
    app()
