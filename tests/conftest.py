"""Datos pequeños y completos para validar los contratos de características."""

import pandas as pd
import pytest


@pytest.fixture
def valid_feature_frames():
    dates = pd.Series(pd.to_datetime([
        '2025-01-01', '2025-01-02', '2026-01-01',
        '2026-06-30', '2026-07-01', '2026-09-30',
    ]))
    common = {
        'station_id': range(1, 7),
        'capacity': range(20, 26),
        'zone': ['GDL'] * 6,
    }
    calendar = {
        'day_of_week': dates.dt.dayofweek,
        'is_weekend': (dates.dt.dayofweek >= 5).astype(int),
        'split': ['train', 'train', 'val', 'val', 'test', 'test'],
    }
    p1 = pd.DataFrame({
        **common, **calendar, 'date': dates, 'trips_count': [10.] * 6,
        'lag_1d': [9.] * 6, 'lag_7d': [8.] * 6, 'lag_14d': [7.] * 6,
        'lag_21d': [6.] * 6, 'rolling_mean_7d': [8.] * 6,
        'month': dates.dt.month, 'is_holiday': [0] * 6,
    })
    p2 = pd.DataFrame({
        **common, **calendar, 'hour_ts': dates, 'hour': [0] * 6,
        'departures': [2.] * 6, 'arrivals': [1.] * 6, 'net_flow': [-1.] * 6,
        'future_3h_net_flow': [-1.] * 6, 'pressure_ratio': [-.05] * 6,
        'net_flow_last_3h': [-1.] * 6, 'pressure_last_3h': [-.05] * 6,
        'pressure_window_lag_24h': [-.05] * 6,
        'pressure_window_lag_168h': [-.05] * 6,
        'pressure_window_lag_336h': [-.05] * 6,
        'risk_class': ['balanceado'] * 6, 'risk_lag_168h': ['balanceado'] * 6,
    })
    p3 = pd.DataFrame({
        **common, 'name': [f'GDL-{i:03d}' for i in range(1, 7)],
        'latitude': [20.67] * 6, 'longitude': [-103.35] * 6,
        'is_ex': [False] * 6, 'is_active': [True] * 6,
        'total_deps': [20] * 6, 'total_arrs': [15] * 6,
        'dep_morning': [.25] * 6, 'dep_midday': [.25] * 6,
        'dep_evening': [.25] * 6, 'dep_weekend': [.25] * 6,
        'median_duration_min': [10.] * 6, 'round_trip_pct': [.1] * 6,
        'avg_daily_trips': [2.] * 6, 'dep_arr_ratio': [1.5] * 6,
    })
    return {'P1': p1, 'P2': p2, 'P3_train': p3, 'P3_holdout': p3.copy()}
