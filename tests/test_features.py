import pandas as pd
import pytest

from mi_bici.features import (
    assign_temporal_split,
    build_p1_daily_features,
    build_p2_hourly_features,
    build_p3_station_profiles,
    is_mx_holiday,
)


@pytest.mark.parametrize(
    'fecha, esperado',
    [
        ('2025-01-01', True),
        ('2025-12-25', True),
        ('2025-02-03', True),  # primer lunes de febrero
        ('2025-02-10', False),  # segundo lunes
        ('2025-03-17', True),  # tercer lunes de marzo
        ('2025-11-17', True),  # tercer lunes de noviembre
        ('2024-10-01', True),  # cambio de gobierno
        ('2025-10-01', False),
        ('2025-06-15', False),
    ],
)
def test_is_mx_holiday(fecha, esperado):
    assert is_mx_holiday(pd.Timestamp(fecha)) is esperado


def test_assign_temporal_split_limites():
    fechas = pd.Series(
        pd.to_datetime(
            ['2025-12-31', '2026-01-01', '2026-06-30', '2026-07-01']
        )
    )
    assert assign_temporal_split(fechas).tolist() == ['train', 'val', 'val', 'test']


def test_assign_temporal_split_serie_vacia():
    vacia = pd.Series(pd.to_datetime([]))
    assert assign_temporal_split(vacia).empty


@pytest.fixture
def viajes_y_estaciones():
    dias = pd.date_range('2025-01-01', periods=40, freq='D')
    viajes = pd.DataFrame({'origin_id': 1, 'start_time': dias})
    estaciones = pd.DataFrame(
        {
            'station_id': [1, 2],
            'capacity': [20, 15],
            'zone': ['centro', 'norte'],
            'is_active': [True, False],
        }
    )
    return viajes, estaciones


def test_p1_columnas_y_sin_nulos(viajes_y_estaciones):
    p1 = build_p1_daily_features(*viajes_y_estaciones)
    for col in ['lag_1d', 'lag_7d', 'lag_14d', 'lag_21d', 'rolling_mean_7d']:
        assert p1[col].notna().all()
    assert {'split', 'is_holiday', 'is_weekend', 'trips_count'} <= set(p1.columns)


def test_p1_solo_estaciones_activas(viajes_y_estaciones):
    p1 = build_p1_daily_features(*viajes_y_estaciones)
    assert set(p1['station_id']) == {1}


@pytest.fixture
def daily_history():
    dates = pd.date_range('2025-12-01', periods=45, freq='D')
    station_1_counts = list(range(1, 46))
    station_1_counts[27] = 0  # 28 de diciembre: sin viajes, pero el día debe existir en la malla.
    station_2_counts = [2 + 3 * (i % 5) for i in range(45)]
    trips = pd.concat([
        pd.DataFrame({'origin_id': 1, 'start_time': dates.repeat(station_1_counts)}),
        pd.DataFrame({'origin_id': 2, 'start_time': dates.repeat(station_2_counts)}),
    ], ignore_index=True)
    stations = pd.DataFrame({
        'station_id': [1, 2], 'capacity': [20, 30],
        'zone': ['GDL', 'ZAP'], 'is_active': [True, True],
    })
    return trips, stations


P1_HISTORY = ['lag_1d', 'lag_7d', 'lag_14d', 'lag_21d', 'rolling_mean_7d']


@pytest.mark.parametrize('station,date,expected', [
    (1, '2025-12-22', [21., 15., 8., 1., 18.]),
    (2, '2025-12-22', [2., 14., 8., 2., 8.]),
    (1, '2025-12-31', [30., 24., 17., 10., 161. / 7]),
    (2, '2025-12-31', [14., 11., 5., 14., 65. / 7]),
])
def test_p1_exact_lags_and_rolling_mean_per_station(daily_history, station, date, expected):
    p1 = build_p1_daily_features(*daily_history).set_index(['station_id', 'date'])
    row = p1.loc[(station, pd.Timestamp(date))]
    assert row[P1_HISTORY].tolist() == pytest.approx(expected)
    assert p1.loc[station].index.min() == pd.Timestamp('2025-12-22')


def test_p1_days_without_trips_count_as_zero(daily_history):
    p1 = build_p1_daily_features(*daily_history).set_index(['station_id', 'date'])
    assert p1.loc[(1, pd.Timestamp('2025-12-28')), 'trips_count'] == 0
    assert p1.loc[(1, pd.Timestamp('2025-12-29')), 'lag_1d'] == 0


@pytest.mark.parametrize('cutoff', ['2025-12-31', '2026-01-01'])
def test_p1_current_and_future_changes_preserve_past_lags(daily_history, cutoff):
    trips, stations = daily_history
    cutoff = pd.Timestamp(cutoff)
    original = build_p1_daily_features(trips, stations)
    additional_trips = trips.loc[trips.start_time >= cutoff]
    modified = build_p1_daily_features(
        pd.concat([trips, additional_trips, additional_trips], ignore_index=True), stations
    )
    keys = ['station_id', 'date']
    expected = original.loc[original.date <= cutoff].set_index(keys)[P1_HISTORY]
    actual = modified.loc[modified.date <= cutoff].set_index(keys)[P1_HISTORY]
    pd.testing.assert_frame_equal(actual, expected)
    # La perturbación sí cambia el objetivo en el corte: la prueba no es vacua.
    assert modified.loc[modified.date == cutoff, 'trips_count'].sum() == (
        3 * original.loc[original.date == cutoff, 'trips_count'].sum()
    )


def test_p1_removing_future_preserves_earlier_rows(daily_history):
    trips, stations = daily_history
    cutoff = pd.Timestamp('2026-01-01')
    complete = build_p1_daily_features(trips, stations)
    truncated = build_p1_daily_features(trips.loc[trips.start_time <= cutoff], stations)
    keys = ['station_id', 'date']
    pd.testing.assert_frame_equal(
        truncated.set_index(keys), complete.loc[complete.date <= cutoff].set_index(keys)
    )


def test_p1_independent_of_trip_order(daily_history):
    trips, stations = daily_history
    pd.testing.assert_frame_equal(
        build_p1_daily_features(trips.sample(frac=1, random_state=42), stations),
        build_p1_daily_features(trips, stations),
    )


def test_p1_new_station_requires_own_history(daily_history):
    trips, stations = daily_history
    trips = trips.loc[(trips.origin_id == 1) | (trips.start_time >= '2025-12-10')]
    p1 = build_p1_daily_features(trips, stations)
    new_station = p1.loc[p1.station_id == 2]
    assert new_station.date.min() == pd.Timestamp('2025-12-31')
    assert new_station.iloc[0]['lag_21d'] == 14
    assert p1.loc[p1.station_id == 1, 'date'].min() == pd.Timestamp('2025-12-22')


@pytest.mark.parametrize('days,expected_rows', [(1, 0), (20, 0), (21, 0), (22, 1)])
def test_p1_history_length_boundary(viajes_y_estaciones, days, expected_rows):
    trips, stations = viajes_y_estaciones
    p1 = build_p1_daily_features(trips.head(days), stations)
    assert len(p1) == expected_rows


def test_temporal_split_preserves_hour_boundaries_and_indices():
    dates = pd.Series(pd.to_datetime([
        '2026-07-01 00:00:00', '2025-12-31 23:59:59',
        '2026-06-30 23:59:59', '2026-01-01 00:00:00',
    ]), index=[9, 4, 7, 2])
    expected = pd.Series(['test', 'train', 'val', 'val'], index=dates.index, dtype='string')
    pd.testing.assert_series_equal(assign_temporal_split(dates), expected)


@pytest.fixture
def hourly_history():
    hours = pd.date_range('2025-12-15', periods=24 * 20, freq='h')
    departures = [1 + i % 5 for i in range(len(hours))]
    arrivals = [i % 4 for i in range(len(hours))]
    trips = pd.concat([
        pd.DataFrame({
            'origin_id': 1, 'destination_id': 2,
            'start_time': hours.repeat(departures) + pd.Timedelta(minutes=5),
        }),
        pd.DataFrame({
            'origin_id': 2, 'destination_id': 1,
            'start_time': hours.repeat(arrivals) + pd.Timedelta(minutes=10),
        }),
    ], ignore_index=True)
    trips['end_time'] = trips.start_time + pd.Timedelta(minutes=5)
    stations = pd.DataFrame({
        'station_id': [1, 2], 'capacity': [10, 10],
        'zone': ['GDL', 'ZAP'], 'is_active': [True, True],
    })
    return trips, stations


def test_p2_exact_flows_lags_and_target(hourly_history):
    p2 = build_p2_hourly_features(*hourly_history).set_index(['station_id', 'hour_ts'])
    when = pd.Timestamp('2025-12-31 12:00')
    row = p2.loc[(1, when)]
    assert row[['departures', 'arrivals', 'net_flow', 'net_flow_last_3h',
                'future_3h_net_flow']].tolist() == [2., 0., -2., -3., -6.]
    assert row[['pressure_last_3h', 'pressure_window_lag_24h',
                'pressure_window_lag_168h', 'pressure_window_lag_336h']].tolist() == (
        pytest.approx([-.3, -.4, -.2, -.3])
    )
    assert row['risk_class'] == 'deficit'
    assert row['risk_lag_168h'] == 'balanceado'
    assert p2.loc[(2, when), 'risk_class'] == 'excedente'


def test_p2_future_changes_target_but_not_predictors(hourly_history):
    trips, stations = hourly_history
    cutoff = pd.Timestamp('2025-12-31 12:00')
    original = build_p2_hourly_features(trips, stations)
    additional_trips = trips.loc[trips.start_time >= cutoff + pd.Timedelta(hours=1)]
    modified = build_p2_hourly_features(pd.concat([trips, additional_trips]), stations)
    predictors = [
        'hour', 'day_of_week', 'is_weekend', 'departures', 'arrivals', 'net_flow',
        'net_flow_last_3h', 'pressure_last_3h', 'pressure_window_lag_24h',
        'pressure_window_lag_168h', 'pressure_window_lag_336h', 'risk_lag_168h',
    ]
    keys = ['station_id', 'hour_ts']
    pd.testing.assert_frame_equal(
        modified.loc[modified.hour_ts <= cutoff].set_index(keys)[predictors],
        original.loc[original.hour_ts <= cutoff].set_index(keys)[predictors],
    )
    key = (1, cutoff)
    assert modified.set_index(keys).loc[key, 'future_3h_net_flow'] == (
        2 * original.set_index(keys).loc[key, 'future_3h_net_flow']
    )


def test_p2_station_selection_uses_only_training_data(hourly_history):
    trips, stations = hourly_history
    future_trips = trips.loc[(trips.start_time >= '2026-01-01') & (trips.origin_id == 2)]
    expanded = pd.concat([trips, *([future_trips] * 30)], ignore_index=True)
    assert expanded.origin_id.value_counts().idxmax() == 2
    for trips_input in (trips, expanded):
        p2 = build_p2_hourly_features(trips_input, stations, top_n_stations=1)
        assert not p2.empty
        assert set(p2.station_id) == {1}


@pytest.fixture
def profile_trips():
    start_times = pd.to_datetime([
        '2025-01-06 06:00', '2025-01-06 11:00', '2025-01-06 16:00',
        '2025-01-11 12:00', '2025-01-11 13:00', '2025-01-11 14:00', '2025-01-06 08:00',
    ])
    trips = pd.DataFrame({
        'trip_id': range(1, 8), 'origin_id': [1, 1, 1, 1, 1, 1, 2],
        'destination_id': [2, 1, 2, 2, 2, 2, 1], 'start_time': start_times,
        'end_time': start_times + pd.to_timedelta([1, 10, 180, 20, .5, 181, 30], unit='min'),
    })
    stations = pd.DataFrame({
        'station_id': [1, 2], 'is_active': [True, False],
        'latitude': [20.67, 20.68], 'longitude': [-103.35, -103.36],
    })
    return trips, stations


def test_p3_exact_profile_and_duration_boundaries(profile_trips):
    profile = build_p3_station_profiles(*profile_trips)
    assert profile.station_id.tolist() == [1]
    row = profile.iloc[0]
    assert row['total_deps'] == 4  # Incluye 1 y 180 minutos; excluye .5 y 181.
    assert row['total_arrs'] == 2
    assert row['median_duration_min'] == 15
    assert row['avg_daily_trips'] == pytest.approx(4 / 6)
    assert row['dep_arr_ratio'] == 2
    assert row[['dep_morning', 'dep_midday', 'dep_evening', 'dep_weekend',
                'round_trip_pct']].tolist() == [.25] * 5


def test_p3_no_arrivals_avoids_division_by_zero(profile_trips):
    trips, stations = profile_trips
    trips = trips.loc[trips.origin_id == 1].assign(destination_id=2)
    row = build_p3_station_profiles(trips, stations).iloc[0]
    assert row['total_arrs'] == 0
    assert row['dep_arr_ratio'] == 4
