import pandas as pd
import pytest

from mi_bici.features import assign_temporal_split, build_p1_daily_features, is_mx_holiday


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


def test_p1_lag_no_usa_el_dia_actual(viajes_y_estaciones):
    viajes, estaciones = viajes_y_estaciones
    # un viaje por día, el lag_1d debe ser siempre 1 y nunca el valor del mismo día
    p1 = build_p1_daily_features(viajes, estaciones)
    assert (p1['lag_1d'] == 1).all()
    assert p1['date'].min() == pd.Timestamp('2025-01-22')


def test_p1_pocos_dias_devuelve_vacio(viajes_y_estaciones):
    viajes, estaciones = viajes_y_estaciones
    p1 = build_p1_daily_features(viajes.head(10), estaciones)
    assert p1.empty
