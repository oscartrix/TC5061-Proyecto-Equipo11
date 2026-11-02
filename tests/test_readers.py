import pandas as pd
import pytest

from mi_bici.data.readers import detect_encoding, parse_datetime, read_stations, read_trips
from mi_bici.data.schemas import TRIP_SOURCE_COLUMNS
from mi_bici.data.sources import select_trip_urls, validate_year_month


def test_detect_encoding_utf8(tmp_path):
    archivo = tmp_path / 'a.csv'
    archivo.write_text('Año,Género\n', encoding='utf-8')
    assert detect_encoding(archivo).startswith('utf-8')


def test_detect_encoding_latin(tmp_path):
    archivo = tmp_path / 'b.csv'
    archivo.write_bytes('Año,Género\n'.encode('latin-1'))
    assert not detect_encoding(archivo).startswith('utf-8')


def test_parse_datetime_formato_invalido():
    with pytest.raises(ValueError):
        parse_datetime(pd.Series(['no es fecha', 'tampoco']))


def test_parse_datetime_con_nulos():
    parsed, _ = parse_datetime(pd.Series(['01/09/2024 0:00', None]))
    assert parsed.notna().sum() == 1


@pytest.mark.parametrize('valor', ['2024-1', '24-01', '2024/01', '', '2024-00'])
def test_validate_year_month_invalidos(valor):
    with pytest.raises(ValueError):
        validate_year_month(valor)


def test_select_trip_urls_rango():
    fuentes = {'trips': {'2024-01': 'a', '2024-02': 'b', '2024-03': 'c'}}
    assert select_trip_urls(fuentes, '2024-02', '2024-03') == {'2024-02': 'b', '2024-03': 'c'}
    assert select_trip_urls(fuentes, '2025-01', '2025-02') == {}


@pytest.mark.parametrize('invalid', ['no es fecha', '31/02/2024 12:00'])
def test_parse_datetime_rejects_errors_among_valid_dates(invalid):
    values = pd.Series(['2024-09-01 10:00:00', invalid, '02/09/2024 11:00'])
    with pytest.raises(ValueError, match='ningún formato válido'):
        parse_datetime(values)


@pytest.mark.parametrize('encoding', ['utf-8', 'latin-1'])
def test_read_trips_cleans_data_and_excludes_personal_columns(tmp_path, encoding):
    base = {
        'Viaje_Id': 1, 'Usuario_Id': 'usuario-de-prueba', 'Genero': 'dato omitido',
        'Año_de_nacimiento': 'no cargar', 'Inicio_del_viaje': '2024-09-01 10:00:00',
        'Fin_del_viaje': '2024-09-01 10:10:00', 'Origen_Id': 10, 'Destino_Id': 20,
    }
    rows = [
        base,
        {**base, 'Usuario_Id': 'otro-usuario'},  # Duplicado tras excluir datos personales.
        {**base, 'Viaje_Id': 2, 'Fin_del_viaje': '2024-09-01 09:59:00'},
        {**base, 'Viaje_Id': 3, 'Origen_Id': None},
        {**base, 'Viaje_Id': 4, 'Destino_Id': None},
        {**base, 'Viaje_Id': 5, 'Inicio_del_viaje': None},
        {**base, 'Viaje_Id': 6, 'Fin_del_viaje': None},
        {**base, 'Viaje_Id': 7, 'Origen_Id': 11, 'Destino_Id': 21,
         'Inicio_del_viaje': '02/09/2024 11:00', 'Fin_del_viaje': '02/09/2024 11:15'},
    ]
    path = tmp_path / 'trips.csv'
    pd.DataFrame(rows, columns=TRIP_SOURCE_COLUMNS).to_csv(path, index=False, encoding=encoding)

    result = read_trips(path, year_month='2024-09')

    assert set(result.columns) == {
        'trip_id', 'start_time', 'end_time', 'origin_id', 'destination_id', 'year_month'
    }
    assert result.trip_id.tolist() == [1, 7]
    assert result.origin_id.tolist() == [10, 11]
    assert result.destination_id.tolist() == [20, 21]
    assert result.start_time.tolist() == [
        pd.Timestamp('2024-09-01 10:00'), pd.Timestamp('2024-09-02 11:00')
    ]
    assert result.end_time.tolist() == [
        pd.Timestamp('2024-09-01 10:10'), pd.Timestamp('2024-09-02 11:15')
    ]
    assert result.year_month.tolist() == ['2024-09', '2024-09']
    assert str(result.origin_id.dtype) == 'Int32'
    assert not result.duplicated().any()


def test_read_stations_detects_inactive_stations_and_zones(tmp_path):
    path = tmp_path / 'stations.csv'
    pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['GDL-001', 'ex-ZAP-002', 'TLA-003', 'Nombre sin código'],
        'latitude': [20.67] * 4, 'longitude': [-103.35] * 4,
        'dpcapacity': [20, 15, 0, 10],
    }).to_csv(path, index=False)

    result = read_stations(path)

    assert result.station_id.tolist() == [1, 2, 3, 4]
    assert result.zone.tolist() == ['GDL', 'ZAP', 'TLA', 'OTRO']
    assert result.is_ex.tolist() == [False, True, False, False]
    assert result.is_active.tolist() == [True, False, False, True]
