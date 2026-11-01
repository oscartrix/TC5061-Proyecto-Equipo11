import pandas as pd
import pytest

from mi_bici.data.readers import detect_encoding, parse_datetime
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
