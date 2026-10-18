import pandas as pd
import pytest

from mi_bici.data import validate_header, validate_year_month
from mi_bici.data.readers import parse_datetime
from mi_bici.data.schemas import TRIP_SOURCE_COLUMNS


def test_validate_year_month():
    assert validate_year_month('2024-01') == '2024-01'
    with pytest.raises(ValueError):
        validate_year_month('2024-13')


def test_validate_header_variants():
    cols_ok = list(TRIP_SOURCE_COLUMNS)
    validate_header(cols_ok)

    cols_variant = list(TRIP_SOURCE_COLUMNS)
    cols_variant[3] = 'A}äe_nacimiento'
    validate_header(cols_variant)


def test_parse_datetime_mixed_formats():
    s = pd.Series(['01/09/2024 0:00', '01/09/2024 10:15'])
    parsed, fmt = parse_datetime(s)
    assert parsed.notna().all()
    assert fmt == '%d/%m/%Y %H:%M'
