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


@pytest.mark.parametrize('defect', ['missing', 'extra', 'name', 'order'])
def test_validate_header_rejects_invalid_contract(defect):
    columns = list(TRIP_SOURCE_COLUMNS)
    if defect == 'missing':
        columns.pop()
    elif defect == 'extra':
        columns.append('Extra')
    elif defect == 'name':
        columns[0] = 'Id_incorrecto'
    else:
        columns[-1], columns[-2] = columns[-2], columns[-1]
    with pytest.raises(ValueError):
        validate_header(columns)


def test_parse_datetime_mixed_formats():
    s = pd.Series(['2024-09-01 10:15:00', '02/09/2024 11:45', None], index=[5, 3, 9])
    parsed, fmt = parse_datetime(s)
    expected = pd.Series(
        [pd.Timestamp('2024-09-01 10:15'), pd.Timestamp('2024-09-02 11:45'), pd.NaT],
        index=s.index,
    )
    pd.testing.assert_series_equal(parsed, expected)
    assert fmt == '%Y-%m-%d %H:%M:%S+%d/%m/%Y %H:%M'
