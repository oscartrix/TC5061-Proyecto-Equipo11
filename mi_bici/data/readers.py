"""Leer y validar los CSV de MiBici"""

from pathlib import Path

import pandas as pd

from mi_bici.data.schemas import (
    TRIP_COLUMNS,
    TRIP_DATETIME_COLUMNS,
    TRIP_DATETIME_FORMATS,
    TRIP_DTYPES,
    TRIP_HEADER_VARIANTS,
    TRIP_PERSONAL_COLUMNS,
    TRIP_SOURCE_COLUMNS,
    STATION_COLUMNS,
    STATION_DTYPES,
    STATION_HEADER_VARIANTS,
    STATION_SOURCE_COLUMNS,
)


CANDIDATE_ENCODINGS = ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1')


def detect_encoding(path: Path, sample_bytes: int = 200_000) -> str:
    with open(path, 'rb') as f:
        sample = f.read(sample_bytes)
    for encoding in CANDIDATE_ENCODINGS:
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f'No se pudo detectar el encoding de {path}')


def read_header(path: Path) -> tuple[str, list[str]]:
    """Regresa la codificación y los nombres de columna de un CSV"""
    encoding = detect_encoding(path)
    columns = pd.read_csv(path, encoding=encoding, nrows=0).columns.tolist()
    return encoding, columns


def compare_headers(trip_paths: dict[str, Path]) -> pd.DataFrame:
    """Compara los headers de todos los archivos para identificar diferencias en
    encodings, número de columnas, nombres de las columnas y el órden de las columnas.
    """
    rows = []
    for year_month, path in trip_paths.items():
        encoding, columns = read_header(path)
        rows.append(
            {
                'year_month': year_month,
                'encoding': encoding,
                'n_columns': len(columns),
                'columns': tuple(columns),
            }
        )
    df = pd.DataFrame(rows)

    df['header_variant'] = pd.factorize(df['columns'])[0]
    return df


def read_raw_csv(path: Path, **kwargs) -> pd.DataFrame:
    encoding = detect_encoding(path)
    return pd.read_csv(path, encoding=encoding, **kwargs)


def validate_header(
    columns: list[str],
    expected: tuple[str, ...] = TRIP_SOURCE_COLUMNS,
    variants: dict[str, set[str]] = TRIP_HEADER_VARIANTS,
) -> None:
    """Verifica número, nombre y orden de columnas contra el contrato"""
    if len(columns) != len(expected):
        raise ValueError(
            f'Se esperaban {len(expected)} columnas y llegaron {len(columns)}: {columns}'
        )
    
    mismatches = [
        (i, found, name)
        for i, (found, name) in enumerate(zip(columns, expected))
        if found != name and found not in variants.get(name, set())
    ]

    if mismatches:
        raise ValueError(f'Columnas con diferencia en el contrato: {mismatches}')


def parse_datetime(
    values: pd.Series, formats: tuple[str, ...] = TRIP_DATETIME_FORMATS
) -> tuple[pd.Series, str]:
    """Convierte el valor a fecha en un formato válido"""
    for fmt in formats:
        try:
            return pd.to_datetime(values, format=fmt), fmt
        except ValueError:
            continue

    examples = values.dropna().drop_duplicates().head(3).tolist()

    raise ValueError(
        f'La fecha no coincide con ningún formato válido. Formatos: {formats}. Ejemplos: {examples}'
    )


def read_trips(path: Path) -> pd.DataFrame:
    encoding, columns = read_header(path)
    validate_header(columns)

    keep_positions = [
        i for i, name in enumerate(TRIP_COLUMNS) if name not in TRIP_PERSONAL_COLUMNS
    ]

    df = pd.read_csv(
        path,
        encoding=encoding,
        usecols=keep_positions,
        header=0,
        names=[TRIP_COLUMNS[i] for i in keep_positions],
        dtype=TRIP_DTYPES,
    )
    datetime_formats = {}

    for column in TRIP_DATETIME_COLUMNS:
        df[column], datetime_formats[column] = parse_datetime(df[column])
    
    df.attrs['datetime_formats'] = datetime_formats
    return df


def read_stations(path: Path) -> pd.DataFrame:
    encoding, columns = read_header(path)
    validate_header(columns, STATION_SOURCE_COLUMNS, STATION_HEADER_VARIANTS)
    return pd.read_csv(
        path,
        encoding=encoding,
        header=0,
        names=list(STATION_COLUMNS),
        dtype=STATION_DTYPES,
    )
