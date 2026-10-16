'''Descarga y lectura de los datos de MiBici'''

from pathlib import Path

import pandas as pd
import requests
import yaml

from mi_bici.config import PROJ_ROOT, RAW_DATA_DIR

SOURCES_PATH = PROJ_ROOT / 'config' / 'sources.yaml'
RAW_DIR = RAW_DATA_DIR

# El servidor de MiBici responde 406 al User-Agent por defecto de requests.
HEADERS = {
    'User-Agent': 'mi-bici-tc5061-equipo11',
    'Accept': 'text/csv,text/plain,*/*',
}

ENCODINGS = ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1')


def load_sources(path=SOURCES_PATH):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def select_months(sources, start, end):
    '''Filtra los meses de viajes en el rango [start, end], con formato AAAA-MM'''
    return {
        month: url
        for month, url in sorted(sources['trips'].items())
        if start <= month <= end
    }


def download_file(url, dest, overwrite=False, timeout=60, headers=None):
    '''Descarga un archivo por partes.
    Si el archivo ya existe y overwrite es False, no lo vuelve a descargar.
    '''
    dest = Path(dest)
    if dest.exists() and not overwrite:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    headers = HEADERS if headers is None else headers
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    tmp.replace(dest)
    return dest


def download_months(months, raw_dir=RAW_DIR, overwrite=False):
    paths = {}
    for month, url in months.items():
        dest = Path(raw_dir) / 'trips' / Path(url).name
        paths[month] = download_file(url, dest, overwrite=overwrite)
    return paths


def detect_encoding(path, sample_bytes=200_000):
    with open(path, 'rb') as f:
        sample = f.read(sample_bytes)
    for encoding in ENCODINGS:
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f'No se pudo detectar el encoding de {path}')


def read_header(path):
    encoding = detect_encoding(path)
    columns = pd.read_csv(path, encoding=encoding, nrows=0).columns.tolist()
    return encoding, columns


def compare_headers(paths):
    rows = []
    for month, path in paths.items():
        encoding, columns = read_header(path)
        rows.append({
            'month': month,
            'encoding': encoding,
            'n_columns': len(columns),
            'columns': tuple(columns),
        })
    df = pd.DataFrame(rows)

    df['schema'] = pd.factorize(df['columns'])[0]
    return df


def read_month(path, **kwargs):
    encoding = detect_encoding(path)
    return pd.read_csv(path, encoding=encoding, **kwargs)


# Headers
EXPECTED_COLUMNS = (
    'Viaje_Id',
    'Usuario_Id',
    'Genero',
    'Año_de_nacimiento',
    'Inicio_del_viaje',
    'Fin_del_viaje',
    'Origen_Id',
    'Destino_Id',
)

# Nombres canónicos, en el mismo orden que EXPECTED_COLUMNS
COLUMN_NAMES = (
    'trip_id',
    'user_id',
    'gender',
    'birth_year',
    'start_time',
    'end_time',
    'origin_id',
    'destination_id',
)

# Columnas que incluyen datos personales
PERSONAL_COLUMNS = ('user_id', 'gender', 'birth_year')

# 2024-02 y 2024-05 publican 'A}äe_nacimiento' en lugar de 'Año_de_nacimiento'.
TOLERATED_HEADER_POSITIONS = {3}

# Formatos de fecha conocidos, en orden de prueba.
DATETIME_FORMATS = (
    '%Y-%m-%d %H:%M:%S',
    '%d/%m/%Y %H:%M',
)


def validate_header(columns):
    '''Verifica número y orden de columnas contra EXPECTED_COLUMNS'''
    if len(columns) != len(EXPECTED_COLUMNS):
        raise ValueError(
            f'Se esperaban {len(EXPECTED_COLUMNS)} columnas y llegaron {len(columns)}: {columns}'
        )
    mismatches = [
        (i, found, expected)
        for i, (found, expected) in enumerate(zip(columns, EXPECTED_COLUMNS))
        if found != expected and i not in TOLERATED_HEADER_POSITIONS
    ]
    if mismatches:
        raise ValueError(f'Columnas inesperadas (posición, encontrada, esperada): {mismatches}')


def parse_datetime(values, formats=DATETIME_FORMATS):
    for fmt in formats:
        try:
            return pd.to_datetime(values, format=fmt), fmt
        except ValueError:
            continue
    examples = values.dropna().drop_duplicates().head(3).tolist()
    raise ValueError(f'Ningún formato conocido coincide. Formatos: {formats}. Ejemplos: {examples}')


def read_trips(path):
    encoding, columns = read_header(path)
    validate_header(columns)

    keep = [i for i, name in enumerate(COLUMN_NAMES) if name not in PERSONAL_COLUMNS]
    df = pd.read_csv(
        path,
        encoding=encoding,
        usecols=keep,
        header=0,
        names=[COLUMN_NAMES[i] for i in keep],
        dtype={'trip_id': 'int64', 'origin_id': 'Int32', 'destination_id': 'Int32'},
    )
    formats = {}
    for column in ('start_time', 'end_time'):
        df[column], formats[column] = parse_datetime(df[column])
    # Se guarda el formato usado para el reporte de calidad.
    df.attrs['datetime_formats'] = formats
    return df
