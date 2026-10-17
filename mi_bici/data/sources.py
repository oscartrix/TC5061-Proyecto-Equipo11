"""Descarga de datos de MiBici"""

from pathlib import Path
import re

import requests
import yaml

from mi_bici.config import PROJ_ROOT, RAW_DATA_DIR

SOURCES_PATH = PROJ_ROOT / 'config' / 'sources.yaml'

# Formato AAAA-MM con mes de 01 a 12
YEAR_MONTH_PATTERN = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')

# El servidor de MiBici responde 406 al User-Agent por defecto de requests.
HTTP_HEADERS = {
    'User-Agent': 'mi-bici-tc5061-equipo11',
    'Accept': 'text/csv,text/plain,*/*',
}


def load_sources(path: Path = SOURCES_PATH) -> dict:
    """Carga las URL de viajes y estaciones de sources.yaml"""
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def validate_year_month(value: str) -> str:
    """Validar formato (AAAA-MM) de fecha"""
    if not YEAR_MONTH_PATTERN.match(value):
        raise ValueError(f'{value!r} no tiene formato AAAA-MM')
    return value


def select_trip_urls(sources: dict, start: str, end: str) -> dict[str, str]:
    """Regresa el url de los viajes dentro del rango"""
    validate_year_month(start)
    validate_year_month(end)
    return {
        year_month: url
        for year_month, url in sorted(sources['trips'].items())
        if start <= year_month <= end
    }


def download_file(
    url: str,
    dest: Path,
    overwrite: bool = False,
    timeout: int = 60,
    http_headers: dict | None = None,
) -> Path:
    """Descarga un archivo.
    Si el archivo ya existe y overwrite es False, no lo vuelve a descargar.
    """
    dest = Path(dest)
    if dest.exists() and not overwrite:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    http_headers = HTTP_HEADERS if http_headers is None else http_headers
    with requests.get(url, headers=http_headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(tmp, 'wb') as f:
            f.writelines(response.iter_content(chunk_size=1024 * 1024))
    tmp.replace(dest)
    return dest


def download_trips(
    trip_urls: dict[str, str], raw_dir: Path = RAW_DATA_DIR, overwrite: bool = False
) -> dict[str, Path]:
    """Descarga los CSV de las url de los viajes"""
    paths = {}
    for year_month, url in trip_urls.items():
        dest = Path(raw_dir) / 'trips' / Path(url).name
        paths[year_month] = download_file(url, dest, overwrite=overwrite)
    return paths


def download_stations(
    sources: dict, raw_dir: Path = RAW_DATA_DIR, overwrite: bool = False
) -> Path:
    """Descarga el catálogo de estaciones"""
    url = sources['stations']
    dest = Path(raw_dir) / 'stations' / Path(url).name
    return download_file(url, dest, overwrite=overwrite)
