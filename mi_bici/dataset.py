"""Comando para ingesta, limpieza a Parquet y versionado de datos.

Uso:
    python -m mi_bici.dataset --start 2024-01 --end 2026-09
"""

import hashlib
import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mi_bici.config import (
    END_MONTH,
    MANIFEST_PATH,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    START_MONTH,
    STATIONS_PARQUET_PATH,
    TRIPS_PARQUET_PATH,
)
from mi_bici.data import (
    download_stations,
    download_trips,
    load_sources,
    select_trip_urls,
    compare_headers,
    validate_header,
    validate_year_month,
    read_stations,
    read_trips,
)

app = typer.Typer()


def year_month_option(value: str) -> str:
    """Validar formato de fecha."""
    try:
        return validate_year_month(value)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


def compute_sha256(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@app.command()
def main(
    start: str = typer.Option(
        START_MONTH, help='Fecha inicio, formato AAAA-MM', callback=year_month_option
    ),
    end: str = typer.Option(
        END_MONTH, help='Fecha fin, formato AAAA-MM', callback=year_month_option
    ),
    overwrite: bool = typer.Option(False, help='Descargar aunque el archivo ya exista'),
):

    if start > end:
        raise typer.BadParameter(f'La fecha de inicio ({start}) es posterior a la fecha fin ({end})')

    sources = load_sources()
    trip_urls = select_trip_urls(sources, start, end)

    expected = {str(p) for p in pd.period_range(start, end, freq='M')}
    missing = sorted(expected - set(trip_urls))
    if missing:
        logger.warning(f'Fechas sin URL en config/sources.yaml: {missing}')
    if not trip_urls:
        raise typer.BadParameter(f'No hay archivos disponibles entre {start} y {end}')

    stations_path = download_stations(sources, overwrite=overwrite)
    try:
        stations = read_stations(stations_path)
    except ValueError as error:
        raise ValueError(
            f'Catálogo de estaciones inválido ({stations_path.name}): {error}'
        ) from error
    logger.info(f'Catálogo de estaciones validado: {len(stations)} estaciones')

    trip_paths = download_trips(trip_urls, overwrite=overwrite)
    total_mb = sum(path.stat().st_size for path in trip_paths.values()) / 1e6
    logger.success(f'Se descargaron los viajes de {len(trip_paths)} meses. Total: {total_mb:,.1f} MB')

    # Viajes: revisar la estructura de los encabezados de cada archivo
    header_summary = compare_headers(trip_paths)
    for year_month, columns in zip(header_summary['year_month'], header_summary['columns']):
        try:
            validate_header(list(columns))
        except ValueError as error:
            raise ValueError(f'Viajes: encabezado inválido en {year_month}: {error}') from error

    encodings = header_summary['encoding'].value_counts().to_dict()
    logger.success(f'Viajes: encabezados válidos. Encodings: {encodings}')

    # Normalizar y guardar datos limpios en Parquet
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    monthly_frames = []
    raw_files_manifest = []

    for year_month, csv_path in sorted(trip_paths.items()):
        df_month = read_trips(csv_path, year_month=year_month)
        monthly_frames.append(df_month)
        raw_files_manifest.append(
            {
                'year_month': year_month,
                'file': csv_path.name,
                'rows_clean': int(len(df_month)),
                'size_bytes': int(csv_path.stat().st_size),
                'sha256': compute_sha256(csv_path),
            }
        )

    trips = pd.concat(monthly_frames, ignore_index=True)
    trips.to_parquet(TRIPS_PARQUET_PATH, index=False)
    stations.to_parquet(STATIONS_PARQUET_PATH, index=False)

    trips_sha = compute_sha256(TRIPS_PARQUET_PATH)
    stations_sha = compute_sha256(STATIONS_PARQUET_PATH)
    raw_stations_sha = compute_sha256(stations_path)

    # La versión se calcula con los CSV crudos: el binario Parquet cambia según la versión de pyarrow
    raw_hashes = [item['sha256'] for item in raw_files_manifest] + [raw_stations_sha]
    data_version = hashlib.sha256(':'.join(raw_hashes).encode()).hexdigest()[:12]

    manifest = {
        'data_version': data_version,
        'start_month': start,
        'end_month': end,
        'n_months': len(trip_paths),
        'total_clean_trips': int(len(trips)),
        'total_stations': int(len(stations)),
        'active_stations': int(stations['is_active'].sum()),
        'processed_files': {
            'trips_parquet': {
                'path': TRIPS_PARQUET_PATH.relative_to(PROJ_ROOT).as_posix(),
                'rows': int(len(trips)),
                'size_bytes': int(TRIPS_PARQUET_PATH.stat().st_size),
                'sha256': trips_sha,
            },
            'stations_parquet': {
                'path': STATIONS_PARQUET_PATH.relative_to(PROJ_ROOT).as_posix(),
                'rows': int(len(stations)),
                'size_bytes': int(STATIONS_PARQUET_PATH.stat().st_size),
                'sha256': stations_sha,
            },
        },
        'raw_stations_file': {
            'file': stations_path.name,
            'size_bytes': int(stations_path.stat().st_size),
            'sha256': raw_stations_sha,
        },
        'raw_trip_files': raw_files_manifest,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.success(
        f'Datos guardados en Parquet ({len(trips):,} viajes, {len(stations)} estaciones). '
        f'Versión de datos: {data_version} -> {MANIFEST_PATH}'
    )


if __name__ == '__main__':
    app()
