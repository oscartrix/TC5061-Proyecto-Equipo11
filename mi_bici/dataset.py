"""Comando para ingesta de datos.

Uso:
    python -m mi_bici.dataset --start 2024-01 --end 2026-09
"""

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from mi_bici.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from mi_bici.data import (
    download_stations,
    download_trips,
    load_sources,
    select_trip_urls,
    compare_headers,
    validate_header,
    validate_year_month,
    read_stations,
)

app = typer.Typer()


def year_month_option(value: str) -> str:
    """Validar formato de fecha."""
    try:
        return validate_year_month(value)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


@app.command()
def main(
    start: str = typer.Option(
        '2024-01', help='Fecha inicio, formato AAAA-MM', callback=year_month_option
    ),
    end: str = typer.Option(
        '2026-09', help='Fecha fin, formato AAAA-MM', callback=year_month_option
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
    logger.info(f'Catálogo de estaciones: {stations_path}')

    trip_paths = download_trips(trip_urls, overwrite=overwrite)
    total_mb = sum(path.stat().st_size for path in trip_paths.values()) / 1e6
    logger.success(f'Se descargaron los viajes de {len(trip_paths)} meses. Total: {total_mb:,.0f} MB')

    # Viajes: revisar la estructura de los encabezados de cada archivo
    header_summary = compare_headers(trip_paths)
    for year_month, columns in zip(header_summary['year_month'], header_summary['columns']):
        try:
            validate_header(list(columns))
        except ValueError as error:
            raise ValueError(f'Viajes: encabezado inválido en {year_month}: {error}') from error

    encodings = header_summary['encoding'].value_counts().to_dict()
    logger.success(f'Viajes: encabezados válidos. Encodings: {encodings}')


if __name__ == '__main__':
    app()
