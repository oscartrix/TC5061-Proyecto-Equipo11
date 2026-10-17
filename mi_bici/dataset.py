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
)

app = typer.Typer()


@app.command()
def main(
    start: str = typer.Option('2024-01', help='Fecha inicio, formato AAAA-MM'),
    end: str = typer.Option('2026-09', help='Fecha fin, formato AAAA-MM'),
    overwrite: bool = typer.Option(False, help='Descargar aunque el archivo ya exista'),
):

    sources = load_sources()
    trip_urls = select_trip_urls(sources, start, end)

    expected = {str(p) for p in pd.period_range(start, end, freq='M')}
    missing = sorted(expected - set(trip_urls))
    if missing:
        logger.warning(f'Fechas sin URL en config/sources.yaml: {missing}')
    if not trip_urls:
        raise typer.BadParameter(f'No hay archivos disponibles entre {start} y {end}')

    stations_path = download_stations(sources, overwrite=overwrite)
    logger.info(f'Catálogo de estaciones: {stations_path}')

    trip_paths = download_trips(trip_urls, overwrite=overwrite)
    total_mb = sum(path.stat().st_size for path in trip_paths.values()) / 1e6
    logger.success(f'Se descargaron los viajes de {len(trip_paths)} meses. Total: {total_mb:,.0f} MB')


if __name__ == '__main__':
    app()
