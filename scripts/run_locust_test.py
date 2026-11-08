"""Ejecuta una prueba de carga con Locust contra la API y genera un reporte HTML interactivo con gráficas.

Uso:
    python scripts/run_locust_test.py --users 20 --spawn-rate 5 --run-time 15s
"""

import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Annotated

import requests
import typer

app = typer.Typer(help='Ejecuta pruebas de carga con Locust y genera reporte con gráficas.')


def wait_for_api(url: str, timeout: float = 10.0) -> bool:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            res = requests.get(f'{url.rstrip("/")}/health', timeout=1.0)
            if res.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


@app.command()
def run(
    host: Annotated[str, typer.Option(help='URL base de la API')] = 'http://127.0.0.1:8000',
    users: Annotated[int, typer.Option(help='Número máximo de usuarios concurrentes')] = 20,
    spawn_rate: Annotated[int, typer.Option(help='Usuarios generados por segundo')] = 5,
    run_time: Annotated[str, typer.Option(help='Duración de la prueba (ej: 30s, 1m)')] = '15s',
    report_path: Annotated[
        Path, typer.Option(help='Ruta del archivo HTML de reporte generado')
    ] = Path('reports/locust_report.html'),
    autostart_server: Annotated[
        bool, typer.Option(help='Iniciar automáticamente un servidor local de FastAPI si no está corriendo')
    ] = True,
):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    server_process = None

    if not wait_for_api(host, timeout=1.5):
        if not autostart_server:
            typer.echo(f'Error: La API en {host} no está disponible y autostart_server=False.', err=True)
            raise typer.Exit(code=1)

        typer.echo(f'Iniciando servidor de FastAPI temporal en {host}...')
        server_cmd = [
            sys.executable,
            '-m',
            'uvicorn',
            'mi_bici.api.main:app',
            '--host',
            '127.0.0.1',
            '--port',
            '8000',
            '--log-level',
            'warning',
        ]
        server_process = subprocess.Popen(server_cmd, env=dict(os.environ))
        if not wait_for_api(host, timeout=15.0):
            if server_process:
                server_process.terminate()
            typer.echo('Error: No se pudo levantar el servidor de FastAPI a tiempo.', err=True)
            raise typer.Exit(code=1)
        typer.echo('Servidor listo.')

    try:
        typer.echo(
            f'Ejecutando Locust: {users} usuarios, spawn-rate={spawn_rate}, duración={run_time}...'
        )
        locust_cmd = [
            sys.executable,
            '-m',
            'locust',
            '-f',
            'scripts/locustfile.py',
            '--headless',
            '-u',
            str(users),
            '-r',
            str(spawn_rate),
            '-t',
            run_time,
            '--host',
            host,
            '--html',
            str(report_path),
        ]
        res = subprocess.run(locust_cmd, env=dict(os.environ), check=False)
        if res.returncode == 0:
            typer.echo(f'\nReporte HTML con gráficas generado exitosamente en: {report_path}')
        else:
            typer.echo(f'Locust finalizó con código de salida: {res.returncode}', err=True)
            raise typer.Exit(code=res.returncode)
    finally:
        if server_process:
            typer.echo('Deteniendo servidor de FastAPI temporal...')
            server_process.terminate()
            server_process.wait()


if __name__ == '__main__':
    app()
