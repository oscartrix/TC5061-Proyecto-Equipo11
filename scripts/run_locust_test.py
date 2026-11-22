"""Ejecuta una prueba de carga con Locust contra la API y genera un reporte HTML interactivo con gráficas.

Uso:
    python scripts/run_locust_test.py --users 20 --spawn-rate 5 --run-time 15s

    # Escalones de usuarios contra docker compose, con CSV y docker stats:
    python scripts/run_locust_test.py --no-autostart-server --steps 20,50,100,150,200 \
        --output-dir reports/carga/onnx --stats
"""

import os
from pathlib import Path
import signal
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
    steps: Annotated[
        str | None, typer.Option(help='Usuarios por escalón, ej: 20,50,100 (ignora --users)')
    ] = None,
    step_seconds: Annotated[int, typer.Option(help='Duración de cada escalón')] = 60,
    output_dir: Annotated[
        Path | None, typer.Option(help='Carpeta para el HTML, los CSV de Locust y docker stats')
    ] = None,
    stats: Annotated[bool, typer.Option(help='Guardar docker stats durante la prueba')] = False,
    async_tasks: Annotated[bool, typer.Option(help='Incluir /p1/predict-async (requiere Redis)')] = False,
):
    if output_dir:
        report_path = output_dir / 'locust.html'
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
        if steps:
            # La forma de carga termina sola; -t solo es el límite
            run_time = f'{len(steps.split(",")) * step_seconds + 5}s'
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
        env = dict(os.environ)
        if steps:
            env.update(LOCUST_STEPS=steps, LOCUST_STEP_SECONDS=str(step_seconds))
        if async_tasks:
            env['LOCUST_ASYNC'] = '1'
        if output_dir:
            # locust_stats.csv, locust_stats_history.csv y locust_failures.csv
            locust_cmd += ['--csv', str(output_dir / 'locust'), '--csv-full-history']
        stats_process = None
        if stats:
            stats_csv = report_path.parent / 'docker_stats.csv'
            stats_process = subprocess.Popen(
                [sys.executable, 'scripts/docker_stats.py', '--output', str(stats_csv)]
            )
        try:
            res = subprocess.run(locust_cmd, env=env, check=False)
        finally:
            if stats_process:
                stats_process.send_signal(signal.SIGINT)
                stats_process.wait()
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
