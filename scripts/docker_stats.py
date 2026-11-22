"""Guarda en CSV el CPU y la memoria de los contenedores con docker stats.

Uso:
    python scripts/docker_stats.py --output reports/carga/onnx/docker_stats.csv
    (se detiene con Ctrl+C o al terminar --seconds)
"""

import csv
import json
from pathlib import Path
import subprocess
import time
from typing import Annotated

import typer

app = typer.Typer()

UNITS = {
    'B': 1 / 2**20,
    'KiB': 1 / 1024,
    'MiB': 1,
    'GiB': 1024,
    'kB': 1 / 1024,
    'MB': 1,
    'GB': 1024,
}


def to_mb(text: str) -> float:
    """'85.6MiB' -> 85.6"""
    for unit in sorted(UNITS, key=len, reverse=True):
        if text.endswith(unit):
            return float(text[: -len(unit)]) * UNITS[unit]
    return float(text)


def running(engine: str, containers: list[str]) -> list[str]:
    """Solo los contenedores que están corriendo; docker stats falla con uno inexistente."""
    names = subprocess.run(
        [engine, 'ps', '--format', '{{.Names}}'], capture_output=True, text=True, check=True
    ).stdout.split()
    return [name for name in containers if name in names]


def read_stats(engine: str, containers: list[str]) -> list[dict]:
    command = [engine, 'stats', '--no-stream', '--format', '{{json .}}', *containers]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    rows = []
    for line in output.splitlines():
        item = json.loads(line)
        rows.append(
            {
                'container': item['Name'],
                'cpu_pct': float(item['CPUPerc'].rstrip('%')),
                'mem_mb': round(to_mb(item['MemUsage'].split('/')[0].strip()), 1),
            }
        )
    return rows


@app.command()
def main(
    output: Path = Path('reports/carga/docker_stats.csv'),
    containers: Annotated[list[str], typer.Option('--container')] = [  # noqa: B006
        'mibici-api',
        'mibici-worker',
        'mibici-redis',
        'mibici-equipo11',
    ],
    seconds: int = 600,
    engine: str = 'docker',
):
    output.parent.mkdir(parents=True, exist_ok=True)
    containers = running(engine, containers)
    start = time.time()
    with output.open('w', newline='') as file:
        writer = csv.DictWriter(file, ['timestamp', 'elapsed_s', 'container', 'cpu_pct', 'mem_mb'])
        writer.writeheader()
        try:
            while time.time() - start < seconds:
                # docker stats tarda ~2 s por lectura, no hace falta esperar más
                now = time.time()
                for row in read_stats(engine, containers):
                    writer.writerow(
                        {'timestamp': int(now), 'elapsed_s': round(now - start, 1), **row}
                    )
                file.flush()
        except KeyboardInterrupt:
            pass
    print(f'docker stats -> {output}')


if __name__ == '__main__':
    app()
