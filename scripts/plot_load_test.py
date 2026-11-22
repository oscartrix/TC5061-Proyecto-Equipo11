"""Gráficas y resumen por escalón de las pruebas de carga (Locust + docker stats).

Uso:
    python scripts/plot_load_test.py --runs reports/carga/onnx --runs reports/carga/sklearn
"""

from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import typer

app = typer.Typer()

# Mismo color para cada motor en todas las gráficas; la corrida sin ajuste va punteada
COLORS = {'onnx': '#2a78d6', 'sklearn': '#eb6834', 'onnx-8-hilos': '#1baf7a'}
DASHED = {'onnx-8-hilos'}
ONE = '/p1/predict (1 instancia)'
BATCH = '/p1/predict (batch 10)'
# Segundos al inicio de cada escalón que no se cuentan (los usuarios van llegando)
WARMUP_S = 10
# Los conteos intermedios mientras sube la carga duran 1 o 2 s y no son escalones
MIN_STEP_S = 30


def load_run(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = pd.read_csv(folder / 'locust_stats_history.csv')
    history = history[history['User Count'] > 0]
    stats = pd.read_csv(folder / 'docker_stats.csv')
    stats = stats[stats['container'] == 'mibici-api']
    start = history['Timestamp'].min()
    history = history.assign(t=history['Timestamp'] - start)
    stats = stats.assign(t=stats['timestamp'] - start)
    return history, stats


def summarize(history: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Una fila por escalón de usuarios, sin los primeros segundos de cada uno."""
    rows = []
    for users, step in history.groupby('User Count'):
        if step['t'].max() - step['t'].min() < MIN_STEP_S:
            continue
        steady = step[step['t'] >= step['t'].min() + WARMUP_S]
        total = steady[steady['Name'] == 'Aggregated']
        one = steady[steady['Name'] == ONE]
        batch = steady[steady['Name'] == BATCH]
        requests = total['Total Request Count']
        failures = total['Total Failure Count']
        sent = requests.max() - requests.min()
        cpu = stats[stats['t'].between(steady['t'].min(), steady['t'].max())]
        rows.append(
            {
                'usuarios': users,
                'rps_total': total['Requests/s'].mean(),
                'rps_prediccion': one['Requests/s'].mean() + batch['Requests/s'].mean(),
                'filas_s': one['Requests/s'].mean() + 10 * batch['Requests/s'].mean(),
                'p50_ms': pd.to_numeric(one['50%'], errors='coerce').median(),
                'p95_ms': pd.to_numeric(one['95%'], errors='coerce').median(),
                'error_pct': 100 * (failures.max() - failures.min()) / sent if sent else 0.0,
                'cpu_pct': cpu['cpu_pct'].mean(),
                'mem_mb': cpu['mem_mb'].max(),
            }
        )
    return pd.DataFrame(rows).round(2)


def style(ax, title: str, ylabel: str, xlabel: str) -> None:
    ax.set_title(title, loc='left', fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', color='#e4e4e0', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(bottom=0)


@app.command()
def main(
    runs: Annotated[
        list[Path], typer.Option(help='Carpetas de cada corrida; el nombre es el motor')
    ],
    output: Path = Path('reports/carga/figuras'),
):
    output.mkdir(parents=True, exist_ok=True)
    data = {folder.name: load_run(folder) for folder in runs}
    summaries = {name: summarize(*pair) for name, pair in data.items()}
    table = pd.concat([s.assign(motor=name) for name, s in summaries.items()])
    table.to_csv(output / 'resumen_escalones.csv', index=False)
    print(table.to_string(index=False))

    # 1. Desempeño por número de usuarios
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [
        ('rps_prediccion', 'Peticiones de predicción por segundo', 'req/s'),
        ('p95_ms', 'Latencia p95 de /p1/predict (1 fila)', 'ms'),
        ('error_pct', 'Tasa de error', '%'),
    ]
    for ax, (column, title, unit) in zip(axes, panels):
        for name, summary in summaries.items():
            ax.plot(
                summary['usuarios'],
                summary[column],
                marker='o',
                markersize=5,
                linewidth=2,
                color=COLORS.get(name),
                label=name,
                linestyle='--' if name in DASHED else '-',
            )
        style(ax, title, unit, 'usuarios concurrentes')
    axes[0].legend(frameon=False)
    if table['error_pct'].max() == 0:
        axes[2].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output / 'desempeno_por_usuarios.png', dpi=160)

    # 2. Serie de tiempo de la corrida
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for name, (history, _) in data.items():
        total = history[history['Name'] == 'Aggregated']
        one = history[history['Name'] == ONE]
        axes[0].plot(
            total['t'],
            total['Requests/s'],
            linewidth=1.5,
            color=COLORS.get(name),
            label=name,
            linestyle='--' if name in DASHED else '-',
        )
        axes[1].plot(
            one['t'],
            pd.to_numeric(one['95%'], errors='coerce'),
            linewidth=1.5,
            color=COLORS.get(name),
            label=name,
            linestyle='--' if name in DASHED else '-',
        )
    style(axes[0], 'Peticiones por segundo (todas las rutas)', 'req/s', 'segundos')
    style(axes[1], 'Latencia p95 de /p1/predict (1 fila)', 'ms', 'segundos')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / 'serie_tiempo.png', dpi=160)

    # 3. CPU y memoria del contenedor de la API
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for name, (_, stats) in data.items():
        axes[0].plot(
            stats['t'],
            stats['cpu_pct'],
            linewidth=1.5,
            color=COLORS.get(name),
            label=name,
            linestyle='--' if name in DASHED else '-',
        )
        axes[1].plot(
            stats['t'],
            stats['mem_mb'],
            linewidth=1.5,
            color=COLORS.get(name),
            label=name,
            linestyle='--' if name in DASHED else '-',
        )
    style(axes[0], 'CPU de mibici-api (límite 1 CPU = 100 %)', '%', 'segundos')
    style(axes[1], 'Memoria de mibici-api (límite 1 GB)', 'MB', 'segundos')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / 'recursos_api.png', dpi=160)
    print(f'figuras -> {output}')


if __name__ == '__main__':
    app()
