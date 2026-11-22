"""Costo por inferencia a partir de la prueba de carga y de precios de referencia de AWS.

Uso:
    python scripts/finops.py --summary reports/carga/figuras/resumen_escalones.csv

Precios on-demand de us-east-1 (Linux) consultados en la AWS Price List API,
vigentes desde 2026-11-01. Minutos de GitHub Actions según la tabla de precios
de GitHub de 2026 (los repositorios públicos no pagan los runners estándar).
"""

from pathlib import Path

import pandas as pd
import typer

app = typer.Typer()

# USD por hora
FARGATE_VCPU_H = 0.04048
FARGATE_GB_H = 0.004445
EC2 = {
    't3.small (2 vCPU, 2 GB)': 0.0208,
    't3.medium (2 vCPU, 4 GB)': 0.0416,
    't4g.small (2 vCPU, 2 GB, ARM)': 0.0168,
}
# USD por minuto
ACTIONS_LINUX_MIN = 0.006
ACTIONS_MACOS_MIN = 0.062
HOURS_MONTH = 730

# Predicciones (filas) por día en cada escenario
SCENARIOS = {
    'Lote diario (470 estaciones)': 470,
    'Replaneación cada hora': 470 * 24,
    'App pública (1 M filas/día)': 1_000_000,
    'Alto volumen (50 M filas/día)': 50_000_000,
}


def fargate_hour(vcpu: float, memory_gb: float) -> float:
    return vcpu * FARGATE_VCPU_H + memory_gb * FARGATE_GB_H


def capacity(summary: pd.DataFrame, slo_p95_ms: float) -> pd.DataFrame:
    """Escalón más alto de cada motor que cumple el p95 y no tiene errores."""
    ok = summary[(summary['p95_ms'] <= slo_p95_ms) & (summary['error_pct'] == 0)]
    return ok.sort_values('usuarios').groupby('motor').tail(1).set_index('motor')


@app.command()
def main(
    summary: Path = Path('reports/carga/figuras/resumen_escalones.csv'),
    slo_p95_ms: float = 300,
    api_vcpu: float = 1.0,
    api_memory_gb: float = 1.0,
    output: Path = Path('reports/carga/figuras/finops.csv'),
):
    table = pd.read_csv(summary)
    best = capacity(table, slo_p95_ms)
    hour = fargate_hour(api_vcpu, api_memory_gb)
    print(
        f'Réplica de la API en Fargate ({api_vcpu} vCPU, {api_memory_gb} GB): {hour:.5f} USD/h\n'
    )

    rows = []
    for motor, row in best.iterrows():
        rows_per_hour = row['filas_s'] * 3600
        rows.append(
            {
                'motor': motor,
                'usuarios_slo': int(row['usuarios']),
                'peticiones_s': row['rps_prediccion'],
                'filas_s': row['filas_s'],
                'p95_ms': row['p95_ms'],
                'cpu_pct': row['cpu_pct'],
                'mem_mb': row['mem_mb'],
                # Réplica ocupada al 100 %: el piso del costo por inferencia
                'usd_por_millon_filas': 1e6 * hour / rows_per_hour,
            }
        )
    result = pd.DataFrame(rows).round(4)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f'Capacidad con p95 <= {slo_p95_ms:.0f} ms y 0 % de errores')
    print(result.to_string(index=False), '\n')

    # Con poco tráfico la réplica está casi siempre ociosa: el costo lo fija el tiempo encendida
    print(
        'Costo mensual por escenario (Fargate, réplicas siempre encendidas, pico de 5x el promedio)'
    )
    scenario_rows = []
    for name, per_day in SCENARIOS.items():
        peak = 5 * per_day / 86400
        for motor in ('onnx', 'sklearn'):
            if motor not in best.index:
                continue
            for vcpu, memory in ((api_vcpu, api_memory_gb), (0.25, 0.5)):
                # Con menos CPU la capacidad baja en proporción
                rows_s = best.loc[motor, 'filas_s'] * vcpu / api_vcpu
                replicas = max(1, -(-peak // rows_s))
                monthly = replicas * fargate_hour(vcpu, memory) * HOURS_MONTH
                scenario_rows.append(
                    {
                        'escenario': name,
                        'motor': motor,
                        'replica': f'{vcpu} vCPU / {memory} GB',
                        'replicas': int(replicas),
                        'usd_mes': round(monthly, 2),
                        'usd_por_millon_filas': round(monthly / (per_day * 30) * 1e6, 2),
                    }
                )
    scenarios = pd.DataFrame(scenario_rows)
    scenarios.to_csv(output.with_name('finops_escenarios.csv'), index=False)
    print(scenarios.to_string(index=False), '\n')

    print('Stack completo de docker compose en una sola instancia EC2 (USD/mes)')
    for name, price in EC2.items():
        print(f'  {name:30s} {price * HOURS_MONTH:6.2f}')
    print()

    # Duraciones medidas en las corridas de main (minutos por corrida)
    # ci.yml: pruebas en Linux e imágenes; macOS aparte. pipeline.yml: calidad y pipeline
    ci_linux = 3.4 + 4.5
    ci_macos = 4.0
    pipeline = 3.0 + 16.0
    private = (ci_linux + pipeline) * ACTIONS_LINUX_MIN + ci_macos * ACTIONS_MACOS_MIN
    macos_share = ci_macos * ACTIONS_MACOS_MIN / private
    print(
        f'CI/CD si el repositorio fuera privado: {private:.3f} USD por push a main '
        f'(macOS es {100 * macos_share:.0f} % del costo con {ci_macos:.0f} de '
        f'{ci_linux + ci_macos + pipeline:.0f} minutos)'
    )


if __name__ == '__main__':
    app()
