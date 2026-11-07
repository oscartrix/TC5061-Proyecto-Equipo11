"""Compara la latencia de sklearn y ONNX Runtime con el champion de P1.

Uso:
    python scripts/benchmark_inference.py --model-dir models/p1_champion
"""

from pathlib import Path
import time

import mlflow.sklearn
import numpy as np
import pandas as pd
import typer

from mi_bici.api.onnx_model import load_onnx_model
from mi_bici.api.schemas import EXAMPLE_FEATURES
from mi_bici.modeling.train import P1_CAT_COLS, P1_NUM_COLS

app = typer.Typer()


def sample(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = pd.DataFrame([EXAMPLE_FEATURES] * rows)
    for column in P1_NUM_COLS[:5]:
        data[column] = rng.uniform(0, 60, rows)
    return data


def measure(predict, data: pd.DataFrame, repeats: int) -> tuple[float, float]:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict(data)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


@app.command()
def main(model_dir: Path = Path('models/p1_champion'), repeats: int = 200):
    pipeline = mlflow.sklearn.load_model(str(model_dir))
    onnx_model = load_onnx_model(model_dir)
    columns = P1_NUM_COLS + P1_CAT_COLS
    print('filas | motor   | p50 ms | p95 ms | filas/s')
    for rows in (1, 100, 500):
        data = sample(rows)
        for name, predict in (
            ('sklearn', lambda d: pipeline.predict(d[columns])),
            ('onnx', onnx_model.predict),
        ):
            predict(data)  # calentamiento
            p50, p95 = measure(predict, data, repeats)
            print(f'{rows:5d} | {name:7s} | {p50:6.2f} | {p95:6.2f} | {rows / (p50 / 1000):.0f}')


if __name__ == '__main__':
    app()
