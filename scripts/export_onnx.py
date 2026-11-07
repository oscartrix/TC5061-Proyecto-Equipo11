"""Convierte el champion de P1 a ONNX y revisa que dé las mismas predicciones.

Uso:
    python scripts/export_onnx.py --model-dir models/p1_champion
"""

from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
import typer

from mi_bici.api.schemas import EXAMPLE_FEATURES
from mi_bici.modeling.train import P1_CAT_COLS, P1_NUM_COLS

app = typer.Typer()


def patch_bool_attributes():
    """skl2onnx 1.20 pasa booleanos en los árboles de HGB y onnx 1.23 los rechaza."""
    import skl2onnx.common._container as container

    original = container.make_node

    def make_node(op_type, inputs, outputs, **attrs):
        for key, value in attrs.items():
            if isinstance(value, (list, tuple)):
                attrs[key] = [int(v) if isinstance(v, (bool, np.bool_)) else v for v in value]
        return original(op_type, inputs, outputs, **attrs)

    container.make_node = make_node


def to_onnx(pipeline) -> bytes:
    patch_bool_attributes()
    types = [(c, FloatTensorType([None, 1])) for c in P1_NUM_COLS]
    types += [(c, StringTensorType([None, 1])) for c in P1_CAT_COLS]
    return convert_sklearn(pipeline, initial_types=types, target_opset=17).SerializeToString()


@app.command()
def main(model_dir: Path = Path('models/p1_champion'), max_mean_diff: float = 0.25):
    pipeline = mlflow.sklearn.load_model(str(model_dir))
    (model_dir / 'model.onnx').write_bytes(to_onnx(pipeline))

    # Paridad con datos reales de prueba; sin ellos, con filas sintéticas
    from mi_bici.api.onnx_model import load_onnx_model

    data_path = Path('data/processed/p1_features.parquet')
    if data_path.is_file():
        rows = pd.read_parquet(data_path).query("split == 'test'").sample(20000, random_state=0)
    else:
        rng = np.random.default_rng(0)
        rows = pd.DataFrame([EXAMPLE_FEATURES] * 200)
        for column in P1_NUM_COLS[:5]:
            rows[column] = rng.uniform(0, 60, len(rows))
    expected = pipeline.predict(rows[P1_NUM_COLS + P1_CAT_COLS])
    diff = np.abs(expected - load_onnx_model(model_dir).predict(rows))
    # ONNX usa float32: en los árboles algunos valores cambian de hoja
    print(f'diferencia media: {diff.mean():.4f}  p99: {np.percentile(diff, 99):.4f}  máxima: {diff.max():.4f}')
    if diff.mean() > max_mean_diff:
        raise typer.Exit(code=1)


if __name__ == '__main__':
    app()
