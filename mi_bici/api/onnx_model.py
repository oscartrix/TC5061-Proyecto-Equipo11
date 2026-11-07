"""Inferencia del modelo de P1 con ONNX Runtime."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from mi_bici.config import REGISTERED_MODELS

STRING_COLUMNS = {'zone'}


class OnnxServedModel:
    """Misma interfaz que ServedModel: name, version, run_id y predict."""

    def __init__(self, name: str, version: str, run_id: str | None, path: Path):
        self.name = name
        self.version = version
        self.run_id = run_id
        self.session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        self.columns = [item.name for item in self.session.get_inputs()]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        # Un tensor de una columna por variable; los números van como float32
        inputs = {}
        for column in self.columns:
            dtype = object if column in STRING_COLUMNS else np.float32
            inputs[column] = features[column].to_numpy().astype(dtype).reshape(-1, 1)
        return self.session.run(None, inputs)[0].ravel().astype(np.float32)


def load_onnx_model(model_dir: Path) -> OnnxServedModel:
    """Carga model.onnx y los datos del registry guardados junto al modelo."""
    info = (model_dir / 'registry.txt').read_text().split()
    name, version, run_id = info if len(info) == 3 else (REGISTERED_MODELS['p1'][0], '0', '')
    return OnnxServedModel(name, version, run_id or None, model_dir / 'model.onnx')
