"""Inferencia del modelo de P1 con ONNX Runtime."""

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from mi_bici.config import REGISTERED_MODELS

STRING_COLUMNS = {'zone'}


class OnnxServedModel:
    """Misma interfaz que ServedModel: name, version, run_id y predict."""

    backend = 'onnx'

    def __init__(self, name: str, version: str, run_id: str | None, path: Path):
        self.name = name
        self.version = version
        self.run_id = run_id
        # En el contenedor ONNX Runtime ve todos los núcleos del host aunque tenga 1 CPU;
        # con un hilo por núcleo en espera activa se gasta la cuota de CPU sin predecir
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv('ORT_THREADS', '1'))
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
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
