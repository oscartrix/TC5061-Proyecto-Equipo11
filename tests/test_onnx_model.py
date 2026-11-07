"""Pruebas de la inferencia con ONNX Runtime."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pytest.importorskip('onnxruntime')
skl2onnx = pytest.importorskip('skl2onnx')

from skl2onnx.common.data_types import FloatTensorType, StringTensorType  # noqa: E402

from mi_bici.api.onnx_model import load_onnx_model  # noqa: E402

NUM = ['a', 'b']
CAT = ['zone']


@pytest.fixture
def model_dir(tmp_path):
    rng = np.random.default_rng(1)
    x = pd.DataFrame({
        'a': rng.uniform(0, 10, 60),
        'b': rng.uniform(0, 5, 60),
        'zone': rng.choice(['GDL', 'ZPN', 'TLQ'], 60),
    })
    prep = ColumnTransformer([
        ('num', StandardScaler(), NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT),
    ])
    pipe = Pipeline([('prep', prep), ('reg', Ridge())]).fit(x, x['a'] * 2 + x['b'])
    types = [(c, FloatTensorType([None, 1])) for c in NUM]
    types += [(c, StringTensorType([None, 1])) for c in CAT]
    (tmp_path / 'model.onnx').write_bytes(
        skl2onnx.convert_sklearn(pipe, initial_types=types).SerializeToString()
    )
    (tmp_path / 'registry.txt').write_text('modelo-prueba\n2\nrun-1\n')
    return tmp_path, pipe, x


def test_onnx_matches_sklearn(model_dir):
    path, pipe, x = model_dir
    served = load_onnx_model(path)
    assert np.allclose(served.predict(x), pipe.predict(x), atol=1e-4)


def test_onnx_keeps_registry_data(model_dir):
    served = load_onnx_model(model_dir[0])
    assert (served.name, served.version, served.run_id) == ('modelo-prueba', '2', 'run-1')


def test_onnx_one_prediction_per_row(model_dir):
    path, _, x = model_dir
    assert load_onnx_model(path).predict(x.head(3)).shape == (3,)
