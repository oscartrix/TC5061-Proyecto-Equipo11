
"""Pruebas de preparación de datos para monitoreo P2."""

from unittest.mock import patch

import pandas as pd

from mi_bici.monitoring.prepare_data import prepare_monitoring_data
from mi_bici.modeling.train import P2_FEATURES


def test_monitoring_data():
    # Crear datos simulados con las variables reales de P2
    sample = {feature: [1.0, 2.0] for feature in P2_FEATURES}
    sample["split"] = ["train", "test"]
    df = pd.DataFrame(sample)

    # Simular la lectura del archivo Parquet
    with patch(
        "mi_bici.monitoring.prepare_data.pd.read_parquet",
        return_value=df,
    ):
        reference, current = prepare_monitoring_data()

    assert reference.shape == (1, len(P2_FEATURES))
    assert current.shape == (1, len(P2_FEATURES))

    assert list(reference.columns) == list(P2_FEATURES)
    assert list(current.columns) == list(P2_FEATURES)

    assert reference.dtypes.equals(current.dtypes)

    assert not reference.isna().any().any()
    assert not current.isna().any().any()
