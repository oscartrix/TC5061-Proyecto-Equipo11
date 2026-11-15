
"""Pruebas de preparación de datos para monitoreo P2."""

from unittest.mock import patch

import pandas as pd

from mi_bici.monitoring.prepare_data import MONITORING_COLUMNS, prepare_monitoring_data
from mi_bici.modeling.train import P2_FEATURES


def test_monitoring_data():
    # Crear datos simulados con las variables reales de P2
    sample = {feature: [1.0, 2.0] for feature in P2_FEATURES}
    sample["station_id"] = [10, 10]
    sample["hour_ts"] = pd.to_datetime(["2025-03-01 08:00", "2026-07-01 08:00"])
    sample["risk_class"] = ["deficit", "balanceado"]
    sample["split"] = ["train", "test"]
    df = pd.DataFrame(sample)

    # Simular la lectura del archivo Parquet
    with patch(
        "mi_bici.monitoring.prepare_data.pd.read_parquet",
        return_value=df,
    ):
        reference, current = prepare_monitoring_data()

    assert reference.shape == (1, len(MONITORING_COLUMNS))
    assert current.shape == (1, len(MONITORING_COLUMNS))

    assert list(reference.columns) == MONITORING_COLUMNS
    assert list(current.columns) == MONITORING_COLUMNS
    assert "split" not in reference.columns

    assert reference.dtypes.equals(current.dtypes)

    assert not reference.isna().any().any()
    assert not current.isna().any().any()
