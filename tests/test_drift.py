
"""Pruebas automatizadas para el análisis de data drift."""

import numpy as np
import pandas as pd

from mi_bici.validation.drift_validation import (
    sample_data,
    validate_drift,
)


def test_sample_data_reproducible():
    """El muestreo debe producir los mismos registros."""

    df = pd.DataFrame({"value": range(6000)})

    sample1 = sample_data(df)
    sample2 = sample_data(df)

    assert len(sample1) == 5000
    pd.testing.assert_frame_equal(sample1, sample2)


def test_validate_drift():
    """DeepChecks debe devolver resultados válidos."""

    rng = np.random.default_rng(42)

    train = pd.DataFrame({
        "trips": rng.normal(100, 15, 100),
        "capacity": rng.integers(10, 50, 100),
    })

    test = pd.DataFrame({
        "trips": rng.normal(110, 15, 100),
        "capacity": rng.integers(10, 50, 100),
    })

    result = validate_drift(
        "TEST",
        train,
        test,
        categorical_features=[],
    )

    assert not result.empty
    assert set(result["feature"]) == {"trips", "capacity"}
    assert result["drift_score"].notna().all()
    assert np.isfinite(result["drift_score"]).all()
