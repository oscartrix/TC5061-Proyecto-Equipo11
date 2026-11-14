"""Pruebas unitarias para la detección de Data Drift con Evidently."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from mi_bici.modeling.train import P2_FEATURES
from mi_bici.monitoring.evidently_drift import (
    get_two_most_recent_months_data,
    generate_drift_report,
)


@pytest.fixture
def dummy_p2_dataframe():
    """Genera un DataFrame simulado con timestamps de dos meses consecutivos y columnas P2."""
    n_rows_per_month = 20
    data_aug = {feat: [float(i) for i in range(n_rows_per_month)] for feat in P2_FEATURES}
    data_aug["hour_ts"] = pd.date_range("2026-08-01", periods=n_rows_per_month, freq="h")

    data_sep = {feat: [float(i + 5) for i in range(n_rows_per_month)] for feat in P2_FEATURES}
    data_sep["hour_ts"] = pd.date_range("2026-09-01", periods=n_rows_per_month, freq="h")

    df_aug = pd.DataFrame(data_aug)
    df_sep = pd.DataFrame(data_sep)
    return pd.concat([df_aug, df_sep], ignore_index=True)


def test_get_two_most_recent_months_data(dummy_p2_dataframe):
    """Verifica la extracción y separación de los dos meses más recientes."""
    with patch("mi_bici.monitoring.evidently_drift.pd.read_parquet", return_value=dummy_p2_dataframe):
        with patch.object(Path, "exists", return_value=True):
            ref, curr, ref_m, curr_m = get_two_most_recent_months_data(sample_size=10)

    assert ref_m == "2026-08"
    assert curr_m == "2026-09"
    assert len(ref) == 10
    assert len(curr) == 10
    assert list(ref.columns) == list(P2_FEATURES)
    assert list(curr.columns) == list(P2_FEATURES)
    assert not ref.isna().any().any()
    assert not curr.isna().any().any()


def test_generate_drift_report_execution(tmp_path):
    """Verifica que el reporte de drift con Evidently se ejecute y guarde el HTML."""
    ref_df = pd.DataFrame({
        "hour": [1, 2, 3, 4, 5],
        "departures": [10.0, 20.0, 15.0, 30.0, 25.0],
        "arrivals": [12.0, 18.0, 14.0, 28.0, 26.0],
    })
    curr_df = pd.DataFrame({
        "hour": [1, 2, 3, 4, 5],
        "departures": [11.0, 21.0, 16.0, 29.0, 24.0],
        "arrivals": [13.0, 19.0, 15.0, 27.0, 25.0],
    })

    out_file = tmp_path / "test_report.html"
    result = generate_drift_report(
        reference_data=ref_df,
        current_data=curr_df,
        output_html_path=out_file,
    )

    assert out_file.exists()
    assert out_file.stat().st_size > 0
    assert "metrics_summary" in result
    assert result["html_path"] == str(out_file)
