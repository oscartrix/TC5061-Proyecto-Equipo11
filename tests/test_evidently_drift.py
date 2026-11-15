"""Pruebas unitarias para la detección de Data Drift con Evidently."""

import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from mi_bici.modeling.train import P2_FEATURES
from mi_bici.monitoring.evidently_drift import (
    DRIFT_CAT_METHOD,
    DRIFT_NUM_METHOD,
    DRIFT_THRESHOLD,
    additional_comparisons,
    get_two_most_recent_months_data,
    generate_drift_report,
    load_monitoring_frames,
    monitored_columns,
    save_drift_summary,
    summarize_comparison,
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


def monitoring_frame(start, periods, shift=0.0, seed=0):
    """DataFrame con identificadores, variables P2, objetivo y una variable desplazable."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({feat: rng.normal(0, 1, periods) for feat in P2_FEATURES})
    df["departures"] = rng.normal(shift, 1, periods)
    df["hour"] = rng.integers(0, 24, periods)
    df["day_of_week"] = rng.integers(0, 7, periods)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["station_id"] = 1
    df["hour_ts"] = pd.date_range(start, periods=periods, freq="h")
    df["risk_class"] = rng.choice(["deficit", "balanceado", "excedente"], periods)
    return df


def test_load_monitoring_frames_agrega_predicciones(tmp_path):
    """Une las predicciones del champion por estación y hora."""
    ref = monitoring_frame("2025-01-01", 10)
    curr = monitoring_frame("2026-07-01", 10)
    preds = pd.concat([ref, curr])[["station_id", "hour_ts"]].assign(prediction="deficit")
    preds_path = tmp_path / "p2_predictions.parquet"
    preds.to_parquet(preds_path)

    with patch(
        "mi_bici.monitoring.evidently_drift.prepare_monitoring_data", return_value=(ref, curr)
    ):
        ref_out, curr_out = load_monitoring_frames(predictions_path=preds_path)

    assert (ref_out["prediction"] == "deficit").all()
    assert (curr_out["prediction"] == "deficit").all()
    assert len(ref_out) == 10


def test_load_monitoring_frames_sin_predicciones(tmp_path):
    """Si no hay archivo de predicciones, el reporte sigue sin esa columna."""
    ref = monitoring_frame("2025-01-01", 10)
    curr = monitoring_frame("2026-07-01", 10)
    with patch(
        "mi_bici.monitoring.evidently_drift.prepare_monitoring_data", return_value=(ref, curr)
    ):
        ref_out, _ = load_monitoring_frames(predictions_path=tmp_path / "no_existe.parquet")

    assert "prediction" not in ref_out.columns
    assert "prediction" not in monitored_columns(ref_out)
    assert "risk_class" in monitored_columns(ref_out)


def test_generate_drift_report_detalles_y_calidad():
    """Registra prueba, umbral y decisión por variable, y la calidad de datos."""
    ref = monitoring_frame("2025-01-01", 1500, seed=1)
    curr = monitoring_frame("2026-07-01", 1500, shift=3.0, seed=2)
    columns = monitored_columns(ref)

    result = generate_drift_report(ref[columns], curr[columns], output_html_path=None)
    details = result["metrics_summary"]["drift_details"]

    assert result["html_path"] is None
    assert details["departures"]["method"] == DRIFT_NUM_METHOD
    assert details["risk_class"]["method"] == DRIFT_CAT_METHOD
    assert details["departures"]["threshold"] == DRIFT_THRESHOLD
    assert details["departures"]["drift"] is True
    assert details["arrivals"]["drift"] is False
    assert result["metrics_summary"]["data_quality_current"]["Missing values count"] == 0


def test_summarize_comparison_excluye_variables_fijas():
    """Las variables fijas no cuentan en la proporción de drift."""
    details = {
        "departures": {"drift": True},
        "arrivals": {"drift": False},
        "capacity": {"drift": True},
        "hour": {"drift": True},
    }
    ref = monitoring_frame("2025-01-01", 5)
    curr = monitoring_frame("2026-07-01", 5)

    summary = summarize_comparison(ref, curr, {"drift_details": details})

    assert summary["drifted_columns"] == ["departures"]
    assert summary["drifted_share_monitored"] == 0.5
    assert summary["excluded_static_columns"] == ["hour", "capacity"]
    assert summary["current"]["start"] == "2026-07-01 00:00:00"


def test_additional_comparisons_por_mes_y_mismos_meses():
    """Compara cada mes actual y los mismos meses de la referencia."""
    ref = pd.concat(
        [monitoring_frame("2025-01-01", 48, seed=1), monitoring_frame("2025-07-01", 48, seed=2)],
        ignore_index=True,
    )
    curr = pd.concat(
        [monitoring_frame("2026-07-01", 48, seed=3), monitoring_frame("2026-08-01", 48, seed=4)],
        ignore_index=True,
    )

    comparisons = additional_comparisons(ref, curr)

    assert set(comparisons) == {
        "train_vs_2026-07",
        "train_vs_2026-08",
        "mismos_meses_train_vs_actual",
    }
    same_months = comparisons["mismos_meses_train_vs_actual"]["reference"]
    assert same_months["start"].startswith("2025-07")
    assert same_months["rows"] == 48


def test_save_drift_summary(tmp_path):
    """Guarda el resumen en JSON legible."""
    path = save_drift_summary({"comparisons": {"a": {"drifted_columns": ["x"]}}}, tmp_path / "s.json")

    assert json.loads(path.read_text(encoding="utf-8"))["comparisons"]["a"]["drifted_columns"] == ["x"]
