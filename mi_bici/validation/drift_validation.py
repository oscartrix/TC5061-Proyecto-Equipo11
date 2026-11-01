
"""Análisis de data drift para P1, P2 y P3 de MiBici."""

import numpy as np
import pandas as pd

# Compatibilidad temporal: DeepChecks 0.19.1 + NumPy 2.x
if not hasattr(np, "Inf"):
    np.Inf = np.inf

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureDrift

from mi_bici.config import (
    P1_FEATURES_PATH,
    P2_FEATURES_PATH,
    P3_TRAIN_FEATURES_PATH,
    P3_HOLDOUT_FEATURES_PATH,
    REPORTS_DIR,
    RANDOM_SEED,
)

SAMPLE_SIZE = 5000
OUTPUT_DIR = REPORTS_DIR / "data_drift"


def sample_data(df):
    """Obtiene una muestra reproducible sin alterar los datos."""
    if len(df) <= SAMPLE_SIZE:
        return df.copy()

    return df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_SEED,
    )


def validate_drift(name, train, test, categorical_features):
    """Calcula el drift entre dos conjuntos de características."""

    print(f"\n===== DATA DRIFT {name} =====")

    if train.empty or test.empty:
        raise ValueError(f"{name}: train o test está vacío.")

    if list(train.columns) != list(test.columns):
        raise ValueError(f"{name}: las columnas no coinciden.")

    train = sample_data(train)
    test = sample_data(test)

    train_dataset = Dataset(
        train,
        cat_features=categorical_features,
    )
    test_dataset = Dataset(
        test,
        cat_features=categorical_features,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = FeatureDrift().run(train_dataset, test_dataset)
    html_path = OUTPUT_DIR / f"{name.lower()}_drift_report.html"

    result.save_as_html(str(html_path))

    print(f"Reporte HTML guardado en: {html_path}")

    rows = []

    for feature, metrics in result.value.items():
        rows.append({
            "dataset": name,
            "feature": feature,
            "drift_score": float(metrics["Drift score"]),
            "method": metrics["Method"],
            "train_samples": len(train),
            "test_samples": len(test),
        })

    report = pd.DataFrame(rows)
    report = report.sort_values(
        "drift_score",
        ascending=False,
    )

    print(report[["feature", "drift_score", "method"]].to_string(
        index=False
    ))

    return report


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reports = []

    # P1: Regresión
    p1 = pd.read_parquet(P1_FEATURES_PATH)

    p1_excluded = {
        "station_id", "date", "trips_count", "split"
    }
    p1_features = [
        col for col in p1.columns
        if col not in p1_excluded
    ]

    reports.append(validate_drift(
        "P1",
        p1.loc[p1["split"] == "train", p1_features],
        p1.loc[p1["split"] == "test", p1_features],
        categorical_features=["zone"],
    ))

    # P2: Clasificación
    p2 = pd.read_parquet(P2_FEATURES_PATH)

    p2_excluded = {
        "station_id",
        "hour_ts",
        "risk_class",
        "future_3h_net_flow",
        "split",
    }
    p2_features = [
        col for col in p2.columns
        if col not in p2_excluded
    ]

    reports.append(validate_drift(
        "P2",
        p2.loc[p2["split"] == "train", p2_features],
        p2.loc[p2["split"] == "test", p2_features],
        categorical_features=[
            col for col in ["zone", "risk_lag_168h"]
            if col in p2_features
        ],
    ))

    # P3: Clustering
    p3_train = pd.read_parquet(P3_TRAIN_FEATURES_PATH)
    p3_holdout = pd.read_parquet(P3_HOLDOUT_FEATURES_PATH)

    p3_excluded = {
        "station_id", "name", "latitude", "longitude"
    }
    p3_features = [
        col for col in p3_train.columns
        if col not in p3_excluded
    ]

    reports.append(validate_drift(
        "P3",
        p3_train[p3_features],
        p3_holdout[p3_features],
        categorical_features=[
            col for col in ["zone", "is_ex", "is_active"]
            if col in p3_features
        ],
    ))

    # Consolidar resultados
    final_report = pd.concat(reports, ignore_index=True)

    output = OUTPUT_DIR / "drift_summary.csv"
    final_report.to_csv(output, index=False)

    print(f"\nReporte guardado en: {output}")


if __name__ == "__main__":
    main()
