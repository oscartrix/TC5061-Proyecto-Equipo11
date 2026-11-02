
"""Análisis de data drift para P1, P2 y P3 de MiBici."""

import numpy as np
import pandas as pd

# Compatibilidad temporal: DeepChecks 0.19.1 + NumPy 2.x
if not hasattr(np, "Inf"):
    np.Inf = np.inf

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureDrift
from deepchecks.core import CheckFailure

from mi_bici.config import (
    DRIFT_THRESHOLDS,
    P1_FEATURES_PATH,
    P2_FEATURES_PATH,
    P3_TRAIN_FEATURES_PATH,
    P3_HOLDOUT_FEATURES_PATH,
    REPORTS_DIR,
    RANDOM_SEED,
)

SAMPLE_SIZE = 5000
OUTPUT_DIR = REPORTS_DIR / "data_drift"


def apply_drift_policy(report, thresholds=None):
    """Clasifica cada característica con umbrales inclusivos por método."""
    thresholds = DRIFT_THRESHOLDS if thresholds is None else thresholds
    if report.empty:
        raise ValueError('El reporte de drift no puede estar vacío.')
    for method in report['method'].unique():
        if method not in thresholds:
            raise ValueError(f'No hay umbrales de drift para el método {method}.')
        warning = thresholds[method]['warning']
        error = thresholds[method]['error']
        if not 0 <= warning < error <= 1:
            raise ValueError(
                f'Umbrales inválidos para {method}: se requiere 0 <= warning < error <= 1.'
            )

    result = report.copy()
    result['warning_threshold'] = result['method'].map(lambda m: thresholds[m]['warning'])
    result['error_threshold'] = result['method'].map(lambda m: thresholds[m]['error'])
    result['status'] = 'pass'
    scores = result['drift_score']
    result.loc[scores >= result['warning_threshold'], 'status'] = 'warning'
    result.loc[scores >= result['error_threshold'], 'status'] = 'error'
    # Un cálculo inválido nunca debe interpretarse como ausencia de drift.
    invalid = ~np.isfinite(scores) | ~scores.between(0, 1)
    result.loc[invalid, 'status'] = 'error'
    return result


def enforce_drift_policy(report):
    """Emite advertencias y detiene el comando cuando hay drift por encima del límite."""
    for row in report.loc[report['status'] == 'warning'].itertuples():
        print(f'ADVERTENCIA: {row.dataset}.{row.feature}: drift={row.drift_score:.4f} '
              f'(umbral={row.warning_threshold:.2f})')
    errors = report.loc[report['status'] == 'error']
    if not errors.empty:
        details = [
            f'{row.dataset}.{row.feature}: drift={row.drift_score} '
            f'(límite={row.error_threshold:.2f}; el score debe ser finito y estar entre 0 y 1)'
            for row in errors.itertuples()
        ]
        raise SystemExit('Drift no aceptable:\n' + '\n'.join(details))


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
    if isinstance(result, CheckFailure):
        raise ValueError(f'{name}: no se pudo calcular el drift: {result.exception}')

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

    report = apply_drift_policy(pd.DataFrame(rows))
    report = report.sort_values(
        "drift_score",
        ascending=False,
    )

    print(report[["feature", "drift_score", "method", "status"]].to_string(
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
    enforce_drift_policy(final_report)


if __name__ == "__main__":
    main()
