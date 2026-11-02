
"""Ejecuta validaciones de calidad sobre datos reales de MiBici."""

from pathlib import Path

import pandas as pd

from mi_bici.config import (
    P1_FEATURES_PATH,
    P2_FEATURES_PATH,
    P3_TRAIN_FEATURES_PATH,
    P3_HOLDOUT_FEATURES_PATH,
)
from mi_bici.validation.deepchecks_validation import validate_data_quality
from mi_bici.validation.contracts import validate_feature_contract


REPORTS_DIR = Path("reports") / "data_quality"


def main():
    datasets = {
        "P1": P1_FEATURES_PATH,
        "P2": P2_FEATURES_PATH,
        "P3_train": P3_TRAIN_FEATURES_PATH,
        "P3_holdout": P3_HOLDOUT_FEATURES_PATH,
    }

    missing_files = [
        f"{name}: {path}"
        for name, path in datasets.items()
        if not path.is_file()
    ]
    if missing_files:
        raise SystemExit(
            "Faltan archivos de datos requeridos:\n"
            + "\n".join(missing_files)
            + "\nEjecuta la ingesta y la generación de características antes de validar."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    problemas = []
    error_rows = []

    for name, path in datasets.items():
        print(f"\n===== VALIDANDO {name} =====")

        df = pd.read_parquet(path)
        dataset_errors = validate_feature_contract(name, df)

        # DeepChecks puede ser costoso con datasets grandes.
        # Usamos una muestra reproducible para los chequeos.
        sample = df.sample(
            n=min(len(df), 5000),
            random_state=42,
        ) if not df.empty else df

        if not dataset_errors:
            results = validate_data_quality(sample)
            dataset_errors.extend(f'{name}: {error}' for error in results['errors'])
        problemas.extend(dataset_errors)
        error_rows.extend({'dataset': name, 'error': error} for error in dataset_errors)

        # Resumen completo calculado sobre el dataset original.
        missing = df.isna().sum()

        print(f"Registros totales: {len(df):,}")
        print(f"Columnas: {len(df.columns)}")
        print(f"Filas duplicadas: {df.duplicated().sum():,}")
        print(f"Valores faltantes totales: {missing.sum():,}")

        for column in df.columns:
            summary_rows.append({
                "dataset": name,
                "column": column,
                "dtype": str(df[column].dtype),
                "total_rows": len(df),
                "missing_count": int(missing[column]),
                "missing_percent": round(
                    100 * missing[column] / len(df), 2
                ) if len(df) else 0,
                "duplicate_rows": int(df.duplicated().sum()),
                "status": "error" if dataset_errors else "pass",
            })

    report = pd.DataFrame(summary_rows)
    output = REPORTS_DIR / "quality_summary.csv"
    report.to_csv(output, index=False)
    pd.DataFrame(error_rows, columns=['dataset', 'error']).to_csv(
        REPORTS_DIR / 'quality_errors.csv', index=False
    )

    print(f"\nReporte guardado en: {output}")

    if problemas:
        raise SystemExit("Problemas de calidad:\n" + "\n".join(problemas))


if __name__ == "__main__":
    main()
