
"""Validaciones de calidad de datos para MiBici."""

import pandas as pd
import numpy as np
# Compatibilidad temporal entre DeepChecks 0.19.1 y NumPy 2.x
if not hasattr(np, "Inf"):
    np.Inf = np.inf

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    MixedDataTypes,
    MixedNulls,
    DataDuplicates,
)


def validate_data_quality(df: pd.DataFrame):
    """Ejecuta chequeos básicos de calidad sobre un DataFrame."""

    if df.empty:
        raise ValueError("El dataset no puede estar vacío.")

    dataset = Dataset(df, cat_features=[])

    checks = {
        "mixed_data_types": MixedDataTypes(),
        "mixed_nulls": MixedNulls(),
        "duplicates": DataDuplicates(),
    }

    results = {}

    for name, check in checks.items():
        results[name] = check.run(dataset)
    
    # Resumen explícito de calidad de datos
    results["summary"] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_types": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
    }

    print("\n===== RESULTADOS DE DEEPCHECKS =====")

    for name in ["mixed_data_types", "mixed_nulls", "duplicates"]:
        result = results[name]

        print(f"\nChequeo: {name}")
        print(result.value)


    return results
