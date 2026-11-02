
"""Validaciones de calidad de datos para MiBici."""

import pandas as pd
import numpy as np
# Compatibilidad temporal entre DeepChecks 0.19.1 y NumPy 2.x
if not hasattr(np, "Inf"):
    np.Inf = np.inf

from deepchecks.tabular import Dataset
from deepchecks.core import CheckFailure
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
        "mixed_data_types": MixedDataTypes().add_condition_rare_type_ratio_not_in_range((0, 1)),
        "mixed_nulls": MixedNulls().add_condition_different_nulls_less_equal_to(0),
        "duplicates": DataDuplicates().add_condition_ratio_less_or_equal(0),
    }

    results = {"errors": []}

    for name, check in checks.items():
        result = check.run(dataset)
        results[name] = result
        if isinstance(result, CheckFailure):
            results["errors"].append(f"DeepChecks {name}: {result.exception}")
        elif not result.passed_conditions(fail_if_warning=True):
            results["errors"].append(f"DeepChecks {name}: condición de aceptación incumplida")
    
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
        print(result.exception if isinstance(result, CheckFailure) else result.value)


    return results
