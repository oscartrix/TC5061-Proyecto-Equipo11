
"""Pruebas de validación de datos con DeepChecks."""

import numpy as np
import pandas as pd
from deepchecks.core import CheckFailure

from mi_bici.validation.deepchecks_validation import validate_data_quality


def test_deepchecks_data_quality():
    """Verifica que DeepChecks pueda analizar datos con problemas de calidad."""

    df = pd.DataFrame({
        "station_id": [1, 2, 3, 3, 5, 6],
        "capacity": [20, 30, np.nan, np.nan, 15, 25],
        "daily_trips": [100, 120, 80, 80, 90, 110],
        "zone": ["centro", "norte", "sur", "sur", "centro", "norte"],
    })

    results = validate_data_quality(df)
    print("\n===== RESUMEN DE CALIDAD DE DATOS =====")

    summary = results["summary"]

    print(f"Total de registros: {summary['total_rows']}")
    print(f"Total de columnas: {summary['total_columns']}")

    print("\nValores faltantes:")
    for column, count in summary["missing_values"].items():
        print(f"  {column}: {count}")

    print(f"\nFilas duplicadas: {summary['duplicate_rows']}")

    print("\nTipos de datos:")
    for column, dtype in summary["column_types"].items():
        print(f"  {column}: {dtype}")

    assert "mixed_data_types" in results
    assert "mixed_nulls" in results
    assert "duplicates" in results

    assert results["duplicates"] is not None

    
    # Comprobar que se detecten los problemas esperados
    summary = results["summary"]

    assert summary["total_rows"] == 6
    assert summary["total_columns"] == 4

    # Dos valores faltantes en capacity
    assert summary["missing_values"]["capacity"] == 2

    # Una fila duplicada
    assert summary["duplicate_rows"] == 1

    # Verificar los tipos de datos
    assert summary["column_types"]["station_id"] == "int64"
    assert summary["column_types"]["capacity"] == "float64"
    assert not results['mixed_nulls'].passed_conditions()
    assert not results['duplicates'].passed_conditions()
    assert any('mixed_nulls' in error for error in results['errors'])
    assert any('duplicates' in error for error in results['errors'])


def test_mixed_types_fail_deepchecks_condition():
    result = validate_data_quality(pd.DataFrame({'value': [1, 2, 'incorrecto', 3, 4, 5]}))
    assert not result['mixed_data_types'].passed_conditions()
    assert any('mixed_data_types' in error for error in result['errors'])


def test_clean_data_passes_deepchecks_conditions():
    result = validate_data_quality(pd.DataFrame({'value': range(10)}))
    assert result['errors'] == []


def test_deepchecks_execution_failure_is_reported(monkeypatch):
    from mi_bici.validation import deepchecks_validation

    monkeypatch.setattr(deepchecks_validation.MixedNulls, 'run', lambda self, dataset:
                        CheckFailure(self, RuntimeError('chequeo no disponible')))
    result = validate_data_quality(pd.DataFrame({'value': range(10)}))
    assert any('mixed_nulls' in error and 'chequeo no disponible' in error
               for error in result['errors'])

