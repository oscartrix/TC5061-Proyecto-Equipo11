
"""Pruebas de generación de combinaciones para búsqueda de hiperparámetros."""

import pytest

from mi_bici.config import SEARCH_GRIDS
from mi_bici.modeling.search import generate_combinations


def test_generate_combinations_p1():
    """P1 debe generar las seis configuraciones de Ridge."""
    combinations = generate_combinations(SEARCH_GRIDS["p1"]["ridge"])

    assert len(combinations) == 6
    assert {"alpha": 0.1} in combinations
    assert {"alpha": 10000.0} in combinations


def test_generate_combinations_p2():
    """P2 debe generar las doce configuraciones de HGB."""
    combinations = generate_combinations(SEARCH_GRIDS["p2"]["hgb"])

    assert len(combinations) == 12

    assert {
        "max_iter": 200,
        "learning_rate": 0.08,
        "max_depth": 6,
    } in combinations


def test_generate_combinations_p3():
    """P3 debe generar las cuatro configuraciones de KMeans."""
    combinations = generate_combinations(
        SEARCH_GRIDS["p3"]["kmeans_usage"]
    )

    assert len(combinations) == 4
    assert {"n_clusters": 6} in combinations


def test_generate_combinations_empty_grid():
    """Un grid sin parámetros representa una sola configuración."""
    combinations = generate_combinations({})

    assert combinations == [{}]


def test_generate_combinations_no_duplicates():
    """Las configuraciones generadas no deben repetirse."""
    combinations = generate_combinations(SEARCH_GRIDS["p2"]["hgb"])

    unique = {
        tuple(sorted(params.items()))
        for params in combinations
    }

    assert len(unique) == len(combinations)
