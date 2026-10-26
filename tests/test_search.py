
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

from mi_bici.modeling.search import is_better_score


def test_minimize_mae():
    """P1 debe seleccionar un MAE menor."""
    assert is_better_score(5.5, 5.8, "min")
    assert not is_better_score(6.0, 5.8, "min")


def test_maximize_f1():
    """P2 debe seleccionar un F1-macro mayor."""
    assert is_better_score(0.65, 0.60, "max")
    assert not is_better_score(0.55, 0.60, "max")


def test_equal_scores_do_not_replace_best():
    """Un empate conserva la primera configuración encontrada."""
    assert not is_better_score(5.5817, 5.5817, "min")
    assert not is_better_score(0.60, 0.60, "max")


def test_invalid_direction():
    """Una dirección desconocida debe generar un error."""
    with pytest.raises(ValueError, match="Dirección de optimización inválida"):
        is_better_score(0.5, 0.6, "invalid")
