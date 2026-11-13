
"""Pruebas de preparación de datos para monitoreo P2."""

from mi_bici.monitoring.prepare_data import prepare_monitoring_data
from mi_bici.modeling.train import P2_FEATURES


def test_monitoring_data():
    reference, current = prepare_monitoring_data()

    # Los conjuntos deben contener observaciones
    assert not reference.empty
    assert not current.empty

    # Ambos deben utilizar las características del modelo P2
    assert list(reference.columns) == list(P2_FEATURES)
    assert list(current.columns) == list(P2_FEATURES)

    # Deben conservar los mismos tipos de datos
    assert reference.dtypes.equals(current.dtypes)

    # No deben contener valores faltantes
    assert not reference.isna().any().any()
    assert not current.isna().any().any()
