
"""Preparación de datos de referencia y actuales para monitoreo P2."""

import pandas as pd

from mi_bici.config import P2_FEATURES_PATH
from mi_bici.modeling.train import P2_FEATURES


def prepare_monitoring_data():
    """Obtiene los conjuntos temporales para detectar data drift."""

    df = pd.read_parquet(P2_FEATURES_PATH)

    reference = df.loc[
        df["split"] == "train", P2_FEATURES
    ].copy()

    current = df.loc[
        df["split"] == "test", P2_FEATURES
    ].copy()

    if reference.empty or current.empty:
        raise ValueError("Los conjuntos de monitoreo están vacíos.")

    if reference.isna().any().any() or current.isna().any().any():
        raise ValueError("Se encontraron valores faltantes en las variables P2.")

    return reference, current


if __name__ == "__main__":
    reference, current = prepare_monitoring_data()

    print(f"Referencia: {reference.shape}")
    print(f"Actual: {current.shape}")
    print(f"Variables: {list(reference.columns)}")
