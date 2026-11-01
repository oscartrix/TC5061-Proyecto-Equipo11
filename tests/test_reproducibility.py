
import numpy as np
import pandas as pd

from mi_bici.modeling.train import build_p1_model
from mi_bici.modeling.train import build_p3_model
from mi_bici.modeling.train import build_p2_model

def test_p1_reproducibility():
    """Dos entrenamientos iguales deben producir las mismas predicciones."""

    rng = np.random.default_rng(42)
    n = 100

    X = pd.DataFrame({
        "lag_1d": rng.random(n),
        "lag_7d": rng.random(n),
        "lag_14d": rng.random(n),
        "lag_21d": rng.random(n),
        "rolling_mean_7d": rng.random(n),
        "capacity": rng.integers(10, 30, n),
        "day_of_week": rng.integers(0, 7, n),
        "month": rng.integers(1, 13, n),
        "is_weekend": rng.integers(0, 2, n),
        "is_holiday": rng.integers(0, 2, n),
        "zone": rng.choice(["centro", "norte"], n),
    })

    y = rng.random(n) * 40

    model_1 = build_p1_model("ridge", {"alpha": 1.0})
    model_2 = build_p1_model("ridge", {"alpha": 1.0})

    model_1.fit(X, y)
    model_2.fit(X, y)

    pred_1 = model_1.predict(X)
    pred_2 = model_2.predict(X)

    np.testing.assert_allclose(pred_1, pred_2, rtol=1e-10, atol=1e-10)


def test_random_forest_reproducibility():
    """Verifica que Random Forest sea reproducible con una semilla fija."""

    rng = np.random.default_rng(42)
    n = 100

    X = pd.DataFrame({
        "lag_1d": rng.random(n),
        "lag_7d": rng.random(n),
        "lag_14d": rng.random(n),
        "lag_21d": rng.random(n),
        "rolling_mean_7d": rng.random(n),
        "capacity": rng.integers(10, 30, n),
        "day_of_week": rng.integers(0, 7, n),
        "month": rng.integers(1, 13, n),
        "is_weekend": rng.integers(0, 2, n),
        "is_holiday": rng.integers(0, 2, n),
        "zone": rng.choice(["centro", "norte"], n),
    })

    y = rng.random(n) * 40

    hparams = {
        "n_estimators": 20,
        "max_depth": 5
    }

    model_1 = build_p1_model("rf", hparams)
    model_2 = build_p1_model("rf", hparams)

    model_1.fit(X, y)
    model_2.fit(X, y)

    pred_1 = model_1.predict(X)
    pred_2 = model_2.predict(X)

    np.testing.assert_allclose(
        pred_1,
        pred_2,
        rtol=1e-10,
        atol=1e-10
    )

def test_p2_reproducibility():
    """Verifica la reproducibilidad de HistGradientBoosting para P2."""

    rng = np.random.default_rng(42)
    n = 150

    # Datos sintéticos para clasificación multiclase
    X = pd.DataFrame({
        "pressure": rng.random(n),
        "capacity": rng.integers(10, 40, n),
        "hour": rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
        "risk_lag_168h": rng.random(n),
    })

    # Tres clases de riesgo
    y = np.array(
        ["deficit", "balanceado", "excedente"] * 50
    )

    hparams = {
        "max_iter": 200,
        "learning_rate": 0.08,
        "max_depth": 6
    }

    model_1 = build_p2_model("hgb", hparams)
    model_2 = build_p2_model("hgb", hparams)

    model_1.fit(X, y)
    model_2.fit(X, y)

    pred_1 = model_1.predict(X)
    pred_2 = model_2.predict(X)

    np.testing.assert_array_equal(pred_1, pred_2)

def test_kmeans_reproducibility():
    """Verifica la reproducibilidad de KMeans para P3."""

    rng = np.random.default_rng(42)
    n = 100

    # Datos sintéticos para simular perfiles de uso
    X = pd.DataFrame({
        "dep_morning": rng.random(n),
        "dep_evening": rng.random(n),
        "dep_weekend": rng.random(n),
        "median_duration_min": rng.uniform(5, 40, n),
        "round_trip_pct": rng.random(n),
        "avg_daily_trips": rng.uniform(1, 100, n),
        "dep_arr_ratio": rng.uniform(0.5, 2.0, n),
    })

    hparams = {"n_clusters": 6}

    model_1 = build_p3_model("kmeans_usage", hparams)
    model_2 = build_p3_model("kmeans_usage", hparams)

    model_1.fit(X)
    model_2.fit(X)

    labels_1 = model_1.predict(X)
    labels_2 = model_2.predict(X)

    np.testing.assert_array_equal(labels_1, labels_2)
