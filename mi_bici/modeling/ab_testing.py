
"""Evaluación A/B offline: Champion vs Retrain para P2."""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from scipy.stats import binomtest

from mi_bici.config import (
    MLFLOW_TRACKING_URI,
    P2_FEATURES_PATH,
    REGISTERED_MODELS,
)
from mi_bici.modeling.evaluation import class_statistics
from mi_bici.modeling.train import P2_FEATURES


LABELS = ["deficit", "balanceado", "excedente"]
OUTPUT_DIR = Path("reports/ab_testing")


def load_ab_test_data(
    start_date="2026-09-01",
    end_date="2026-09-30",
):
    """Carga exclusivamente los datos test del periodo A/B."""

    df = pd.read_parquet(P2_FEATURES_PATH)

    required = [*P2_FEATURES, "hour_ts", "risk_class", "split"]
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    timestamps = pd.to_datetime(df["hour_ts"])
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    mask = (
        df["split"].eq("test")
        & timestamps.ge(start)
        & timestamps.lt(end)
    )

    test_df = df.loc[mask].copy()

    if test_df.empty:
        raise ValueError("No hay datos para el periodo A/B.")

    if test_df[required].isna().any().any():
        raise ValueError("Hay valores faltantes.")

    return test_df.sort_values(
        "hour_ts", kind="stable"
    ).reset_index(drop=True)


def load_ab_models():
    """Carga las versiones Champion y Retrain desde MLflow."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient(MLFLOW_TRACKING_URI)
    name = REGISTERED_MODELS["p2"][0]

    versions = {
        alias: client.get_model_version_by_alias(name, alias)
        for alias in ("champion", "retrain")
    }

    models = {
        alias: mlflow.pyfunc.load_model(
            f"models:/{name}/{version.version}"
        )
        for alias, version in versions.items()
    }

    return models, versions


def save_confusion_matrix(confusion, alias):
    """Guarda la matriz como CSV y PNG."""

    matrix = pd.DataFrame(
        confusion,
        index=LABELS,
        columns=LABELS,
    )

    matrix.to_csv(OUTPUT_DIR / f"{alias}_confusion.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusión - {alias}")

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(
                j, i, str(confusion[i, j]),
                ha="center", va="center",
                color="white" if confusion[i, j] >
                confusion.max() / 2 else "black",
            )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{alias}_confusion.png",
        dpi=200,
    )
    plt.close(fig)


def statistical_comparison(y, pred_a, pred_b, df):
    """McNemar y bootstrap pareado para recall de déficit."""

    correct_a = pred_a == y
    correct_b = pred_b == y

    both_correct = int(np.sum(correct_a & correct_b))
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    # McNemar exacto: discordancias de acierto global.
    discordant = only_a + only_b
    p_value = (
        binomtest(only_a, discordant, 0.5).pvalue
        if discordant else 1.0
    )

    # Diferencia de recall para registros con déficit real.
    deficit_mask = y == "deficit"
    detected_a = pred_a[deficit_mask] == "deficit"
    detected_b = pred_b[deficit_mask] == "deficit"

    observed_diff = (
        detected_b.mean() - detected_a.mean()
    )

    # Bootstrap pareado por observación.
    rng = np.random.default_rng(42)
    n = len(detected_a)
    differences = np.empty(2000)

    for i in range(2000):
        indices = rng.integers(0, n, size=n)
        differences[i] = (
            detected_b[indices].mean()
            - detected_a[indices].mean()
        )

    lower, upper = np.percentile(
        differences, [2.5, 97.5]
    )

    results = {
        "both_correct": both_correct,
        "only_champion_correct": only_a,
        "only_retrain_correct": only_b,
        "both_wrong": both_wrong,
        "mcnemar_p_value": p_value,
        "recall_deficit_difference": observed_diff,
        "recall_deficit_ci_lower": lower,
        "recall_deficit_ci_upper": upper,
    }

    pd.Series(results).to_csv(
        OUTPUT_DIR / "statistical_results.csv",
        header=["value"],
    )

    # Gráfica del intervalo de confianza.
    fig, ax = plt.subplots(figsize=(7, 3))

    ax.errorbar(
        observed_diff * 100,
        0,
        xerr=np.array([
            [(observed_diff - lower) * 100],
            [(upper - observed_diff) * 100],
        ]),
        fmt="o",
        capsize=6,
    )

    ax.axvline(0, color="gray", linestyle="--")
    ax.set_yticks([])
    ax.set_xlabel("Diferencia de recall déficit (puntos porcentuales)")
    ax.set_title("Retrain - Champion | IC 95% bootstrap")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "recall_confidence_interval.png",
        dpi=200,
    )
    plt.close(fig)

    return results


def save_comparison_chart(comparison):
    """Genera gráfica comparativa de métricas principales."""

    metrics = [
        "accuracy",
        "f1_macro",
        "recall_deficit",
        "precision_deficit",
    ]

    labels = [
        "Accuracy",
        "F1 macro",
        "Recall déficit",
        "Precision déficit",
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        comparison.loc["champion", metrics] * 100,
        width,
        label="Champion v1",
    )

    ax.bar(
        x + width / 2,
        comparison.loc["retrain", metrics] * 100,
        width,
        label="Retrain v2",
    )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Porcentaje (%)")
    ax.set_title("A/B Testing P2 - Septiembre 2026")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "metrics_comparison.png",
        dpi=200,
    )
    plt.close(fig)


def compare_ab_models():
    """Ejecuta la evaluación A/B completa."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_ab_test_data()
    X = df[P2_FEATURES]
    y = df["risk_class"].to_numpy()

    models, versions = load_ab_models()

    print("\n" + "=" * 50)
    print("A/B TESTING OFFLINE - P2")
    print("=" * 50)
    print(f"Registros: {len(df)}")
    print("Periodo: septiembre 2026")

    results = {}
    predictions = {}

    for alias, model in models.items():

        pred = np.asarray(model.predict(X))

        if len(pred) != len(y):
            raise ValueError(
                f"{alias}: número incorrecto de predicciones"
            )

        predictions[alias] = pred

        metrics, per_class, confusion = class_statistics(
            y, pred
        )

        results[alias] = metrics

        per_class.to_csv(
            OUTPUT_DIR / f"{alias}_per_class.csv",
            index=False,
        )

        save_confusion_matrix(confusion, alias)

        print(
            f"\n{alias.upper()} "
            f"v{versions[alias].version}"
        )
        print(pd.Series(metrics).to_string())

    # Comparación de métricas.
    comparison = pd.DataFrame(results).T
    comparison.index.name = "modelo"

    comparison.to_csv(
        OUTPUT_DIR / "comparison_metrics.csv"
    )

    differences = (
        comparison.loc["retrain"]
        - comparison.loc["champion"]
    )

    differences.rename(
        "retrain_minus_champion"
    ).to_csv(
        OUTPUT_DIR / "comparison_differences.csv"
    )

    save_comparison_chart(comparison)

    # Pruebas estadísticas.
    stats = statistical_comparison(
        y,
        predictions["champion"],
        predictions["retrain"],
        df,
    )

    print("\n" + "=" * 50)
    print("COMPARACIÓN A/B")
    print("=" * 50)
    print(comparison.to_string())

    print("\nRESULTADOS ESTADÍSTICOS")
    print(pd.Series(stats).to_string())

    print("\nCONCLUSIÓN PRELIMINAR")

    if stats["mcnemar_p_value"] >= 0.05:
        print(
            "No se encontró una diferencia significativa "
            "en el acierto global (McNemar, alfa=0.05)."
        )
    else:
        print(
            "Se encontró una diferencia significativa "
            "en el acierto global (McNemar, alfa=0.05)."
        )

    if (
        comparison.loc["retrain", "recall_deficit"]
        <= comparison.loc["champion", "recall_deficit"]
    ):
        print(
            "Retrain no mejora la métrica prioritaria. "
            "Se recomienda mantener Champion por ahora."
        )
    else:
        print(
            "Retrain mejora la métrica prioritaria. "
            "Evaluar promoción considerando incertidumbre "
            "y costos operativos."
        )

    print(f"\nResultados en: {OUTPUT_DIR.resolve()}")

    return comparison, stats


if __name__ == "__main__":
    compare_ab_models()
