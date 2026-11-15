import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from loguru import logger

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from mi_bici.config import P2_FEATURES_PATH, PROCESSED_DATA_DIR, REPORTS_DIR
from mi_bici.modeling.train import P2_FEATURES, plain_dtypes
from mi_bici.monitoring.prepare_data import prepare_monitoring_data


OUTPUT_DIR = REPORTS_DIR / "data_drift"
DEFAULT_REPORT_PATH = OUTPUT_DIR / "evidently_drift_p2.html"

# Pruebas de drift fijas (con más de 1000 filas son las que Evidently elige por defecto).
# Ambas son distancias: hay drift cuando el score es mayor o igual al umbral.
DRIFT_NUM_METHOD = "wasserstein"
DRIFT_CAT_METHOD = "jensenshannon"
DRIFT_THRESHOLD = 0.1
CATEGORICAL_COLUMNS = ["hour", "day_of_week", "is_weekend", "risk_class", "prediction"]

# Predicciones del champion generadas con: mlflow run . -e predict -P question=p2
P2_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "p2_predictions.parquet"
MONITORED_COLUMNS = [*P2_FEATURES, "risk_class", "prediction"]

# Variables fijas por construcción (malla de estaciones x horas): no pueden derivar,
# por lo que se excluyen de la proporción de columnas con drift del resumen.
STATIC_COLUMNS = ["hour", "day_of_week", "is_weekend", "capacity"]
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "evidently_drift_p2_summary.json"


def build_data_definition(df: pd.DataFrame) -> DataDefinition:
    """Declara qué columnas son numéricas y cuáles categóricas para Evidently."""
    return DataDefinition(
        numerical_columns=[c for c in df.columns if c not in CATEGORICAL_COLUMNS],
        categorical_columns=[c for c in df.columns if c in CATEGORICAL_COLUMNS],
    )


def add_predictions(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    predictions_path: Path = P2_PREDICTIONS_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Une las predicciones del champion por estación y hora, si el archivo existe."""
    if not Path(predictions_path).exists():
        logger.warning(
            f"No se encontraron predicciones en {predictions_path}; "
            "el reporte no incluirá el drift de las predicciones."
        )
        return reference, current

    predictions = pd.read_parquet(
        predictions_path, columns=["station_id", "hour_ts", "prediction"]
    )
    reference = reference.merge(predictions, on=["station_id", "hour_ts"], how="left")
    current = current.merge(predictions, on=["station_id", "hour_ts"], how="left")
    return reference, current


def load_monitoring_frames(
    predictions_path: Path = P2_PREDICTIONS_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga la referencia (train 2024-2025) y el periodo actual (jul-sep 2026) de P2."""
    reference, current = prepare_monitoring_data()
    reference, current = add_predictions(reference, current, predictions_path)
    return plain_dtypes(reference), plain_dtypes(current)


def monitored_columns(df: pd.DataFrame) -> list:
    """Variables, objetivo y predicción presentes en el DataFrame."""
    return [c for c in MONITORED_COLUMNS if c in df.columns]


def sample_rows(df: pd.DataFrame, sample_size: int, random_state: int = 42) -> pd.DataFrame:
    """Toma una muestra aleatoria para que el reporte HTML sea manejable."""
    if sample_size and len(df) > sample_size:
        return df.sample(n=sample_size, random_state=random_state)
    return df


def get_reference_and_current_data(
    sample_size: int = 5000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Muestras de la referencia de entrenamiento y del periodo actual."""
    reference, current = load_monitoring_frames()
    reference = sample_rows(reference, sample_size, random_state)
    current = sample_rows(current, sample_size, random_state)
    logger.info(
        f"Muestras listas - Referencia: {reference.shape}, Actual: {current.shape}"
    )
    return reference, current


def get_two_most_recent_months_data(
    sample_size: int = 5000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Carga las variables de P2 y extrae los datos de los dos meses más recientes.

    Returns:
        reference_data: DataFrame del penúltimo mes (referencia).
        current_data: DataFrame del último mes (actual).
        reference_month: Cadena con el año-mes de referencia (ej. '2026-08').
        current_month: Cadena con el año-mes actual (ej. '2026-09').
    """
    if not P2_FEATURES_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {P2_FEATURES_PATH}")

    logger.info(f"Cargando dataset de características P2 desde {P2_FEATURES_PATH}...")
    df = pd.read_parquet(P2_FEATURES_PATH)

    if "hour_ts" not in df.columns:
        raise ValueError("La columna 'hour_ts' no está en el dataset de P2.")

    df["month"] = pd.to_datetime(df["hour_ts"]).dt.to_period("M").astype(str)
    unique_months = sorted(df["month"].unique())

    if len(unique_months) < 2:
        raise ValueError(
            f"Se requieren al menos 2 meses distintos, encontrados: {unique_months}"
        )

    reference_month = unique_months[-2]
    current_month = unique_months[-1]

    logger.info(f"Mes de referencia: {reference_month} | Mes actual: {current_month}")

    ref_subset = df.loc[df["month"] == reference_month, P2_FEATURES].copy()
    curr_subset = df.loc[df["month"] == current_month, P2_FEATURES].copy()

    if ref_subset.empty or curr_subset.empty:
        raise ValueError("Uno de los subconjuntos mensuales está vacío.")

    # Muestreo estratificado / representativo para balance y rendimiento del reporte
    if sample_size and len(ref_subset) > sample_size:
        ref_subset = ref_subset.sample(n=sample_size, random_state=random_state)
    if sample_size and len(curr_subset) > sample_size:
        curr_subset = curr_subset.sample(n=sample_size, random_state=random_state)

    logger.info(
        f"Muestras listas - Referencia: {ref_subset.shape}, Actual: {curr_subset.shape}"
    )
    return ref_subset, curr_subset, reference_month, current_month


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_html_path: Optional[Path] = DEFAULT_REPORT_PATH,
) -> Dict[str, Any]:
    """Ejecuta el análisis de Data Drift de Evidently y guarda el reporte HTML."""
    logger.info("Configurando y ejecutando DataDriftPreset y DataSummaryPreset de Evidently...")
    report = Report(
        [
            DataDriftPreset(
                num_method=DRIFT_NUM_METHOD,
                cat_method=DRIFT_CAT_METHOD,
                threshold=DRIFT_THRESHOLD,
            ),
            DataSummaryPreset(),
        ]
    )
    data_definition = build_data_definition(reference_data)
    snapshot = report.run(
        current_data=Dataset.from_pandas(current_data, data_definition=data_definition),
        reference_data=Dataset.from_pandas(reference_data, data_definition=data_definition),
    )

    output_path = None
    if output_html_path is not None:
        output_path = Path(output_html_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_path))
        logger.info(f"Reporte HTML guardado exitosamente en: {output_path}")

    # Extraer métricas clave estructuradas
    snapshot_dict = snapshot.dump_dict()
    metric_results = snapshot_dict.get("metric_results", {})

    metrics_summary = {}
    drifted_columns_info = {}
    drift_details = {}
    data_quality = {}
    for _, item in metric_results.items():
        disp_name = item.get("display_name", "")
        if "Count of Drifted Columns" in disp_name and "count" in item:
            metrics_summary["drifted_columns_count"] = item["count"].get("value")
            metrics_summary["drifted_columns_share"] = item["share"].get("value")
        elif "Value drift for" in disp_name:
            col_name = disp_name.replace("Value drift for", "").strip()
            score = item.get("value")
            params = item["metric_value_location"]["metric"]["params"]
            drifted_columns_info[col_name] = score
            # Wasserstein y Jensen-Shannon son distancias: hay drift si score >= umbral
            drift_details[col_name] = {
                "method": params.get("method"),
                "threshold": params.get("threshold"),
                "score": score,
                "drift": bool(score >= params.get("threshold")),
            }
        elif disp_name.endswith("in dataset") and isinstance(item.get("value"), (int, float)):
            data_quality[disp_name] = item["value"]
        elif disp_name == "Count and share of missing values in dataset":
            data_quality["Missing values count"] = item["count"].get("value")
            data_quality["Missing values share"] = item["share"].get("value")

    metrics_summary["drift_by_column"] = drifted_columns_info
    metrics_summary["drift_details"] = drift_details
    metrics_summary["data_quality_current"] = data_quality
    return {
        "html_path": str(output_path) if output_path else None,
        "metrics_summary": metrics_summary,
        "snapshot": snapshot,
    }


def period_of(df: pd.DataFrame) -> Dict[str, Any]:
    """Rango de fechas y filas de un conjunto de monitoreo."""
    period = {"rows": int(len(df))}
    if "hour_ts" in df.columns and len(df):
        period["start"] = str(df["hour_ts"].min())
        period["end"] = str(df["hour_ts"].max())
    return period


def summarize_comparison(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    metrics_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Resumen de una comparación con la proporción de drift sin las variables fijas."""
    details = metrics_summary.get("drift_details", {})
    monitored = {col: d for col, d in details.items() if col not in STATIC_COLUMNS}
    drifted = [col for col, d in monitored.items() if d["drift"]]
    return {
        "reference": period_of(reference_data),
        "current": period_of(current_data),
        "drifted_columns": drifted,
        "drifted_share_monitored": len(drifted) / len(monitored) if monitored else 0.0,
        "excluded_static_columns": [c for c in STATIC_COLUMNS if c in details],
        "evidently_drifted_columns_share": metrics_summary.get("drifted_columns_share"),
        "columns": details,
        "data_quality_current": metrics_summary.get("data_quality_current", {}),
    }


def save_drift_summary(summary: Dict[str, Any], output_path: Path = DEFAULT_SUMMARY_PATH) -> Path:
    """Guarda el resumen de drift en JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Resumen JSON guardado en: {output_path}")
    return output_path


def compare(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_html_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Corre una comparación de Evidently y devuelve su resumen."""
    columns = monitored_columns(reference_data)
    results = generate_drift_report(
        reference_data=reference_data[columns],
        current_data=current_data[columns],
        output_html_path=output_html_path,
    )
    return summarize_comparison(reference_data, current_data, results["metrics_summary"])


def additional_comparisons(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    sample_size: int = 5000,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Drift de cada mes del periodo actual y contra los mismos meses de años anteriores.

    - ``train_vs_<mes>``: referencia completa contra cada mes actual, para ver la evolución.
    - ``mismos_meses_train_vs_actual``: solo los meses del periodo actual dentro de la
      referencia (jul-sep 2024 y 2025), para separar la estacionalidad de un cambio real.
    """
    comparisons = {}
    reference_sample = sample_rows(reference, sample_size, random_state)
    current_months = current["hour_ts"].dt.to_period("M")
    for month in sorted(current_months.unique()):
        month_rows = sample_rows(current[current_months == month], sample_size, random_state)
        comparisons[f"train_vs_{month}"] = compare(reference_sample, month_rows)

    same_months = reference[reference["hour_ts"].dt.month.isin(current["hour_ts"].dt.month.unique())]
    if not same_months.empty:
        comparisons["mismos_meses_train_vs_actual"] = compare(
            sample_rows(same_months, sample_size, random_state),
            sample_rows(current, sample_size, random_state),
        )
    return comparisons


def main():
    """Compara la referencia de entrenamiento contra el periodo actual."""
    reference, current = load_monitoring_frames()
    ref_data = sample_rows(reference, 5000)
    curr_data = sample_rows(current, 5000)
    columns = monitored_columns(ref_data)
    results = generate_drift_report(
        reference_data=ref_data[columns],
        current_data=curr_data[columns],
        output_html_path=DEFAULT_REPORT_PATH,
    )
    summary = results["metrics_summary"]
    drift_summary = {
        "method": {
            "numerical": DRIFT_NUM_METHOD,
            "categorical": DRIFT_CAT_METHOD,
            "threshold": DRIFT_THRESHOLD,
        },
        "comparisons": {
            "train_vs_actual": summarize_comparison(ref_data, curr_data, summary),
            **additional_comparisons(reference, current),
        },
    }
    summary_path = save_drift_summary(drift_summary)
    print("=" * 60)
    print("REPORTE DE DATA DRIFT (referencia train vs actual jul-sep 2026)")
    print(f"Reporte HTML: {results['html_path']}")
    print(f"Resumen JSON: {summary_path}")
    print(f"Columnas con drift detectado: {summary.get('drifted_columns_count')}")
    print(f"Proporción de drift: {summary.get('drifted_columns_share', 0):.2%}")
    print("Variables con drift por comparación:")
    for name, comparison in drift_summary["comparisons"].items():
        print(
            f"  - {name}: {comparison['drifted_columns']} "
            f"({comparison['drifted_share_monitored']:.0%} de las variables monitoreadas)"
        )
    print("Scores de drift por variable (train vs actual):")
    for col, score in summary.get("drift_by_column", {}).items():
        print(f"  - {col}: {score}")
    print("=" * 60)


if __name__ == "__main__":
    main()
