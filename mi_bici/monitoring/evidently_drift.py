from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from loguru import logger

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from mi_bici.config import P2_FEATURES_PATH, REPORTS_DIR
from mi_bici.modeling.train import P2_FEATURES, plain_dtypes
from mi_bici.monitoring.prepare_data import prepare_monitoring_data


OUTPUT_DIR = REPORTS_DIR / "data_drift"
DEFAULT_REPORT_PATH = OUTPUT_DIR / "evidently_drift_p2.html"

# Pruebas de drift fijas (con más de 1000 filas son las que Evidently elige por defecto).
# Ambas son distancias: hay drift cuando el score es mayor o igual al umbral.
DRIFT_NUM_METHOD = "wasserstein"
DRIFT_CAT_METHOD = "jensenshannon"
DRIFT_THRESHOLD = 0.1
CATEGORICAL_COLUMNS = ["hour", "day_of_week", "is_weekend"]


def build_data_definition(df: pd.DataFrame) -> DataDefinition:
    """Declara qué columnas son numéricas y cuáles categóricas para Evidently."""
    return DataDefinition(
        numerical_columns=[c for c in df.columns if c not in CATEGORICAL_COLUMNS],
        categorical_columns=[c for c in df.columns if c in CATEGORICAL_COLUMNS],
    )


def load_monitoring_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga la referencia (train 2024-2025) y el periodo actual (jul-sep 2026) de P2."""
    reference, current = prepare_monitoring_data()
    return plain_dtypes(reference), plain_dtypes(current)


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
    output_html_path: Path = DEFAULT_REPORT_PATH,
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
        "html_path": str(output_path),
        "metrics_summary": metrics_summary,
        "snapshot": snapshot,
    }


def main():
    """Compara la referencia de entrenamiento contra el periodo actual."""
    ref_data, curr_data = get_reference_and_current_data(sample_size=5000)
    results = generate_drift_report(
        reference_data=ref_data[P2_FEATURES],
        current_data=curr_data[P2_FEATURES],
        output_html_path=DEFAULT_REPORT_PATH,
    )
    summary = results["metrics_summary"]
    print("=" * 60)
    print("REPORTE DE DATA DRIFT (referencia train vs actual jul-sep 2026)")
    print(f"Reporte HTML: {results['html_path']}")
    print(f"Columnas con drift detectado: {summary.get('drifted_columns_count')}")
    print(f"Proporción de drift: {summary.get('drifted_columns_share', 0):.2%}")
    print("Scores de drift por variable:")
    for col, score in summary.get("drift_by_column", {}).items():
        print(f"  - {col}: {score}")
    print("=" * 60)


if __name__ == "__main__":
    main()
