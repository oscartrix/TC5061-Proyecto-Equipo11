from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from loguru import logger

from evidently import Report
from evidently.presets import DataDriftPreset

from mi_bici.config import P2_FEATURES_PATH, REPORTS_DIR
from mi_bici.modeling.train import P2_FEATURES


OUTPUT_DIR = REPORTS_DIR / "data_drift"
DEFAULT_REPORT_PATH = OUTPUT_DIR / "evidently_drift_p2_months.html"


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
    logger.info("Configurando y ejecutando DataDriftPreset de Evidently...")
    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=current_data, reference_data=reference_data)

    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))
    logger.info(f"Reporte HTML guardado exitosamente en: {output_path}")

    # Extraer métricas clave estructuradas
    snapshot_dict = snapshot.dump_dict()
    metric_results = snapshot_dict.get("metric_results", {})

    metrics_summary = {}
    drifted_columns_info = {}
    for _, item in metric_results.items():
        disp_name = item.get("display_name", "")
        if "Count of Drifted Columns" in disp_name and "count" in item:
            metrics_summary["drifted_columns_count"] = item["count"].get("value")
            metrics_summary["drifted_columns_share"] = item["share"].get("value")
        elif "Value drift for" in disp_name:
            col_name = disp_name.replace("Value drift for", "").strip()
            drifted_columns_info[col_name] = item.get("value")

    metrics_summary["drift_by_column"] = drifted_columns_info
    return {
        "html_path": str(output_path),
        "metrics_summary": metrics_summary,
        "snapshot": snapshot,
    }


def main():
    """Función de entrada para ejecutar la comparación mensual de drift."""
    ref_data, curr_data, ref_month, curr_month = get_two_most_recent_months_data(
        sample_size=5000
    )
    results = generate_drift_report(
        reference_data=ref_data,
        current_data=curr_data,
        output_html_path=DEFAULT_REPORT_PATH,
    )
    summary = results["metrics_summary"]
    print("=" * 60)
    print(f"REPORTE DE DATA DRIFT ({ref_month} vs {curr_month})")
    print(f"Reporte HTML: {results['html_path']}")
    print(f"Columnas con drift detectado: {summary.get('drifted_columns_count')}")
    print(f"Proporción de drift: {summary.get('drifted_columns_share', 0):.2%}")
    print("Scores de drift por variable:")
    for col, score in summary.get("drift_by_column", {}).items():
        print(f"  - {col}: {score}")
    print("=" * 60)


if __name__ == "__main__":
    main()
