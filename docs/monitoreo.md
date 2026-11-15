# Monitoreo de data drift con Evidently — Equipo 11

Modelo monitoreado: champion de P2 (`mibici-equipo11-p2-riesgo`), clasificación de riesgo de déficit, balance o excedente de bicis en las próximas 3 horas.
Código: `mi_bici/monitoring/prepare_data.py` y `mi_bici/monitoring/evidently_drift.py`
Notebook: `notebooks/02-monitoreo-evidently-drift.ipynb`

## 1. Periodos elegidos

| Conjunto | Periodo | Filas | Origen |
|---|---|---:|---|
| Referencia | 2024-01-15 a 2025-12-31 | 860 400 | `split = train` de `p2_features.parquet` |
| Actual | 2026-07-01 a 2026-09-30 | 110 250 | `split = test` de `p2_features.parquet` |

**Por qué estos periodos**

- **Referencia = datos de entrenamiento.** El champion de P2 se entrenó solo con `split = train`; la validación (ene-jun 2026) se usó para elegir el modelo. Comparar contra lo que el modelo vio es lo que indica si los datos de producción se alejaron.
- **Actual = el periodo más reciente de los datos abiertos de MiBici.** Jul-sep 2026 es posterior al entrenamiento y a la validación, el modelo nunca lo vio y cubre dos situaciones distintas: vacaciones de verano (jul-ago) y regreso a clases (sep).
- **Inicio el 15 de enero de 2024.** Las filas anteriores no tienen el lag de 336 h y se descartan al construir las características.
- **No se usan peticiones de la API.** La API de la semana 8 sirve P1 y no registra las peticiones, por lo que los datos abiertos son la única fuente del periodo de producción de P2.

Cada conjunto incluye `station_id` y `hour_ts` (solo para partir por mes), las 11 variables de entrada de P2, la clase objetivo `risk_class` y la predicción del champion (`data/processed/p2_predictions.parquet`, generado con `mlflow run . -e predict -P question=p2`).

## 2. Método

- **Presets:** `DataDriftPreset` (drift por variable) y `DataSummaryPreset` (calidad de datos: faltantes, duplicados, columnas vacías o constantes).
- **Pruebas fijas:** distancia de Wasserstein normalizada para variables numéricas y Jensen-Shannon para categóricas (`hour`, `day_of_week`, `is_weekend`, `risk_class`, `prediction`). Son las que Evidently elige por defecto con más de 1000 filas; se fijan para que no cambien con el tamaño de la muestra.
- **Umbral:** 0.1. Ambas son distancias, así que **hay drift cuando el score es mayor o igual a 0.1**. No son p-values: un score menor a 0.05 indica distribuciones parecidas, no drift.
- **Muestra:** 5000 filas por conjunto (semilla 42) para que el reporte HTML sea manejable. Los scores cercanos al umbral pueden variar con otra muestra.
- **Variables fijas:** `hour`, `day_of_week`, `is_weekend` y `capacity` no pueden derivar porque los datos son una malla completa de 50 estaciones por todas las horas. Se reportan, pero no cuentan en la proporción de variables con drift del resumen.

### Comparaciones

| Nombre | Referencia | Actual | Para qué |
|---|---|---|---|
| `train_vs_actual` | train | jul-sep 2026 | Reporte principal (HTML) |
| `train_vs_2026-07`, `-08`, `-09` | train | cada mes | Evolución del drift mes a mes |
| `mismos_meses_train_vs_actual` | jul-sep 2024 y 2025 | jul-sep 2026 | Separar la estacionalidad de un cambio real |

## 3. Cómo ejecutarlo

```bash
make drift-report
```

Salidas (en `reports/data_drift/`, ignorado por git):

- `evidently_drift_p2.html`: reporte interactivo de Evidently de `train_vs_actual`.
- `evidently_drift_p2_summary.json`: prueba, umbral, score y decisión por variable de todas las comparaciones, con la calidad de datos del periodo actual.

La evidencia de la entrega se copia a `docs/evidencia/monitoreo/`.

## 4. Relación con el drift de DeepChecks

`make validate` (`mi_bici/validation/drift_validation.py`) ya mide el drift de train contra test con DeepChecks. Los dos chequeos tienen fines distintos:

| | DeepChecks (`make validate`) | Evidently (`make drift-report`) |
|---|---|---|
| Objetivo | Compuerta del pipeline: detiene el CI si los datos no son aptos para entrenar | Monitoreo: describir e interpretar el drift del periodo de producción |
| Pruebas | Kolmogorov-Smirnov y Cramér's V | Wasserstein y Jensen-Shannon |
| Umbrales | Advertencia 0.20, error 0.50 | Drift a partir de 0.1 |
| Alcance | P1, P2 y P3 | P2, con objetivo, predicciones y comparaciones por mes |

Los umbrales no se comparan entre sí porque las pruebas son distintas. El umbral de Evidently es más sensible a propósito: busca señales tempranas para decidir si conviene reentrenar, mientras que DeepChecks solo bloquea cambios graves.
