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

Para relacionar el drift con el desempeño del champion (requiere `reports/evaluation/p2/by_month.csv`, que genera `mlflow run . -e evaluate -P question=p2`):

```bash
python -m mi_bici.monitoring.drift_performance
```

Salida: `reports/data_drift/drift_vs_desempeno_p2.csv`, con el drift de cada mes de ene a sep 2026 contra la referencia y las métricas del champion en ese mes.

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

## 5. Interpretación

Los scores salen de muestras de 5000 filas (semilla 42). Las medias, desviaciones y proporciones se calculan sobre los conjuntos completos.

### 5.1 Variables con drift significativo

Referencia train contra jul-sep 2026 (`train_vs_actual`):

| Variable | Score | ¿Drift? |
|---|---:|---|
| `pressure_window_lag_336h` | 0.125 | Sí |
| `pressure_window_lag_168h` | 0.125 | Sí |
| `pressure_last_3h` | 0.122 | Sí |
| `pressure_window_lag_24h` | 0.115 | Sí |
| `departures` | 0.075 | No |
| `net_flow` | 0.060 | No |
| `risk_class` (objetivo) | 0.041 | No |
| `prediction` (champion) | 0.035 | No |
| `arrivals` | 0.024 | No |

Las cuatro variables con drift son las de **presión**: el flujo neto reciente o futuro dividido entre la capacidad de la estación. Qué cambió:

- **Menos horas con presión extrema.** La desviación estándar de `pressure_last_3h` baja de 0.371 en la referencia a 0.330 en el periodo actual. Las medias casi no cambian (≈ 0.01 y 0.00).
- **Menos déficits y excedentes.** La proporción de `deficit` baja de 12.1 % a 9.2 % y la de `excedente` de 12.8 % a 10.0 %. El score de `risk_class` (0.041) queda bajo el umbral, pero el cambio de prevalencia es importante para un modelo que se evalúa por el recall de déficit.
- **No es solo estacionalidad.** Contra jul-sep de 2024 y 2025, `pressure_last_3h` (0.112) y los lags de 168 h (0.104) y 336 h (0.102) siguen por encima del umbral. En jul-sep de esos años la proporción de déficit fue de 12.3 % y 11.6 %, contra 9.2 % en 2026.
- **La tendencia empezó antes del periodo actual.** La proporción de déficit cae desde dic-2025 (8.9 %) y llega a 6.8 % en may-2026. En los meses de la referencia se movía entre 11 % y 14 %.

Evolución por mes:

- **Julio** es el mes más alejado de la referencia: presiones ≈ 0.16 y también `net_flow` (0.114).
- **Agosto** mantiene el drift en las presiones (≈ 0.11-0.15).
- **Septiembre** deja las presiones bajo el umbral (≈ 0.09), pero `departures` deriva (0.140): el promedio sube a 4.52 salidas por estación y hora, contra 3.63 en julio, 3.67 en agosto y 3.89 en la referencia. Es coherente con el regreso a clases.
- **Mayo y junio de 2026** (validación) tienen el drift más alto de todo 2026: presiones ≈ 0.20, y en mayo `departures` llega a 0.256 con un promedio de 2.75 salidas por hora. Con estos datos no se puede saber la causa (un evento externo o un problema en los datos abiertos); conviene revisarlo.

### 5.2 Implicaciones para el desempeño del modelo

Desempeño del champion (`mibici-equipo11-p2-riesgo` v1, `hgb_balanced`) por mes, junto al drift de ese mes contra la referencia:

| Mes | Split | Variables con drift | `pressure_last_3h` | F1 macro | Recall déficit | Precisión déficit |
|---|---|---:|---:|---:|---:|---:|
| 2026-01 | val | 4 | 0.114 | 0.551 | 0.766 | 0.293 |
| 2026-02 | val | 4 | 0.117 | 0.534 | 0.771 | 0.281 |
| 2026-03 | val | 4 | 0.112 | 0.551 | 0.773 | 0.290 |
| 2026-04 | val | 4 | 0.124 | 0.543 | 0.770 | 0.280 |
| 2026-05 | val | 7 | 0.204 | 0.530 | 0.763 | 0.249 |
| 2026-06 | val | 7 | 0.194 | 0.524 | 0.721 | 0.258 |
| 2026-07 | test | 5 | 0.159 | 0.543 | 0.775 | 0.271 |
| 2026-08 | test | 4 | 0.119 | 0.555 | 0.772 | 0.305 |
| 2026-09 | test | 1 | 0.088 | 0.548 | 0.778 | 0.307 |

1. **En el periodo actual el modelo no se degradó.** En jul-sep el champion tiene F1 macro de 0.549, recall de déficit de 0.775 y precisión de déficit de 0.294, iguales o mejores que en validación (0.541, 0.762 y 0.276). El drift de jul-sep es moderado y no se tradujo en una caída de desempeño.
2. **Cuando el drift es fuerte, el desempeño sí cae.** Los dos meses con más drift (mayo y junio, presiones ≈ 0.20 y 7 variables con drift) son los de peor F1 macro (0.530 y 0.524) y peor precisión de déficit (0.249 y 0.258). Junio también tiene el peor recall de déficit (0.721). En los 9 meses, el score de drift de `pressure_last_3h` tiene correlación de −0.93 con la precisión de déficit, −0.78 con el F1 macro y −0.66 con el recall de déficit.
3. **Mecanismo probable: cambio de prevalencia.** La clase objetivo se define con un umbral fijo sobre la presión de las próximas 3 horas. Si la presión tiene menos dispersión, menos horas cruzan el umbral y baja la proporción de déficit. El modelo se entrenó con pesos balanceados sobre una prevalencia de 12 % y sus alertas de déficit bajan menos que la prevalencia: en ene-abr predice déficit en ≈ 24 % de las horas con ≈ 9 % de déficits reales, y en mayo en 20.9 % con solo 6.8 % reales (≈ 6000 falsas alertas por mes en todo 2026). Por eso **la primera métrica en caer es la precisión**. El recall resiste hasta que el cambio es grande, como en junio.
4. **El champion sigue cumpliendo su objetivo operativo.** En todos los meses detecta más déficits que la regla semanal (`risk_lag_168h`): recall de 0.72 a 0.78 contra 0.34 a 0.42. Mantiene el costo que ya tenía al seleccionarse: menor F1 macro que la regla (0.549 contra 0.588 en jul-sep) y unas 3.5 falsas alertas por cada déficit adicional detectado.

**Qué implica para el proyecto**

- El drift actual **no obliga a un reentrenamiento urgente**: las métricas de jul-sep están al nivel de validación.
- El riesgo está en la **tendencia**: la proporción de déficit baja desde dic-2025 y el drift en las presiones se mantiene aun descontando la estacionalidad. Un reentrenamiento con datos de 2026 permitiría ajustar el modelo a la nueva prevalencia; eso es lo que evalúa la versión challenger (M3) en el A/B testing (M4).
- El score de drift de las variables de presión es la señal de monitoreo más útil: coincide con las caídas de precisión y F1. Es candidato a disparador de continuous training (M5).

**Limitaciones**

- Con 9 meses, las correlaciones son indicativas y no prueban causalidad. Los scores cercanos al umbral pueden cambiar con otra muestra.
- Los meses de validación se usaron para elegir el champion, así que sus métricas no son completamente independientes.
- La evaluación corresponde al champion v1 (corrida `bbb0b0fa`) y a los datos procesados con `data_version` `a6306ebfef13`. Si el equipo regenera los datos o cambia el champion, los números deben recalcularse.

Evidencia: `docs/evidencia/monitoreo/02_drift_vs_desempeno.txt`, `drift_vs_desempeno_p2.csv`, `p2_by_month.csv` y `p2_evaluation_metadata.json`.
