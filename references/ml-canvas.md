# ML Canvas: MiBici (Equipo 11)

| # | Pregunta de negocio | Tipo de tarea | Quién la usaría |
|---|---|---|---|
| **P1** | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Operador del sistema, para planear la flota. |
| **P2** | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | Clasificación | Equipo de rebalanceo de bicicletas. |
| **P3** | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | *Clustering* | Planeación urbana y expansión del sistema. |

---

## 1. Value Proposition (Propuesta de Valor)

### Qué
Anticipar la demanda diaria de bicicletas por estación (P1), clasificar cada hora el riesgo de que una estación se quede sin bicicletas o sin anclajes disponibles en las próximas 3 horas (P2), y segmentar las estaciones según sus patrones horarios y operativos de uso en el Área Metropolitana de Guadalajara (P3).

### Por qué
- **P1 (Regresión):** Una flotilla de bicicletas mal distribuida genera bicicletas ociosas en ciertas estaciones y demanda insatisfecha en otras. Predecir cuántos viajes iniciarán cada día por estación permite planear el inventario inicial diario y dimensionar la flota activa por zona (`GDL`, `ZPN`, `TLQ`).
- **P2 (Clasificación):** Una estación vacía impide iniciar viajes y una estación llena impide devolver bicicletas al finalizar un recorrido. Anticipar en una ventana de **3 horas** si una estación entrará en estado de `deficit` o `excedente` permite priorizar las rutas de las camionetas de rebalanceo antes de que ocurra la falla hacia el usuario.
- **P3 (Clustering):** Sustituir decisiones intuitivas por planeación urbana basada en datos, identificando qué estaciones funcionan como nodos laborales/receptores, residenciales/emisores, recreativos o periféricos.

### Quién y qué decisión toma
| Pregunta | Usuario | Decisión operativa / estratégica |
|---|---|---|
| **P1** | Operador del sistema MiBici | Asignar la dotación diaria de bicicletas por estación y programar el mantenimiento de flota sin afectar horas pico. |
| **P2** | Equipo de rebalanceo de bicicletas | Ordenar las rutas de camiones cada hora: llevar bicicletas a estaciones con predicción de `deficit` y retirar de estaciones con `excedente`. |
| **P3** | Planeación urbana (IMEPLAN / Agencia Metropolitana) | Definir dónde instalar nuevas estaciones o ampliar capacidad (`dpcapacity`) según la vocación de uso del corredor. |

---

## 2. Predict (Especificación de Predicción y Evaluación)

### ML Task
| Pregunta | Tipo de tarea | Unidad / Granularidad | Variable Objetivo (Salida) |
|---|---|---|---|
| **P1** | Regresión supervisada | Estación activa $\times$ Día (`station_id`, `date`) | Número de viajes iniciados en el día (`trips_count` $\ge 0$). |
| **P2** | Clasificación multiclase | Estación $\times$ Hora (`station_id`, `hour_ts`) en las **50 estaciones de mayor flujo** | Clase de riesgo en las próximas **3 horas**: `{deficit, balanceado, excedente}` definida por la razón entre el flujo neto futuro de 3h (`arrivals - departures`) y la capacidad de la estación (`capacity`): `deficit` si $\le -0.30$ (`11.13%`), `excedente` si $\ge +0.30$ (`11.92%`), y `balanceado` en $(-0.30, +0.30)$ (`76.95%`). |
| **P3** | Clustering no supervisado | Estación activa (`station_id`) | Grupo o perfil operativo asignado a cada estación ($k = 4$ clusters): **Cluster 0** (Mixto / Recreativo y universitario), **Cluster 1** (Residencial / Emisor matutino), **Cluster 2** (Laboral / Emisor vespertino de recorridos largos) y **Cluster 3** (Nodos centrales de alta rotación). |

### Making Predictions
| Pregunta | Frecuencia de ejecución | Horizonte de predicción | Latencia requerida |
|---|---|---|---|
| **P1** | Cada 24 horas (cierre nocturno) | **1 día** adelante (próximas 24 horas) | Minutos / Horas (batch diario) |
| **P2** | Cada 1 hora | **3 horas** adelante (ventana acumulada $t+1$ a $t+3$) | Minutos (batch horario para despacho de rutas) |
| **P3** | Mensual o trimestral | No aplica (segmentación estructural) | Horas / Días |

### Offline Evaluation y Partición Temporal (Sin Fuga del Futuro)
| Conjunto | Periodo | Propósito |
|---|---|---|
| **Train** | `2024-01-01` a `2025-12-31` (24 meses) | Ajuste de escaladores, codificadores, perfiles base y entrenamiento de modelos (captura 2 ciclos anuales completos). |
| **Validación** | `2026-01-01` a `2026-06-30` (6 meses) | Selección de hiperparámetros y comparación preliminar contra baselines. |
| **Test** | `2026-07-01` a `2026-09-30` (3 meses) | Evaluación final fuera de muestra sobre los meses más recientes. |

| Pregunta | Baseline | Modelo Candidato Inicial | Métricas de Evaluación |
|---|---|---|---|
| **P1** | *Seasonal Naive* (`lag_7d`: viajes del mismo día de la semana anterior) | *Ridge Regression* con variables cíclicas y rezagos (`lag_1d`, `lag_7d`, `lag_14d`, `lag_21d`, `rolling_mean_7d`) | `MAE`, `RMSE`, `WAPE` (calculadas en `val` y `test`) |
| **P2** | Regla histórica semanal (`risk_lag_168h`: misma clase observada hace 7 días a esa hora) | *HistGradientBoostingClassifier* multiclase sobre flujo reciente y presión histórica | `F1-macro`, `Recall-macro`, `Precision-macro` (en `val` y `test`) + Matriz de confusión |
| **P3** | *K-Means Geográfico* ($k=4$ usando únicamente `latitude` y `longitude`) | *K-Means sobre Perfil de Uso* ($k=4$ con proporciones horarias, duración, viajes redondos y razón salidas/llegadas estandarizadas) | `Silhouette Score`, `Davies-Bouldin Index` (en `train`) y estabilidad temporal `ARI` del modelo de uso (comparando 2024–2025 vs. 2026) |

---

## 3. Learn (Datos e Ingeniería de Características)

### Data Sources, Reglas de Negocio y Tratamiento de Datos Personales
- **Viajes mensuales (`datos_abiertos_AAAA_MM.csv`):** Portal de [Datos Abiertos de MiBici](https://mibici.net/es/datos-abiertos/), abarcando **33 meses** desde `2024-01` hasta `2026-09` (**12,539,260** viajes limpios tras quitar `758` duplicados).
- **Nomenclatura de estaciones (`nomenclatura_2026_09.csv`):** **484 estaciones** registradas (**470 activas** con `capacity > 0` y sin prefijo `ex-`, y **14 retiradas** con prefijo `ex-`), distribuidas en tres zonas: `GDL` (357), `ZPN` (110) y `TLQ` (17).
- **Privacidad y condiciones de uso:** Conforme a los lineamientos de datos abiertos y buenas prácticas de MLOps, las columnas de datos personales del usuario (`Usuario_Id`, `Genero` y `Año_de_nacimiento`) se descartan durante la lectura inicial del CSV y nunca se almacenan en los archivos `.parquet` ni se registran en MLflow.
- **Diagnóstico de calidad de datos y reglas operativas ([Términos y Condiciones de MiBici](https://www.mibici.net/es/termino-y-condiciones/)):**
  - **Valores nulos y consistencia mensual:** `0` valores nulos y `0` viajes con fecha fuera del mes del archivo correspondiente.
  - **Duplicados:** `758 (0.006%)` viajes exactamente duplicados (`1,516` filas con `trip_id` repetido, `0.012%`), concentrados únicamente en el archivo `2026-09` (`0` IDs repetidos entre meses distintos).
  - **Duración de viajes:** El tiempo máximo oficial por viaje es de **`8 horas`** y el horario de servicio es de **`5:00 am a 12:59 am`**. En el histórico existen `26,664 (0.213%)` viajes con duración `0 min`, `125,400 (1.000%)` de $\le 11$ s, `340,142 (2.712%)` menores a `1 min` en la **misma estación** de origen y destino (fallo en el anclaje), `22,390 (0.179%)` menores a `1 min` entre estaciones distintas (estaciones cercanas), `367 (0.003%)` mayores a `8 horas` y `139 (0.001%)` fuera del horario de servicio.
  - **Estaciones fuera de catálogo:** `3,954 (0.032%)` viajes hacen referencia a estaciones no presentes en el catálogo (`3,951` en estación `407` y `3` en estación `405`), los cuales quedan excluidos al cruzar contra el catálogo de estaciones activas.

### Collecting Data y Versionado
- El comando reproducible `python -m mi_bici.dataset --start 2024-01 --end 2026-09` descarga automáticamente los CSV, detecta codificación (`utf-8-sig` / `cp1252`), homologa formatos de fecha (`YYYY-MM-DD HH:MM:SS` y `DD/MM/YYYY HH:MM`), guarda las tablas limpias en `data/processed/trips.parquet` y `data/processed/stations.parquet`, y genera el manifiesto versionado en Git `config/data_manifest.json` con conteo de registros y hashes SHA-256 (`hashlib`).

### Features (Sin Fuga Temporal)
- **P1:** `day_of_week`, `month`, `is_weekend`, `is_holiday`, `lag_1d`, `lag_7d`, `lag_14d`, `lag_21d`, `rolling_mean_7d` (calculada sobre `shift(1)`), `capacity` y `zone` (`GDL`, `ZPN`, `TLQ`).
- **P2:** `hour`, `day_of_week`, `is_weekend`, `departures`, `arrivals`, `net_flow`, `pressure_last_3h`, `pressure_window_lag_24h`, `pressure_window_lag_168h`, `pressure_window_lag_336h` y `capacity`; `risk_lag_168h` se usa como baseline.
- **P3:** `dep_morning` (6–10h entre semana), `dep_midday` (11–15h), `dep_evening` (16–20h), `dep_weekend`, `median_duration_min`, `round_trip_pct`, `avg_daily_trips` y `dep_arr_ratio`, ajustando `StandardScaler` exclusivamente sobre el periodo `Train` (2024–2025).

### Sustento de Números, Grupos, Horizontes y Rezagos (Evidencia en Notebooks)
| Parámetro / Grupo | Valor elegido | Evidencia empírica y justificación de negocio |
|---|---|---|
| **Rango y partición temporal** | `2024-01` a `2026-09` (`24m Train / 6m Val / 3m Test`) | `Train` (`2024–2025`) cubre **2 ciclos anuales completos** (capturando estacionalidad de lluvias de verano en Guadalajara, periodos vacacionales y calendario escolar); `Val` (`Ene–Jun 2026`) evalúa un semestre completo; `Test` (`Jul–Sep 2026`) mide desempeño fuera de muestra en el trimestre más reciente (`1.0-ogt-prueba-limpieza-y-parquet.ipynb`). |
| **P1: Horizonte y rezagos** | Horizonte `1d`; lags `1, 7, 14, 21d` y `rolling_mean_7d` | El corte operativo de dotación de flota es diario (`1 día`). El análisis de autocorrelación en `Train` demuestra que la correlación de Pearson con `trips_count` es máxima en múltiplos de 7 días (`lag_7d` $r = 0.9487$, `lag_14d` $r = 0.9417$, `lag_21d` $r = 0.9363$) frente a rezagos entre semana no múltiplos (`lag_2d` $r = 0.8674$, `lag_3d` $r = 0.8528$), porque el uso depende del día de la semana; `3` semanas capturan este ciclo sin perder más de 21 días iniciales, mientras que `lag_1d` ($r = 0.9129$) y `rolling_mean_7d` ($r = 0.9363$) capturan la inercia reciente (`2.0-ogt-prueba-p1-regresion.ipynb`). |
| **P2: Subconjunto de estaciones** | `TOP_N_STATIONS = 50` estaciones activas | Las `50` estaciones de mayor demanda en `Train` (`10.6%` de las estaciones activas) concentran el **`37.15%` de todos los viajes del sistema** (`93.3` viajes/día en promedio vs. `24.6` en el resto), siendo los nodos donde realmente ocurren saturaciones y vaciamientos horarios (`3.0-ogt-prueba-p2-clasificacion.ipynb`). |
| **P2: Horizonte, umbral y rezagos** | Horizonte `3h`; umbral `±0.30`; lags `3h, 24h, 168h, 336h` | Se asume `3 horas` como la ventana para planear y desplazar camionetas de rebalanceo en el tráfico del AMG (MiBici no publica ese dato). Con `THRESHOLD = 0.30` ($\pm 30\%$ de `dpcapacity`, equivalente a $\pm 5$ bicicletas netas en una estación mediana de 15 anclajes), se aísla el **`23.05%` de horas críticas** (`11.13%` `deficit` y `11.92%` `excedente`) frente a `76.95%` `balanceado` (con `0.20` las alertas subirían a `37.44%` saturando al equipo, y con `0.40` caerían a `15.39%`). Los rezagos `last_3h` ($t-2..t$) y las ventanas de hace `24h` (1d), `168h` (7d) y `336h` (14d) capturan la tendencia inmediata y la estacionalidad diaria/semanal (`3.0-ogt-prueba-p2-clasificacion.ipynb`). |
| **P3: Filtro de duración y franjas** | `1 a 180 min`; franjas `6–10h`, `11–15h`, `16–20h`, `fin de semana` | El corte `< 1 min` descarta los `340,142 (2.712%)` falsos inicios por fallo de anclaje en la misma estación documentados en limpieza, y `<= 180 min` acota viajes atípicos largos. Las 3 franjas entre semana capturan las olas de movilidad del AMG: entrada matutina (`6–10h`), mediodía/comida (`11–15h`) y salida vespertina (`16–20h`) (`4.0-ogt-prueba-p3-clustering.ipynb`). |
| **P3: Número de grupos y perfiles** | `BEST_K = 4` clusters | Al evaluar $k \in \{3, 4, 5, 6\}$, el paso de $k=3$ a **$k=4$** logra la mayor reducción relativa del índice Davies-Bouldin (de `1.7999` a `1.5849`, **`-11.94%`**, con Silueta `0.1810` vs. `0.0735` del baseline geográfico y estabilidad `ARI = 0.4665` en MLflow), identificando 4 perfiles urbanos accionables en `Train` (`371` estaciones):<br>- **Cluster 0 (Mixto / Recreativo y universitario, $n=36$):** Mayor proporción al mediodía (`28.4%`) y mayor tasa de viajes redondos (`7.6%`, duración mediana `9.90 min`).<br>- **Cluster 1 (Residencial / Emisor matutino, $n=147$):** Pico matutino de salida (`29.8%` entre 6–10h) y alta participación en fin de semana (`21.3%`).<br>- **Cluster 2 (Laboral / Emisor vespertino de recorridos largos, $n=96$):** Pico vespertino (`38.2%` entre 16–20h), mayor duración mediana (`12.39 min`) y razón salidas/llegadas de `1.34`.<br>- **Cluster 3 (Nodos centrales de alta rotación, $n=92$):** Demanda intensiva de **`67.4` viajes/día** por estación con flujo balanceado (`dep_arr_ratio = 1.03`) (`4.0-ogt-prueba-p3-clustering.ipynb`). |

---

## 4. Evaluate (Monitoreo y Operación Continua)

| Pregunta | Métrica a monitorear en producción | Criterio de reentrenamiento |
|---|---|---|
| **P1** | `WAPE` semanal/mensual contra viajes reales observados | Mensual al publicarse cada nuevo CSV o si `WAPE` sube más de 5 puntos porcentuales. |
| **P2** | `Recall-macro` y `F1-macro` sobre estados de `deficit` y `excedente` | Mensual o ante cambios en la capacidad de estaciones (`dpcapacity`). |
| **P3** | Estabilidad de asignación de clusters entre periodos (`Adjusted Rand Index - ARI`) | Trimestral o cuando se inauguren/retiren estaciones en el AMG. |
