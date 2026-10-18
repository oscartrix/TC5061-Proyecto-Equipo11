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
| **P1** | Regresión supervisada | Estación $\times$ Día (`station_id`, `date`) | Número de viajes iniciados en el día (`trips_count` $\ge 0$). |
| **P2** | Clasificación multiclase | Estación $\times$ Hora (`station_id`, `hour_ts`) | Clase de riesgo en las próximas **3 horas**: `{deficit, balanceado, excedente}` definida por la razón entre el flujo neto futuro de 3h (`arrivals - departures`) y la capacidad de la estación (`capacity`): `deficit` si $\le -0.30$, `excedente` si $\ge +0.30$, y `balanceado` en $(-0.30, +0.30)$. |
| **P3** | Clustering no supervisado | Estación activa (`station_id`) | Grupo o perfil operativo asignado a cada estación ($k = 4$ clusters). |

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
| **P2** | Regla histórica semanal (`risk_lag_168h`: misma clase observada hace 7 días a esa hora) | *Logistic Regression* multiclase con pesos balanceados (`class_weight='balanced'`) | `F1-macro`, `Recall-macro`, `Precision-macro` (en `val` y `test`) + Matriz de confusión |
| **P3** | *K-Means Geográfico* ($k=4$ usando únicamente `latitude` y `longitude`) | *K-Means sobre Perfil de Uso* ($k=4$ con proporciones horarias, duración, viajes redondos y razón salidas/llegadas estandarizadas) | `Silhouette Score`, `Davies-Bouldin Index` (en `train`) y estabilidad temporal `ARI` (comparando 2024–2025 vs. 2026) |

---

## 3. Learn (Datos e Ingeniería de Características)

### Data Sources y Tratamiento de Datos Personales
- **Viajes mensuales (`datos_abiertos_AAAA_MM.csv`):** Portal de [Datos Abiertos de MiBici](https://mibici.net/es/datos-abiertos/), abarcando **33 meses** desde `2024-01` hasta `2026-09`.
- **Nomenclatura de estaciones (`nomenclatura_2026_09.csv`):** 484 estaciones registradas (470 activas y 14 retiradas con prefijo `ex-`), con identificador, nombre, latitud, longitud y capacidad (`dpcapacity`).
- **Privacidad y condiciones de uso:** Conforme a los lineamientos de datos abiertos y buenas prácticas de MLOps, las columnas de datos personales del usuario (`Usuario_Id`, `Genero` y `Año_de_nacimiento`) se descartan durante la lectura inicial del CSV y nunca se almacenan en los archivos `.parquet` ni se registran en MLflow.

### Collecting Data y Versionado
- El comando reproducible `python -m mi_bici.dataset --start 2024-01 --end 2026-09` descarga automáticamente los CSV, detecta codificación (`utf-8-sig` / `cp1252`), homologa formatos de fecha (`YYYY-MM-DD HH:MM:SS` y `DD/MM/YYYY HH:MM`), guarda las tablas limpias en `data/processed/trips.parquet` y `data/processed/stations.parquet`, y genera el manifiesto versionado en Git `config/data_manifest.json` con conteo de registros y hashes SHA-256 (`hashlib`).

### Features (Sin Fuga Temporal)
- **P1:** `day_of_week`, `month`, `is_weekend`, `is_holiday`, `lag_1d`, `lag_7d`, `lag_14d`, `lag_21d`, `rolling_mean_7d` (calculada sobre `shift(1)`), `capacity` y `zone` (`GDL`, `ZPN`, `TLQ`).
- **P2:** `hour`, `day_of_week`, `is_weekend`, `departures`, `arrivals`, `net_flow`, `net_flow_last_3h`, `net_flow_lag_24h`, `net_flow_lag_168h`, `capacity` y `risk_lag_168h`.
- **P3:** `dep_morning` (6–10h entre semana), `dep_midday` (11–15h), `dep_evening` (16–20h), `dep_weekend`, `median_duration_min`, `round_trip_pct`, `avg_daily_trips` y `dep_arr_ratio`, ajustando `StandardScaler` exclusivamente sobre el periodo `Train` (2024–2025).

---

## 4. Evaluate (Monitoreo y Operación Continua)

| Pregunta | Métrica a monitorear en producción | Criterio de reentrenamiento |
|---|---|---|
| **P1** | `WAPE` semanal/mensual contra viajes reales observados | Mensual al publicarse cada nuevo CSV o si `WAPE` sube más de 5 puntos porcentuales. |
| **P2** | `Recall-macro` y `F1-macro` sobre estados de `deficit` y `excedente` | Mensual o ante cambios en la capacidad de estaciones (`dpcapacity`). |
| **P3** | Estabilidad de asignación de clusters entre periodos (`Adjusted Rand Index - ARI`) | Trimestral o cuando se inauguren/retiren estaciones en el AMG. |
