# ML Canvas: MiBici

---
## PENDIENTES
- Definir número de horas para predicción de déficit y excedente.
- Definir número de días para predecir reabastecimiento.
---

| # | Pregunta de negocio | Tipo de tarea | Quién la usaría |
|---|---|---|---|
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Operador del sistema, para planear la flota. |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | Clasificación | Equipo de rebalanceo de bicicletas. |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | *Clustering* | Planeación urbana y expansión del sistema. |

## Value Proposition
### Qué
Anticipar la demanda y el riesgo de que las estaciones se vacíen o se llenen así como identificar patrones de uso del sistema.

### Porqué
- **P1:** una flotilla de bicicletas mal distribuida tiene bicicletas sin usar y viajes que no pueden realizar los usuarios. Predecir cuántos viajes saldrán de cada estación permite asignar una flotilla óptima por estación o zona para prevenir estos problemas.
- **P2:** una estación vacía o llena es una falla hacia el usuario ya que no puede usar una bicicleta o no puede anclarla. Anticiparlo permite usar mejor el equipo de rebalanceo, suponiendo que le quipo de rebalanceo es limitado.
- **P3:** hacer una planeación basada en datos y evidencia para expandir el sistema MiBici. No tomar decisiones basados en intuiciones.

### Quién
| Pregunta | Usuario |
|---|---|
| P1 | Operador del sistema, para planear la flota |
| P2 | Equipo de rebalanceo de bicicletas |
| P3 | Planeación urbana y expansión del sistema |

Además, el usuario de MiBici se verá beneficiado con las acciones que se tomen en base a los resultados de los modelos.

## Predict

### Decisions
- **P1:** asignar la cantidad óptima de bicicletas por estación para cada rebastecimiento. 
- **P2:** ordenar la ruta de rebalanceo. Llevar bicicletas a las estaciones con `deficit` y retirarlas de las estaciones con `excedente`, atendiendo primero las de mayor probabilidad.
- **P3:** elegir zonas para expandir el sistema o densificar estaciones.

### ML task
| | Tipo | Entrada | Salida |
|---|---|---|---|
| P1 | Regresión | Estación, fecha, historial de viajes | Viajes iniciados por día. Entero ≥ 0. |
| P2 | Clasificación multiclase | Estación, hora, flujo reciente, capacidad | {`deficit`, `excedente`, `balanceado`} en las próximas [PENDIENTE] horas. |
| P3 | Clustering | Perfil de uso por estación | Grupo asignado a cada estación |

### Making predictions
| | Cuándo | Horizonte | Tiempo disponible |
|---|---|---|---|
| P1 | Cada 24 horas | [PENDIENTE] días | Horas |
| P2 | Cada hora | [PENDIENTE] horas | Minutos |
| P3 | Mensual o trimestral | No aplica | Días |

### Offline evaluation
| Conjunto | Periodo |
|---|---|
| Train | Desde Ene 2024 a Dic 2025 |
| Validación | Desde Ene 2026 a Jun 2026 |
| Test | Desde Jul 2026 a Sep 2026 |

| | Baseline | Métricas |
|---|---|---|
| P1 | Seasonal naive (mismo día de la semana anterior) | MAE, RMSE, WAPE |
| P2 | Regla: misma clase que a esta hora hace 7 días | PR-AUC, recall, precision@k |
| P3 | K-Means geográfico (latitud y longitud, mismo k que el modelo) | Silueta, Davies-Bouldin, estabilidad (ARI) |

## Learn

### Data sources
- Viajes mensuales en formato CSV del [portal de datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/), rango 2024-01 a 2026-09.
- [Nomenclatura de estaciones](https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv): id, nombre, latitud, longitud y capacidad.
- Calendario de días festivos de México.

### Collecting data
- MiBici publica un CSV por mes. La ingesta descarga el mes nuevo, lo estandariza y lo versiona con DVC.
- Las etiquetas para P1 y P2 se pueden obtener de los datos, no se requiere etiquetado manual.
- No hay datos para calcular y predecir los viajes perdidos cuando una estación esta vacía o el impacto al usuario cuando una estación esta llena

### Features
- **P1:** día de la semana, mes, festivos, lags de 7, 14 y 21 días, capacidad, zona.
- **P2:** hora, día de la semana, salidas y llegadas de la última hora, flujo neto de la misma hora hace 1 y 7 días, capacidad.
- **P3:** perfil horario normalizado de salidas y llegadas (entre semana y fin de semana), razón salidas/llegadas, volumen diario, duración media, % de viajes redondos.

### Building models
- **P1 y P2:** reentrenamiento mensual cuando se publica el CSV nuevo.
- **P3:** recálculo trimestral o cuando se abren/cierran estaciones nuevas.
- Cada entrenamiento se registra en MLflow con rango de fechas, versión de datos y commit de Git.

## Evaluate

### Live evaluation and monitoring
| | Métrica en producción |
|---|---|
| P1 | WAPE mensual contra viajes reales |
| P2 | Precision@k por día |
| P3 | Cambio en la asignación de grupos entre recálculos (ARI) |

**Drift a monitorear:** volumen por estación, cambios de esquema en los CSV, estaciones nuevas o retiradas.
