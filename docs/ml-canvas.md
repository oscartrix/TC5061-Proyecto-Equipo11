# ML Canvas: MiBici

---
## PENDIENTES
- Definir número de horas para predicción de déficit y excedente.
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

### ML task
| | Tipo | Entrada | Salida |
|---|---|---|---|
| P1 | Regresión | Estación, fecha, historial de viajes | Viajes iniciados por día. Entero ≥ 0. |
| P2 | Clasificación multiclase | Estación, hora, flujo reciente, capacidad | {`deficit`, `excedente`, `balanceado`} en las próximas [PENDIENTE] horas. |
| P3 | Clustering | Perfil de uso por estación | Grupo asignado a cada estación |

### Making predictions

### Offline evaluation

## Learn

### Data sources

### Collecting data

### Features

### Building models

## Evaluate

### Live evaluation and monitoring

## Formulación por pregunta
