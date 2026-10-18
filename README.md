# Proyecto Integrador MLOps — MiBici (Equipo 11)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Propuesta técnica, pipeline reproducible de ingesta y preprocesamiento, módulo de ingeniería de características y experimento base en **MLflow Tracking** para el sistema de bicicleta pública **MiBici** del Área Metropolitana de Guadalajara (AMG).

---

## 1. Propósito del Proyecto y Preguntas de Negocio

El objetivo del proyecto es diseñar y construir una plataforma MLOps reproducible sobre millones de registros históricos de viajes de MiBici para apoyar la toma de decisiones operativas y de planeación urbana mediante tres tareas complementarias de Machine Learning:

| # | Pregunta de negocio | Tipo de tarea | Granularidad y Horizonte | Usuario y Decisión |
|---|---|---|---|---|
| **P1** | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | **Regresión** | Diaria por estación (`station_id`, `date`), horizonte de **1 día** (24 h). | **Operador del sistema:** Planear la dotación diaria de bicicletas y el mantenimiento de flota. |
| **P2** | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | **Clasificación multiclase** | Horaria por estación (`station_id`, `hour_ts`), horizonte de **3 horas** (`deficit`, `balanceado`, `excedente`). | **Equipo de rebalanceo:** Priorizar cada hora las rutas de camiones hacia estaciones con mayor riesgo. |
| **P3** | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | **Clustering** | Por estación activa (`station_id`), recalibración trimestral ($k=4$). | **Planeación urbana:** Decidir la expansión de estaciones y ampliación de capacidad (`dpcapacity`). |

El detalle completo de la formulación se encuentra en [`references/ml-canvas.md`](references/ml-canvas.md) y la arquitectura del sistema en [`references/architecture/architecture.md`](references/architecture/architecture.md).

---

## 2. Datos Utilizados, Privacidad y Partición Temporal

- **Fuente oficial:** [Portal de Datos Abiertos de MiBici](https://mibici.net/es/datos-abiertos/) (URLs declaradas en [`config/sources.yaml`](config/sources.yaml)).
- **Rango seleccionado:** `2024-01` a `2026-09` (**33 meses** completos de operación más el catálogo de estaciones `nomenclatura_2026_09.csv` con **484 estaciones**, de las cuales 470 están activas y 14 tienen prefijo `ex-`).
- **Justificación del rango:** Cubre dos ciclos anuales completos (2024 y 2025) para capturar la estacionalidad climática (temporada de lluvias en verano) y académica del AMG en entrenamiento, dejando 9 meses de 2026 para validación y prueba.
- **Protección de datos personales:** Las columnas `Usuario_Id`, `Genero` y `Año_de_nacimiento` se excluyen directamente durante la lectura del CSV en [`mi_bici/data/readers.py`](mi_bici/data/readers.py), por lo que nunca se almacenan en Parquet ni se registran en MLflow.
- **Partición temporal estricta (sin fuga del futuro):**
  - **Train:** `2024-01-01` a `2025-12-31` (24 meses)
  - **Validación:** `2026-01-01` a `2026-06-30` (6 meses)
  - **Test:** `2026-07-01` a `2026-09-30` (3 meses)

---

## 3. Estructura del Repositorio

```text
├── LICENSE                  <- Licencia del proyecto
├── Makefile                 <- Comandos de utilidad (make data, make test, make lint)
├── MLproject                <- Definición de entry points reproducibles para MLflow Projects
├── README.md                <- Documentación principal del proyecto
├── pyproject.toml           <- Metadatos del paquete mi_bici y declaración de dependencias
├── requirements.txt         <- Dependencias del pipeline con versiones fijas
├── requirements-dev.txt     <- Dependencias de desarrollo (Jupyter, pytest, ruff) con versiones fijas
├── config/
│   ├── sources.yaml         <- Catálogo declarativo de URLs de viajes (2024-01 a 2026-09) y estaciones
│   └── data_manifest.json   <- Manifiesto versionado con hashes SHA-256 (hashlib) y conteos de filas
├── data/
│   ├── raw/                 <- CSVs originales descargados (excluidos de Git)
│   ├── interim/             <- Datos intermedios (excluidos de Git)
│   └── processed/           <- Archivos Parquet limpios y tablas de features (excluidos de Git)
├── models/                  <- Modelos serializados (excluidos de Git)
├── notebooks/
│   ├── 01-jcc-data-ingestion.ipynb                    <- Exploración de ingesta, encodings y esquemas
│   └── pruebas/                                       <- Notebooks de experimentación y validación por caso
│       ├── 1.0-ogt-prueba-limpieza-y-parquet.ipynb    <- Prueba de limpieza, Parquet y manifiesto
│       ├── 2.0-ogt-prueba-p1-regresion.ipynb          <- Experimentación de P1 (Regresión diaria)
│       ├── 3.0-ogt-prueba-p2-clasificacion.ipynb      <- Experimentación de P2 (Clasificación horaria)
│       └── 4.0-ogt-prueba-p3-clustering.ipynb         <- Experimentación de P3 (Clustering de uso)
├── references/
│   ├── ml-canvas.md         <- Machine Learning Canvas completo para P1, P2 y P3
│   └── architecture/        <- Diagramas de arquitectura (Markdown/Mermaid, HTML interactivo y PNG)
├── reports/
│   ├── propuesta_equipo11.md <- Documento integrador de la propuesta y resultados con APA 7
│   └── figures/             <- Gráficas generadas y registradas como artefactos en MLflow
├── tests/
│   └── test_data.py         <- Pruebas unitarias de validación de esquemas y fechas
└── mi_bici/                 <- Módulo principal de código fuente
    ├── __init__.py
    ├── config.py            <- Configuración centralizada (rutas, fechas, semilla=42, MLflow)
    ├── dataset.py           <- CLI de ingesta, escritura a Parquet y manifiesto SHA-256
    ├── features.py          <- Funciones documentadas de feature engineering para P1, P2 y P3
    ├── plots.py             <- Utilidades de visualización
    ├── data/
    │   ├── __init__.py
    │   ├── schemas.py       <- Contratos de columnas, variantes de headers y tipos
    │   ├── sources.py       <- Descarga HTTP con reintentos y soporte de archivos parciales (.part)
    │   └── readers.py       <- Detección de encoding, normalización de fechas y exclusión de PII
    └── modeling/
        ├── __init__.py
        ├── train.py         <- Ejecución y registro en MLflow de baselines y modelos candidatos
        └── predict.py       <- Módulo base para inferencia
```

---

## 4. Requisitos e Instalación Reproducible

- Python 3.10
- [uv](https://docs.astral.sh/uv/) para la gestión reproducible del entorno virtual

Desde un clon limpio del repositorio:

```bash
uv venv --python 3.10 --seed .venv
source .venv/bin/activate
uv pip sync requirements-dev.txt
pip install -e . --no-deps
nbstripout --install --attributes .gitattributes
```

---

## 5. Ejecución del Pipeline (Ingesta, Features y Experimento Base en MLflow)

### Paso 1: Ingesta reproducible, conversión a Parquet y versionado (`dataset.py`)
Descarga los 33 archivos mensuales y el catálogo de estaciones en `data/raw/`, normaliza columnas/codificación/fechas eliminando datos personales, guarda `data/processed/trips.parquet` y `data/processed/stations.parquet`, y actualiza `config/data_manifest.json` con el hash SHA-256 (`data_version`):

```bash
python -m mi_bici.dataset --start 2024-01 --end 2026-09
```

### Paso 2: Construcción de características sin fuga temporal (`features.py`)
Genera los conjuntos procesados `p1_features.parquet`, `p2_features.parquet`, `p3_train_features.parquet` y `p3_test_features.parquet`:

```bash
python -m mi_bici.features
```

### Paso 3:Ejecución del experimento base en MLflow (`modeling/train.py`)
Ejecuta los baselines y modelos candidatos para **P1**, **P2** y **P3**, registrando cada ejecución en el experimento **`mibici-equipo11`** con parámetros (`start_month`, `end_month`, `data_version`, `random_seed`), etiquetas (`team`, `question`, `git_commit`), métricas con prefijo (`val_*`, `test_*`) y artefactos gráficos en `reports/figures/`:

```bash
python -m mi_bici.modeling.train
```

Para abrir la interfaz web de MLflow y revisar o comparar los runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Paso 4: Ejecutar pruebas unitarias
```bash
pytest
```

---

## 6. Buenas Prácticas Obligatorias Aplicadas

| Ámbito | Buenas prácticas implementadas | Dónde se verifica |
|---|---|---|
| **MLflow Tracking** | 1. Experimento único `mibici-equipo11` y runs con nombres descriptivos (`p1-baseline-seasonal-naive`, `p1-candidate-ridge-regression`, `p2-baseline-weekly-rule`, `p2-candidate-logistic-regression`, `p3-baseline-kmeans-geo`, `p3-candidate-kmeans-usage`).<br>2. Parámetros de reproducción completos: rango de fechas, `data_version` (SHA-256 del manifiesto), `random_seed=42` e hiperparámetros.<br>3. Etiquetas (`tags`) con `team=equipo11`, `question=P1|P2|P3` y `git_commit`.<br>4. Métricas con prefijo del conjunto (`val_mae`, `test_mae`, `val_f1_macro`, `test_f1_macro`, `train_silhouette_usage`, `test_ari_stability`).<br>5. Sin credenciales ni datos personales registrados. | [`mi_bici/modeling/train.py`](mi_bici/modeling/train.py) e Interfaz de MLflow |
| **Estructura y Código** | 1. Notebooks usados únicamente para exploración y pruebas en `notebooks/` y `notebooks/pruebas/`; la lógica reproducible vive en el paquete `mi_bici/`.<br>2. Configuración centralizada en un único archivo (`mi_bici/config.py`) sin repetir rutas, fechas ni semillas.<br>3. Dependencias con versión fija en `requirements.txt` y `requirements-dev.txt`. | [`mi_bici/config.py`](mi_bici/config.py), [`requirements.txt`](requirements.txt) |
| **Datos y Git** | 1. `.gitignore` excluye `data/raw/`, `data/processed/`, `*.parquet`, `mlruns/`, `mlartifacts/`, `mlflow.db`, `models/` y `.env`.<br>2. Rama `main` protegida y flujo de trabajo mediante Pull Requests revisados por el equipo. | [`.gitignore`](.gitignore) e historial de PRs en GitHub |

---

## 7. Flujo de Trabajo y Contribución (Git & Pull Requests)

Para mantener la estabilidad del proyecto, la rama `main` está protegida. Todos los cambios se integran a través de un **Pull Request (PR)** revisado y aprobado por otro integrante del equipo siguiendo **GitHub Flow** y **Conventional Commits** (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`):

1. Actualizar rama local:
   ```bash
   git checkout main && git pull origin main
   ```
2. Crear una rama descriptiva (`feat/<nombre>`, `docs/<nombre>`, `chore/<nombre>`):
   ```bash
   git checkout -b feat/nombre-de-tu-tarea
   ```
3. Confirmar cambios atómicos y subir la rama:
   ```bash
   git add .
   git commit -m "feat: descripción breve del cambio"
   git push -u origin feat/nombre-de-tu-tarea
   ```
4. Abrir el Pull Request con GitHub CLI (`gh pr create`) o desde la interfaz web de GitHub y solicitar la revisión de al menos un compañero antes de hacer el merge.

---

## 8. Integrantes del Equipo 11 y Roles

| Integrante | Usuario GitHub | Rol Principal en esta Entrega |
|---|---|---|
| **Oscar** | `@oscartrix` | Arquitectura MLOps, experimentación en notebooks de pruebas, módulo `features.py`, tracking en MLflow (`train.py`) y guía de flujo Git. |
| **Jesús (Chuy)** | `@chuycc` | Diseño e implementación de ingesta reproducible (`sources.py`, `readers.py`, `schemas.py`, `dataset.py`), notebook de ingesta y formulación inicial de ML Canvas. |
| **Jimena** | `@JimenaMaSuMu` | Configuración de seguridad y exclusión de artefactos (`.gitignore`), revisión de contratos de datos y validación de PRs. |
| **Luis Antonio Cárdenas** | `@luis-antonio-cardenas` | Diseño de arquitectura del sistema, generación de diagramas estáticos (Mermaid/PNG) e interactivos (`architecture.html`) y revisión técnica de PRs. |
