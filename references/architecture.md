# Arquitectura actual

Este diagrama describe el flujo que está implementado en el repositorio. Distingue la ingesta ejecutable de los módulos de procesamiento y modelado que siguen siendo plantillas.

```mermaid
flowchart LR
    operator["Usuario / equipo"] -->|python -m mi_bici.dataset<br/>--start --end| cli["CLI de ingesta<br/>Typer · mi_bici/dataset.py"]

    subgraph repository["Repositorio"]
        urls["config/sources.yaml<br/>URLs mensuales y catálogo"]
        sources["mi_bici.data.sources<br/>carga config · selecciona fechas<br/>descarga con requests"]
        contracts["mi_bici.data.schemas<br/>contratos y variantes CSV"]
        readers["mi_bici.data.readers<br/>encoding · encabezados · pandas"]
        paths["mi_bici.config<br/>rutas del proyecto"]
        trips["data/raw/trips/*.csv"]
        stations["data/raw/stations/*.csv"]
        cli --> sources
        urls --> sources
        paths --> cli
        paths --> sources
        sources -->|guarda archivos| trips
        sources -->|guarda archivo| stations
        cli -->|compara encabezados| readers
        cli -->|lee y valida catálogo| readers
        readers --> contracts
        trips -->|solo encabezados en el CLI| readers
        stations -->|CSV completo| readers
    end

    sources -->|HTTP GET| mibici["Portal de datos abiertos MiBici"]
    mibici -->|CSV de respuesta| sources

    notebook["notebooks/01-jcc-data-ingestion.ipynb<br/>exploración interactiva"] -.->|usa helpers y revisa muestras| readers
    notebook -.-> sources

    subgraph scaffold["Plantillas sin flujo conectado"]
        features["mi_bici/features.py"]
        train["mi_bici/modeling/train.py"]
        predict["mi_bici/modeling/predict.py"]
        plots["mi_bici/plots.py"]
        targets["data/interim · data/processed<br/>models · reports/figures"]
        features -.->|sin pipeline implementado| targets
        train -.->|sin entrenamiento implementado| targets
        predict -.->|sin inferencia implementada| targets
        plots -.->|sin generación implementada| targets
    end
```

## Componentes y límites

- `mi_bici.dataset` valida el rango de meses, selecciona URLs, descarga los archivos, carga y valida el catálogo de estaciones y comprueba encabezados de viajes.
- `mi_bici.data.sources` obtiene las URLs de `config/sources.yaml` y guarda los CSV originales bajo `data/raw/`.
- `mi_bici.data.readers` detecta codificaciones y valida contratos definidos en `mi_bici.data.schemas`. El comando CLI revisa los encabezados de viajes, pero no carga ni transforma sus filas. `read_trips` existe como helper, pero no forma parte de ese comando.
- El notebook documenta exploración de los datos y reutiliza helpers del paquete; no es un servicio desplegado.
- `features.py`, `modeling/train.py`, `modeling/predict.py` y `plots.py` contienen código de plantilla. Las rutas de datos procesados, modelos y reportes están configuradas, pero el flujo actual no genera esos artefactos.

## Referencias de implementación

- [Comando de ingesta](../../mi_bici/dataset.py)
- [Descarga y selección de fuentes](../../mi_bici/data/sources.py)
- [Lectura y validación de CSV](../../mi_bici/data/readers.py)
- [Contratos de datos](../../mi_bici/data/schemas.py)
- [Rutas del proyecto](../../mi_bici/config.py)
- [Fuentes de datos](../../config/sources.yaml)
- [Notebook de exploración](../../notebooks/01-jcc-data-ingestion.ipynb)