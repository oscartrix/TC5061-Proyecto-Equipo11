# mi-bici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOps de MiBici proyecto de equipo

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mi_bici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mi_bici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mi_bici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Flujo de Trabajo y Contribución (Git & Pull Requests)

Para mantener la estabilidad del proyecto, la rama `main` está protegida. Todos los cambios deben integrarse a través de un **Pull Request (PR)** revisado y aprobado.

### 1. Actualizar tu rama local
Antes de comenzar cualquier trabajo nuevo, asegúrate de tener la versión más reciente de `main`:
```bash
git checkout main
git pull origin main
```

### 2. Crear una nueva rama
Crea una rama descriptiva para tu tarea utilizando las siguientes convenciones:
- `feature/<nombre>`: Nuevas funcionalidades, modelos o pipelines (ej. `feature/data-pipeline`).
- `fix/<nombre>`: Corrección de errores (ej. `fix/missing-values`).
- `docs/<nombre>`: Mejoras en documentación (ej. `docs/update-readme`).
- `refactor/<nombre>`: Reestructuración de código sin cambio funcional.

```bash
git checkout -b feature/nombre-de-tu-tarea
```

### 3. Trabajar y confirmar cambios
Realiza tus cambios y realiza commits claros y atómicos (se sugiere formato convencional: `feat:`, `fix:`, `docs:`, etc.):
```bash
git add .
git commit -m "docs: agregar guía de contribución y flujo de trabajo"
```

### 4. Subir la rama a GitHub
Publica tu rama en el repositorio remoto:
```bash
git push -u origin feature/nombre-de-tu-tarea
```

### 5. Abrir el Pull Request (PR)
1. Ve al repositorio en GitHub: [TC5061-Proyecto-Equipo11](https://github.com/oscartrix/TC5061-Proyecto-Equipo11).
2. Verás un botón amarillo que dice **"Compare & pull request"**; haz clic en él (o ve a la pestaña **Pull requests** -> **New pull request**).
3. Asegúrate de que la rama base sea `main` y la rama de comparación sea tu rama.
4. Agrega una descripción clara de lo que se realizó.
5. Asigna al menos a un compañero de equipo como revisor (**Reviewers**).

### 6. Revisión y Merge
- Un compañero debe revisar los cambios y seleccionar **"Approve"**.
- Una vez aprobado, se realiza el **Merge pull request** hacia `main`.
- Finalmente, actualiza tu repo local y borra la rama que ya fue integrada:
  ```bash
  git checkout main
  git pull origin main
  git branch -d feature/nombre-de-tu-tarea
  ```
