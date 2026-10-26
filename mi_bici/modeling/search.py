"""Búsqueda de hiperparámetros con MLflow nested runs.

Este módulo reutiliza las funciones de entrenamiento existentes
para registrar una corrida padre y una hija por combinación.
"""

import mlflow
import os
from mi_bici.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, SEARCH_GRIDS
from mi_bici.modeling.train import run_p1, run_p2, run_p3
from itertools import product
from mi_bici.modeling.train import RUNNERS

def search_p1_ridge():
    """Ejecuta una búsqueda de alpha para Ridge con nested runs."""

    alpha_values = SEARCH_GRIDS["p1"]["ridge"]["alpha"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    #mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_alpha = None
    best_mae = float("inf")
    best_run_id = None

    # Corrida padre
    project_run_id = os.environ.get("MLFLOW_RUN_ID")
    with mlflow.start_run(
        run_id=project_run_id,
        run_name="p1-search-ridge" if project_run_id is None else None,
    ) as parent_run:
        # Evita que las corridas hijas reutilicen el ID del proyecto.
        os.environ.pop("MLFLOW_RUN_ID", None)
        mlflow.set_tags({
            "question": "p1",
            "model_family": "RidgeRegression",
            "stage_of_work": "search",
        })

        for alpha in alpha_values:

            # Corrida hija
            with mlflow.start_run(
                run_name=f"p1-ridge-alpha-{alpha}",
                nested=True,
            ) as child_run:

                mlflow.log_param("model", "ridge")
                mlflow.log_param("alpha", alpha)

                run_p1("ridge", {"alpha": alpha})

                # Consultar la métrica registrada por run_p1
                metrics = mlflow.get_run(child_run.info.run_id).data.metrics
                current_mae = metrics["val_mae"]

                if current_mae < best_mae:
                    best_mae = current_mae
                    best_alpha = alpha
                    best_run_id = child_run.info.run_id

        # Registrar la mejor configuración en la corrida padre
        mlflow.log_param("best_alpha", best_alpha)
        mlflow.log_metric("best_val_mae", best_mae)
        mlflow.set_tag("best_child_run_id", best_run_id)

    print(f"Mejor alpha: {best_alpha}")
    print(f"Mejor MAE de validación: {best_mae:.4f}")


def generate_combinations(grid: dict) -> list[dict]:
    """Genera todas las combinaciones de un grid de hiperparámetros."""
    names = list(grid.keys())
    values = list(grid.values())

    return [
        dict(zip(names, combination))
        for combination in product(*values)
    ]

def search_hyperparameters(question: str, model: str) -> None:
    """Ejecuta una búsqueda de hiperparámetros con MLflow nested runs."""

    if question not in SEARCH_GRIDS or model not in SEARCH_GRIDS[question]:
        raise ValueError(f"No existe un grid para {question}/{model}")

    grid = SEARCH_GRIDS[question][model]
    combinations = generate_combinations(grid)

    # Métrica y dirección de optimización por pregunta.
    objectives = {
        "p1": ("val_mae", "min"),
        "p2": ("val_f1_macro", "max"),
        "p3": ("holdout_ari_stability", "max"),
    }

    metric_name, direction = objectives[question]
    best_score = float("inf") if direction == "min" else float("-inf")
    best_params = None
    best_run_id = None

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    project_run_id = os.environ.get("MLFLOW_RUN_ID")

    # Reutilizar la corrida creada por MLflow Projects.
    with mlflow.start_run(run_id=project_run_id) as parent_run:
        os.environ.pop("MLFLOW_RUN_ID", None)

        mlflow.set_tag("mlflow.runName", f"{question}-search-{model}")
        mlflow.set_tags({
            "question": question,
            "model_family": model,
            "stage_of_work": "search",
        })

        mlflow.log_param("search_model", model)
        mlflow.log_param("n_combinations", len(combinations))
        mlflow.log_param("objective_metric", metric_name)

        for index, hparams in enumerate(combinations, start=1):
            run_name = f"{question}-{model}-trial-{index}"

            with mlflow.start_run(
                run_name=run_name,
                nested=True,
            ) as child_run:

                mlflow.set_tags({
                    "question": question,
                    "model_family": model,
                    "stage_of_work": "candidate",
                })

                mlflow.log_params(hparams)

                # Reutilizar el entrenamiento existente.
                RUNNERS[question](model, hparams)

                metrics = mlflow.get_run(child_run.info.run_id).data.metrics

                if metric_name not in metrics:
                    raise ValueError(
                        f"La corrida {run_name} no registró {metric_name}"
                    )

                score = metrics[metric_name]

                is_better = is_better_score(score, best_score, direction)

                if is_better:
                    best_score = score
                    best_params = hparams.copy()
                    best_run_id = child_run.info.run_id

        # Resumen de la mejor configuración en la corrida padre.
        mlflow.log_metric(f"best_{metric_name}", best_score)
        mlflow.log_dict(best_params, "best_hyperparameters.json")
        mlflow.set_tag("best_child_run_id", best_run_id)

    print(f"Mejor configuración para {question}/{model}: {best_params}")
    print(f"Mejor {metric_name}: {best_score:.4f}")

def is_better_score(score: float, best_score: float, direction: str) -> bool:
    """Determina si una métrica mejora el mejor resultado actual."""
    if direction == "min":
        return score < best_score
    if direction == "max":
        return score > best_score
    raise ValueError(f"Dirección de optimización inválida: {direction}")

import typer

app = typer.Typer()


@app.command()
def main(
    question: str = typer.Option("p1"),
    model: str = typer.Option("ridge"),
):
    """Ejecuta una búsqueda de hiperparámetros."""

    if question in SEARCH_GRIDS and model in SEARCH_GRIDS[question]:
        search_hyperparameters(question, model)
    else:
        raise typer.BadParameter(
            f"Búsqueda no disponible para question={question}, model={model}"
        )


if __name__ == "__main__":
    app()
