"""Búsqueda de hiperparámetros con MLflow nested runs.

Este módulo reutiliza las funciones de entrenamiento existentes
para registrar una corrida padre y una hija por combinación.
"""

import mlflow
import os
from mi_bici.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from mi_bici.modeling.train import run_p1, run_p2, run_p3


def search_p1_ridge():
    """Ejecuta una búsqueda de alpha para Ridge con nested runs."""

    alpha_values = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

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

import typer

app = typer.Typer()


@app.command()
def main(
    question: str = typer.Option("p1"),
    model: str = typer.Option("ridge"),
):
    """Ejecuta una búsqueda de hiperparámetros."""

    if question == "p1" and model == "ridge":
        search_p1_ridge()
    else:
        raise typer.BadParameter(
            f"Búsqueda no disponible para question={question}, model={model}"
        )


if __name__ == "__main__":
    app()
