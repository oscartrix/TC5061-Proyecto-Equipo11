"""Búsqueda de hiperparámetros con nested runs: una corrida padre y una hija por combinación.

Uso:
    python -m mi_bici.modeling.search --question p1 --model ridge
"""

from itertools import product
import os

from loguru import logger
import mlflow
import typer

from mi_bici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    SEARCH_GRIDS,
    SELECTION_METRICS,
)
from mi_bici.modeling.train import RUNNERS

app = typer.Typer()


def generate_combinations(grid: dict) -> list[dict]:
    """Genera todas las combinaciones de un grid de hiperparámetros."""
    names = list(grid.keys())
    return [dict(zip(names, values)) for values in product(*grid.values())]


def is_better_score(score: float, best_score: float, direction: str) -> bool:
    """Indica si una métrica mejora el mejor resultado actual."""
    if direction == 'min':
        return score < best_score
    if direction == 'max':
        return score > best_score
    raise ValueError(f'Dirección de optimización inválida: {direction}')


def search_hyperparameters(question: str, model: str) -> None:
    """Corre una búsqueda de hiperparámetros y deja el resumen en la corrida padre."""
    if question not in SEARCH_GRIDS or model not in SEARCH_GRIDS[question]:
        raise ValueError(f'No existe un grid para {question}/{model}')

    combinations = generate_combinations(SEARCH_GRIDS[question][model])
    metric_name, direction = SELECTION_METRICS[question]
    best_score = float('inf') if direction == 'min' else float('-inf')
    best_params = None
    best_run_id = None

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    project_run_id = os.environ.get('MLFLOW_RUN_ID')
    if not project_run_id:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Con mlflow run la corrida padre es la que crea el proyecto
    with mlflow.start_run(run_id=project_run_id):
        os.environ.pop('MLFLOW_RUN_ID', None)
        mlflow.set_tag('mlflow.runName', f'{question}-search-{model}')
        mlflow.set_tags({'question': question, 'model_family': model, 'stage_of_work': 'search'})
        mlflow.log_params(
            {
                'search_model': model,
                'n_combinations': len(combinations),
                'objective_metric': metric_name,
            }
        )

        for index, hparams in enumerate(combinations, start=1):
            run_name = f'{question}-{model}-trial-{index}'
            with mlflow.start_run(run_name=run_name, nested=True) as child:
                RUNNERS[question](model, hparams)

                score = mlflow.get_run(child.info.run_id).data.metrics[metric_name]
                if is_better_score(score, best_score, direction):
                    best_score = score
                    best_params = dict(hparams)
                    best_run_id = child.info.run_id

        mlflow.log_metric(f'best_{metric_name}', best_score)
        mlflow.log_dict(best_params, 'best_hyperparameters.json')
        mlflow.set_tag('best_child_run_id', best_run_id)

    logger.info(f'Mejor configuración de {question}/{model}: {best_params} ({metric_name}={best_score:.4f})')


@app.command()
def main(
    question: str = typer.Option('p1', help='Pregunta: p1, p2 o p3'),
    model: str = typer.Option('ridge', help='Modelo con grid en SEARCH_GRIDS de config.py'),
):
    """Ejecuta una búsqueda de hiperparámetros."""
    if question not in SEARCH_GRIDS or model not in SEARCH_GRIDS[question]:
        raise typer.BadParameter(f'Búsqueda no disponible para question={question}, model={model}')
    search_hyperparameters(question, model)


if __name__ == '__main__':
    app()
