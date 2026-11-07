"""Descarga el modelo champion del Model Registry para empaquetarlo en la imagen.

Uso:
    python scripts/export_model.py --question p1 --output models/p1_champion
"""

from pathlib import Path
import shutil

import mlflow
from mlflow import MlflowClient
import typer

from mi_bici.config import MLFLOW_TRACKING_URI, REGISTERED_MODELS

app = typer.Typer()


@app.command()
def main(
    question: str = 'p1',
    alias: str = 'champion',
    output: Path = Path('models/p1_champion'),
):
    name = REGISTERED_MODELS[question][0]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(name, alias)
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f'models:/{name}/{version.version}', dst_path=str(output)
    )
    # La API lee el nombre y la versión de este archivo
    (output / 'registry.txt').write_text(f'{name}\n{version.version}\n{version.run_id}\n')
    print(f'{name} v{version.version} -> {output}')


if __name__ == '__main__':
    app()
