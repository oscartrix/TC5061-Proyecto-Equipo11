"""Genera un smoke model de P1 para construir y probar la imagen sin entrenar.

Uso:
    python scripts/smoke_model.py --output models/p1_champion
"""

from pathlib import Path
import shutil

from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import typer

from mi_bici.api.schemas import EXAMPLE_FEATURES
from mi_bici.modeling.train import P1_CAT_COLS, P1_NUM_COLS

app = typer.Typer()


@app.command()
def main(output: Path = Path('models/p1_champion')):
    # Mismos tipos que la tabla de P1 para que la firma coincida con la del champion
    x = pd.DataFrame([
        {**EXAMPLE_FEATURES, 'lag_1d': float(i), 'zone': ['GDL', 'ZPN', 'TLQ'][i % 3]}
        for i in range(12)
    ])[P1_NUM_COLS + P1_CAT_COLS].astype(
        {'capacity': 'float64', 'day_of_week': 'int32', 'month': 'int32'}
    )
    prep = ColumnTransformer([
        ('num', 'passthrough', P1_NUM_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), P1_CAT_COLS),
    ])
    pipeline = Pipeline([('prep', prep), ('reg', LinearRegression())]).fit(x, x['lag_1d'])
    if output.exists():
        shutil.rmtree(output)
    mlflow.sklearn.save_model(
        pipeline,
        output,
        signature=infer_signature(x, pipeline.predict(x)),
        serialization_format='cloudpickle',
    )
    # Mismo formato que export_model.py: nombre, versión y corrida
    (output / 'registry.txt').write_text('smoke-model\n0\nsmoke\n')
    print(f'smoke-model v0 -> {output}')


if __name__ == '__main__':
    app()
