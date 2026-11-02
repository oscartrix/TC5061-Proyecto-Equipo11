"""Inferencia mediante la CLI con modelos registrados en un MLflow temporal."""

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from typer.testing import CliRunner

from mi_bici.config import REGISTERED_MODELS
from mi_bici.modeling import predict


@pytest.fixture(scope='module')
def champions(tmp_path_factory):
    root = tmp_path_factory.mktemp('predict-registry')
    uri = f'sqlite:///{root / "mlflow.db"}'
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment(
        'predict', artifact_location=(root / 'artifacts').as_uri()
    )
    x = pd.DataFrame({
        'flow': [1., 2., 3., 4., 5., 6.],
        'capacity': [20., 25., 15., 40., 10., 30.],
    })
    models = {
        'p1': LinearRegression().fit(x, [10., 15., 20., 25., 30., 35.]),
        'p2': DummyClassifier().fit(x, ['balanceado'] * 4 + ['deficit'] * 2),
        'p3': KMeans(n_clusters=2, random_state=42, n_init=10).fit(x),
    }
    try:
        client = mlflow.MlflowClient()
        for question, model in models.items():
            with mlflow.start_run(experiment_id=experiment_id):
                info = mlflow.sklearn.log_model(
                    model,
                    name='model',
                    signature=infer_signature(x, model.predict(x)),
                    input_example=x.head(2),
                )
            name = REGISTERED_MODELS[question][0]
            version = mlflow.register_model(info.model_uri, name)
            client.set_registered_model_alias(name, 'champion', version.version)
        yield uri, x, models
    finally:
        mlflow.set_tracking_uri(previous_uri)


@pytest.fixture
def registry(champions, monkeypatch):
    uri, x, models = champions
    monkeypatch.setattr(predict, 'MLFLOW_TRACKING_URI', uri)
    return x, models


def invoke(question, source, output):
    return CliRunner().invoke(predict.app, [
        '--question', question,
        '--features-path', str(source),
        '--predictions-path', str(output),
    ])


@pytest.mark.parametrize('question', ['p1', 'p2', 'p3'])
@pytest.mark.parametrize('source_format,output_format', [('csv', 'parquet'), ('parquet', 'csv')])
def test_cli_predicts_with_champion_and_preserves_rows(
    registry, tmp_path, question, source_format, output_format
):
    x, models = registry
    # El orden de columnas difiere de la firma y los IDs deben conservar su orden.
    features = x[['capacity', 'flow']].copy()
    features.insert(0, 'station_id', [30, 10, 20, 50, 40, 60])
    source = tmp_path / f'features.{source_format}'
    output = tmp_path / 'predictions' / f'{question}.{output_format}'
    if source_format == 'csv':
        features.to_csv(source, index=False)
    else:
        features.to_parquet(source, index=False)

    result = invoke(question, source, output)

    assert result.exit_code == 0, result.output
    saved = pd.read_csv(output) if output_format == 'csv' else pd.read_parquet(output)
    pd.testing.assert_frame_equal(saved[features.columns], features, check_dtype=False)
    assert saved.columns.tolist() == [*features.columns, 'prediction']
    expected = models[question].predict(x)
    if question == 'p1':
        np.testing.assert_allclose(saved['prediction'], expected)
    else:
        np.testing.assert_array_equal(saved['prediction'], expected)


def test_missing_model_columns_fail_without_output(registry, tmp_path):
    x, _ = registry
    source = tmp_path / 'features.parquet'
    output = tmp_path / 'predictions.parquet'
    x.drop(columns='capacity').to_parquet(source, index=False)

    result = invoke('p1', source, output)

    assert result.exit_code != 0
    assert 'Faltan columnas requeridas' in result.output
    assert 'capacity' in result.output
    assert not output.exists()


def test_incompatible_column_types_fail_without_output(registry, tmp_path):
    x, _ = registry
    source = tmp_path / 'features.parquet'
    output = tmp_path / 'predictions.parquet'
    x.assign(flow='no es un número').to_parquet(source, index=False)

    result = invoke('p1', source, output)

    assert result.exit_code == 2
    assert 'flow' in result.output
    assert not output.exists()


@pytest.mark.parametrize('case,message', [
    ('empty', 'no puede estar vacío'),
    ('missing', 'no encontrado'),
    ('format', 'Formato no soportado'),
    ('output_format', 'Formato no soportado'),
    ('same_path', 'distinta del archivo de entrada'),
    ('prediction_column', 'ya contiene una columna prediction'),
    ('question', 'La pregunta debe ser una de'),
])
def test_invalid_inputs_fail_before_loading_model(tmp_path, monkeypatch, case, message):
    source = tmp_path / ('features.json' if case == 'format' else 'features.csv')
    output = tmp_path / ('predictions.json' if case == 'output_format' else 'predictions.csv')
    features = pd.DataFrame({'flow': [1.], 'capacity': [20.]})
    if case == 'empty':
        features = features.iloc[:0]
    if case == 'prediction_column':
        features['prediction'] = 0
    if case != 'missing':
        features.to_csv(source, index=False)
    if case == 'same_path':
        output = source
    original = source.read_bytes() if source.exists() else None

    def unexpected_load(*args, **kwargs):
        pytest.fail('No se debe cargar el modelo con una entrada inválida')

    monkeypatch.setattr(mlflow.pyfunc, 'load_model', unexpected_load)

    result = invoke('p4' if case == 'question' else 'p1', source, output)

    assert result.exit_code != 0
    assert message in result.output
    if case == 'same_path':
        assert source.read_bytes() == original
    else:
        assert not output.exists()


def test_missing_registered_model_returns_cli_error(registry, tmp_path, monkeypatch):
    x, _ = registry
    source = tmp_path / 'features.csv'
    output = tmp_path / 'predictions.csv'
    x.to_csv(source, index=False)
    monkeypatch.setattr(predict, 'REGISTERED_MODELS', {'p1': ('modelo-inexistente', 'Prueba')})

    result = invoke('p1', source, output)

    assert result.exit_code != 0
    assert 'modelo-inexistente' in result.output
    assert 'not found' in result.output
    assert not output.exists()
