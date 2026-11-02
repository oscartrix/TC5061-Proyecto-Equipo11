"""Evaluación por CLI, trazabilidad y reportes con un registro real temporal."""

import json

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from typer.testing import CliRunner

from mi_bici.modeling import evaluate
from mi_bici.modeling.train import (
    P1_NUM_COLS, P1_CAT_COLS, P2_FEATURES, P3_USAGE_COLS, build_p1_model, build_p3_model,
)


@pytest.fixture
def evaluation_registry(question, valid_feature_frames, tmp_path, monkeypatch):
    previous_uri = mlflow.get_tracking_uri()
    uri = f'sqlite:///{tmp_path / "mlflow.db"}'
    mlflow.set_tracking_uri(uri)
    monkeypatch.setattr(evaluate, 'MLFLOW_TRACKING_URI', uri)
    monkeypatch.delenv('MLFLOW_RUN_ID', raising=False)
    experiment_id = mlflow.create_experiment(
        evaluate.MLFLOW_EXPERIMENT_NAME, artifact_location=(tmp_path / 'artifacts').as_uri()
    )
    for name in ('P3_train', 'P3_holdout'):
        frame = valid_feature_frames[name]
        frame['latitude'] += np.arange(6) * .01
        frame['longitude'] += np.arange(6) * .005
        frame['avg_daily_trips'] = np.arange(1., 7.)
        frame['dep_morning'] = np.arange(6) / 10
    for name, attribute in {
        'P1': 'P1_FEATURES_PATH', 'P2': 'P2_FEATURES_PATH',
        'P3_train': 'P3_TRAIN_FEATURES_PATH', 'P3_holdout': 'P3_HOLDOUT_FEATURES_PATH',
    }.items():
        path = tmp_path / f'{name}.parquet'
        valid_feature_frames[name].to_parquet(path, index=False)
        monkeypatch.setattr(evaluate, attribute, path)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'data_version': 'synthetic'}))
    monkeypatch.setattr(evaluate, 'MANIFEST_PATH', manifest)

    if question == 'p1':
        frame = valid_feature_frames['P1'].query("split == 'train'")
        x = frame[P1_NUM_COLS + P1_CAT_COLS]
        model = build_p1_model('ridge', {'alpha': 1.}).fit(x, frame.trips_count)
    elif question == 'p2':
        frame = valid_feature_frames['P2'].query("split == 'train'")
        x = frame[P2_FEATURES]
        model = DummyClassifier().fit(x, frame.risk_class)
    else:
        x = valid_feature_frames['P3_train'][P3_USAGE_COLS]
        model = build_p3_model('kmeans_usage', {'n_clusters': 2}).fit(x)
    try:
        with mlflow.start_run(experiment_id=experiment_id):
            info = mlflow.sklearn.log_model(
                model, name='model', input_example=x.head(2),
                signature=infer_signature(x, model.predict(x)),
            )
        name = evaluate.REGISTERED_MODELS[question][0]
        version = mlflow.register_model(info.model_uri, name)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(name, 'champion', version.version)
        yield client, version, experiment_id
    finally:
        mlflow.set_tracking_uri(previous_uri)


@pytest.mark.parametrize('question', ['p1', 'p2', 'p3'])
def test_cli_writes_comparisons_and_mlflow_artifacts(evaluation_registry, tmp_path, question):
    client, version, experiment_id = evaluation_registry
    output = tmp_path / 'evaluation'
    result = CliRunner().invoke(evaluate.app, [
        '--question', question, '--output-dir', str(output),
    ])
    assert result.exit_code == 0, result.output
    directory = output / question
    required = {'summary.csv', 'metadata.json'}
    required |= {'by_cluster.csv', 'stations.csv', 'stability.csv'} if question == 'p3' else {
        'by_station.csv', 'by_month.csv'
    }
    if question == 'p2':
        required |= {'by_class.csv', 'confusion.csv'}
    assert required.issubset({path.name for path in directory.iterdir()})
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert metadata['model_uri'].endswith(f'/{version.version}')
    assert metadata['source_run_id'] == version.run_id
    assert metadata['data_version'] == 'synthetic'
    assert metadata['evaluation_changes_selection'] is False
    runs = client.search_runs([experiment_id], "tags.stage_of_work = 'evaluation'")
    assert len(runs) == 1
    assert runs[0].info.status == 'FINISHED'
    artifacts = {item.path.split('/')[-1]
                 for item in client.list_artifacts(runs[0].info.run_id, 'evaluation')}
    assert required.issubset(artifacts)
    assert runs[0].data.metrics
    name = evaluate.REGISTERED_MODELS[question][0]
    assert client.get_model_version_by_alias(name, 'champion').run_id == version.run_id
    assert len(client.search_model_versions(f"name = '{name}'")) == 1


@pytest.mark.parametrize('question', ['p1'])
def test_missing_split_stops_evaluation(evaluation_registry, tmp_path, question):
    import pandas as pd

    frame = pd.read_parquet(evaluate.P1_FEATURES_PATH)
    frame.loc[frame.split != 'val'].to_parquet(evaluate.P1_FEATURES_PATH, index=False)
    output = tmp_path / 'evaluation'
    result = CliRunner().invoke(evaluate.app, [
        '--question', question, '--output-dir', str(output),
    ])
    assert result.exit_code == 2
    assert 'validación y prueba' in result.output
    assert not output.exists()


def test_unknown_question_stops_before_loading_models(tmp_path):
    result = CliRunner().invoke(evaluate.app, ['--question', 'p4', '--output-dir', str(tmp_path)])
    assert result.exit_code == 2
    assert 'La pregunta debe ser una de' in result.output
