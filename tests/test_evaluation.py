"""Métricas verificables y comparaciones pareadas de calidad predictiva."""

import numpy as np
import pandas as pd
import pytest

from mi_bici.modeling.evaluation import (
    classification_reports, clustering_reports, regression_reports,
)


@pytest.fixture
def regression_data():
    return pd.DataFrame({
        'split': ['val', 'val', 'test', 'test'], 'station_id': [1, 2, 1, 2],
        'date': pd.to_datetime(['2026-01-01', '2026-02-01', '2026-07-01', '2026-08-01']),
        'trips_count': [0., 10., 10., 0.], 'lag_7d': [0., 0., 12., 0.],
    })


def test_regression_comparison_has_exact_metrics_and_separate_splits(regression_data):
    reports = regression_reports(regression_data, [-3., 8., 9., 0.])
    summary = reports['summary'].set_index('split')
    val = summary.loc['val']
    assert val['n_samples'] == 2
    assert val['model_mae'] == 1
    assert val['model_rmse'] == pytest.approx(np.sqrt(2))
    assert val['model_wape'] == .2
    assert val['model_bias'] == -1
    assert val['baseline_mae'] == 5
    assert val['mae_improvement'] == 4
    assert val['mae_improvement_pct'] == 80
    assert summary.loc['test', 'mae_improvement_pct'] == 50
    assert set(reports['by_month']['month']) == {'2026-01', '2026-02', '2026-07', '2026-08'}
    assert reports['by_station']['n_samples'].sum() == len(regression_data)
    assert reports['by_month']['n_samples'].sum() == len(regression_data)


def test_zero_demand_does_not_create_false_percentage_metrics(regression_data):
    reports = regression_reports(regression_data, [2., 8., 9., 0.])
    zero = reports['by_station'].query("split == 'val' and station_id == 1").iloc[0]
    assert zero['model_mae'] == 2
    assert zero['baseline_mae'] == 0
    assert zero['mae_improvement'] == -2
    assert np.isnan(zero['model_wape'])
    assert np.isnan(zero['mae_improvement_pct'])


@pytest.mark.parametrize('predictions', [[1.], [1., np.nan, 2., 3.],
                                         [1., np.inf, 2., 3.], [[1.]] * 4])
def test_invalid_predictions_are_rejected(regression_data, predictions):
    with pytest.raises(ValueError):
        regression_reports(regression_data, predictions)


def test_evaluation_does_not_accept_training_observations(regression_data):
    regression_data.loc[0, 'split'] = 'train'
    with pytest.raises(ValueError, match='sin entrenamiento'):
        regression_reports(regression_data, [0.] * 4)


@pytest.fixture
def classification_data():
    return pd.DataFrame({
        'split': ['val'] * 6, 'station_id': [1, 1, 1, 2, 2, 2],
        'hour_ts': pd.date_range('2026-01-01', periods=6, freq='h'),
        'risk_class': ['deficit', 'deficit', 'balanceado', 'balanceado', 'excedente', 'excedente'],
        'risk_lag_168h': ['balanceado'] * 6,
    })


def test_classification_exposes_missed_deficits_and_false_alerts(classification_data):
    reports = classification_reports(classification_data, [
        'deficit', 'balanceado', 'balanceado', 'excedente', 'excedente', 'deficit',
    ])
    row = reports['summary'].iloc[0]
    assert row['n_samples'] == 6
    assert row['model_f1_macro'] == .5
    assert row['baseline_f1_macro'] == pytest.approx(1 / 6)
    assert row['model_recall_deficit'] == .5
    assert row['model_missed_deficits'] == 1
    assert row['baseline_missed_deficits'] == 2
    assert row['missed_deficits_reduction'] == 1
    assert row['model_false_deficit_alerts'] == 1
    assert row['false_deficit_alerts_reduction'] == -1
    assert row['extra_false_alerts_per_avoided_miss'] == 1
    classes = reports['by_class'].set_index('class')
    assert classes.loc['deficit', 'support'] == 2
    assert classes.loc['deficit', 'model_precision'] == .5
    assert classes.loc['deficit', 'model_recall'] == .5
    assert classes.loc['deficit', 'model_f1'] == .5
    assert reports['confusion']['model_count'].sum() == 6
    assert reports['confusion']['baseline_count'].sum() == 6
    assert len(reports['confusion']) == 9
    assert reports['by_station']['n_samples'].sum() == 6


def test_absent_classes_keep_support_and_macro_definition(classification_data):
    classification_data['risk_class'] = 'balanceado'
    reports = classification_reports(classification_data, ['balanceado'] * 6)
    assert reports['summary'].iloc[0]['model_accuracy'] == 1
    assert reports['summary'].iloc[0]['model_f1_macro'] == pytest.approx(1 / 3)
    absent = reports['by_class'].query("`class` == 'deficit'").iloc[0]
    assert absent['support'] == 0
    assert absent['model_recall'] == 0


def test_unknown_risk_labels_are_not_silently_dropped(classification_data):
    with pytest.raises(ValueError, match='desconocidas'):
        classification_reports(classification_data, ['desconocida'] * 6)


@pytest.fixture
def clustering_data():
    train = pd.DataFrame({
        'station_id': [1, 2, 3, 4], 'f1': [0., 0., 10., 10.], 'f2': [0., 1., 10., 11.],
    })
    holdout = pd.DataFrame({
        'station_id': [4, 2, 5, 1, 3], 'f1': [10., 0., 20., 0., 10.],
        'f2': [11., 1., 20., 0., 10.],
    })
    return train, holdout


def cluster_reports(train, holdout):
    return clustering_reports(train, holdout, [0, 0, 1, 1], [1, 0, 1, 0, 1],
                              [0, 1, 0, 1], [1, 1, 0, 0, 0], ['f1', 'f2'])


def test_clustering_compares_same_cohorts_and_aligns_station_ids(clustering_data):
    reports = cluster_reports(*clustering_data)
    summary = reports['summary'].set_index(['model', 'split', 'cohort'])
    assert summary.loc[('model', 'train', 'all'), 'silhouette_usage'] > .9
    assert summary.loc[('baseline', 'train', 'all'), 'silhouette_usage'] < 0
    assert summary.loc[('model', 'holdout', 'all'), 'n_stations'] == 5
    assert summary.loc[('model', 'holdout', 'common'), 'n_stations'] == 4
    stability = reports['stability'].iloc[0]
    assert stability['n_common_stations'] == 4
    assert stability['n_holdout_only'] == 1
    assert stability['ari_fixed_model'] == 1
    assert stability['same_cluster_rate'] == 1
    new = reports['stations'].query("model == 'model' and station_id == 5").iloc[0]
    assert new['cohort'] == 'holdout_only'
    assert pd.isna(new['cluster_changed'])
    assert np.isnan(new['train_cluster'])
    profiles = reports['by_cluster'].query("model == 'model' and split == 'train'")
    assert profiles['f1'].tolist() == [0., 10.]
    assert profiles['n_stations'].tolist() == [2, 2]


def test_holdout_changes_do_not_refit_training_scaler(clustering_data):
    train, holdout = clustering_data
    original = cluster_reports(train, holdout)['summary'].query("split == 'train'")
    modified = holdout.copy()
    modified[['f1', 'f2']] *= 1000
    actual = cluster_reports(train, modified)['summary'].query("split == 'train'")
    pd.testing.assert_frame_equal(actual, original)


def test_undefined_clustering_metrics_are_not_reported_as_zero(clustering_data):
    train, holdout = clustering_data
    reports = clustering_reports(train, holdout, [0] * 4, [0] * 5,
                                 [0] * 4, [0] * 5, ['f1', 'f2'])
    assert reports['summary']['silhouette_usage'].isna().all()
    assert reports['summary']['davies_bouldin_usage'].isna().all()
    assert (reports['summary']['metric_status'] == 'undefined_cluster_count').all()


def test_no_common_stations_has_no_claimed_stability(clustering_data):
    train, holdout = clustering_data
    holdout = holdout.assign(station_id=[14, 12, 15, 11, 13])
    stability = cluster_reports(train, holdout)['stability'].iloc[0]
    assert stability['n_common_stations'] == 0
    assert np.isnan(stability['ari_fixed_model'])
    assert np.isnan(stability['same_cluster_rate'])
