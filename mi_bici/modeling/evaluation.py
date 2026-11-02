"""Comparaciones de calidad sobre las mismas observaciones para modelo y baseline."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


RISK_LABELS = ('deficit', 'balanceado', 'excedente')


def prediction_array(values, size, numeric=False):
    """Rechaza predicciones incompletas o desalineadas antes de calcular métricas."""
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != size or pd.isna(array).any():
        raise ValueError('Se requiere una predicción no nula por observación.')
    if numeric:
        array = array.astype(float)
        if not np.isfinite(array).all():
            raise ValueError('Los valores numéricos de evaluación deben ser finitos.')
    return array


def evaluation_frame(frame, date_column, target, baseline, predictions, numeric=False):
    """Prepara segmentos de validación y prueba sin mezclar entrenamiento."""
    required = ['split', 'station_id', date_column, target, baseline]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Faltan columnas para evaluar: {", ".join(missing)}')
    if frame.empty or not frame['split'].isin(['val', 'test']).all():
        raise ValueError('La evaluación requiere filas de val o test, sin entrenamiento.')
    if frame[required].isna().any().any():
        raise ValueError('Los datos de evaluación no pueden contener nulos.')
    work = frame[required].copy().reset_index(drop=True)
    work['_actual'] = prediction_array(work[target], len(work), numeric)
    work['_baseline'] = prediction_array(work[baseline], len(work), numeric)
    work['_model'] = prediction_array(predictions, len(work), numeric)
    work['month'] = pd.to_datetime(work[date_column]).dt.strftime('%Y-%m')
    return work


def grouped_reports(frame, scoring):
    reports = {}
    for name, columns in [
        ('summary', ['split']), ('by_station', ['split', 'station_id']),
        ('by_month', ['split', 'month']),
    ]:
        rows = []
        for keys, group in frame.groupby(columns, sort=True, observed=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            rows.append({**dict(zip(columns, keys)), **scoring(group)})
        reports[name] = pd.DataFrame(rows)
    return reports


def regression_comparison(group):
    actual = group['_actual'].to_numpy()
    total = float(actual.sum())
    result = {'n_samples': len(group), 'target_sum': total}
    for role in ('model', 'baseline'):
        # La evaluación agregada de train.py también limita demanda predicha a cero.
        residual = np.clip(group[f'_{role}'].to_numpy(), 0, None) - actual
        result.update({
            f'{role}_mae': float(np.abs(residual).mean()),
            f'{role}_rmse': float(np.sqrt(np.square(residual).mean())),
            f'{role}_wape': float(np.abs(residual).sum() / total) if total > 0 else np.nan,
            f'{role}_bias': float(residual.mean()),
        })
    for metric in ('mae', 'rmse', 'wape'):
        result[f'{metric}_improvement'] = result[f'baseline_{metric}'] - result[f'model_{metric}']
    baseline_mae = result['baseline_mae']
    result['mae_improvement_pct'] = (
        100 * result['mae_improvement'] / baseline_mae if baseline_mae > 0 else np.nan
    )
    return result


def regression_reports(frame, predictions):
    work = evaluation_frame(frame, 'date', 'trips_count', 'lag_7d', predictions, numeric=True)
    if (work['_actual'] < 0).any():
        raise ValueError('La demanda real no puede ser negativa.')
    reports = grouped_reports(work, regression_comparison)
    reports['by_station'] = reports['by_station'].sort_values(
        ['split', 'mae_improvement'], kind='stable'
    ).reset_index(drop=True)
    return reports


def class_statistics(actual, predicted):
    matrix = confusion_matrix(actual, predicted, labels=list(RISK_LABELS))
    true_positive = matrix.diagonal()
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    precision = np.divide(true_positive, predicted_support, out=np.zeros(3),
                          where=predicted_support > 0)
    recall = np.divide(true_positive, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3),
                   where=(precision + recall) > 0)
    classes = pd.DataFrame({
        'class': RISK_LABELS, 'support': support, 'predicted_support': predicted_support,
        'precision': precision, 'recall': recall, 'f1': f1,
        'false_negatives': support - true_positive,
        'false_positives': predicted_support - true_positive,
    })
    metrics = {
        'accuracy': float(true_positive.sum() / support.sum()),
        'f1_macro': float(f1.mean()), 'precision_macro': float(precision.mean()),
        'recall_macro': float(recall.mean()),
        'recall_deficit': float(recall[0]), 'precision_deficit': float(precision[0]),
        'missed_deficits': int(support[0] - true_positive[0]),
        'false_deficit_alerts': int(predicted_support[0] - true_positive[0]),
    }
    return metrics, classes, matrix


def classification_comparison(group):
    actual = group['_actual'].to_numpy()
    result = {'n_samples': len(group), 'n_deficit': int((actual == 'deficit').sum())}
    for role in ('model', 'baseline'):
        metrics, _, _ = class_statistics(actual, group[f'_{role}'].to_numpy())
        result.update({f'{role}_{name}': value for name, value in metrics.items()})
    for metric in ('accuracy', 'f1_macro', 'recall_deficit', 'precision_deficit'):
        result[f'{metric}_improvement'] = result[f'model_{metric}'] - result[f'baseline_{metric}']
    for metric in ('missed_deficits', 'false_deficit_alerts'):
        result[f'{metric}_reduction'] = result[f'baseline_{metric}'] - result[f'model_{metric}']
    avoided_misses = result['missed_deficits_reduction']
    result['extra_false_alerts_per_avoided_miss'] = (
        max(-result['false_deficit_alerts_reduction'], 0) / avoided_misses
        if avoided_misses > 0 else np.nan
    )
    return result


def classification_reports(frame, predictions):
    work = evaluation_frame(frame, 'hour_ts', 'risk_class', 'risk_lag_168h', predictions)
    for column in ('_actual', '_model', '_baseline'):
        if not work[column].isin(RISK_LABELS).all():
            raise ValueError(f'Clases de riesgo desconocidas en {column}.')
    reports = grouped_reports(work, classification_comparison)
    reports['by_station'] = reports['by_station'].sort_values(
        ['split', 'f1_macro_improvement'], kind='stable'
    ).reset_index(drop=True)
    class_rows, confusion_rows = [], []
    for split, group in work.groupby('split', sort=True):
        _, model_classes, model_matrix = class_statistics(group['_actual'], group['_model'])
        _, baseline_classes, baseline_matrix = class_statistics(
            group['_actual'], group['_baseline']
        )
        for index, label in enumerate(RISK_LABELS):
            row = {
                'split': split, 'class': label, 'support': int(model_classes.iloc[index].support)
            }
            for role, table in [('model', model_classes), ('baseline', baseline_classes)]:
                row.update({f'{role}_{metric}': table.iloc[index][metric] for metric in (
                    'precision', 'recall', 'f1', 'false_negatives', 'false_positives',
                    'predicted_support',
                )})
            row['f1_improvement'] = row['model_f1'] - row['baseline_f1']
            class_rows.append(row)
            for predicted_index, predicted_label in enumerate(RISK_LABELS):
                confusion_rows.append({
                    'split': split, 'actual_class': label, 'predicted_class': predicted_label,
                    'model_count': int(model_matrix[index, predicted_index]),
                    'baseline_count': int(baseline_matrix[index, predicted_index]),
                })
    reports['by_class'] = pd.DataFrame(class_rows)
    reports['confusion'] = pd.DataFrame(confusion_rows)
    return reports


def cluster_metrics(values, labels):
    count = len(labels)
    sizes = pd.Series(labels).value_counts()
    defined = 1 < len(sizes) < count
    return {
        'n_stations': count, 'n_clusters': len(sizes),
        'largest_cluster_share': float(sizes.max() / count) if count else np.nan,
        'silhouette_usage': float(silhouette_score(values, labels)) if defined else np.nan,
        'davies_bouldin_usage': float(davies_bouldin_score(values, labels)) if defined else np.nan,
        'metric_status': 'defined' if defined else 'undefined_cluster_count',
    }


def clustering_reports(train, holdout, model_train, model_holdout,
                       baseline_train, baseline_holdout, feature_columns):
    """Compara grupos en un espacio común ajustado exclusivamente con train."""
    for frame in (train, holdout):
        if frame.empty or frame['station_id'].isna().any() or frame.station_id.duplicated().any():
            raise ValueError('Los perfiles deben tener datos y una fila por estación.')
        if not np.isfinite(frame[feature_columns].to_numpy(dtype=float)).all():
            raise ValueError('Las características de clustering deben ser finitas.')
    scaler = StandardScaler().fit(train[feature_columns])
    train_values = scaler.transform(train[feature_columns])
    holdout_values = scaler.transform(holdout[feature_columns])
    common_ids = set(train.station_id) & set(holdout.station_id)
    summary_rows, profile_rows, station_tables = [], [], []
    for role, raw_train, raw_holdout in [
        ('model', model_train, model_holdout), ('baseline', baseline_train, baseline_holdout),
    ]:
        train_labels = prediction_array(raw_train, len(train), numeric=True)
        holdout_labels = prediction_array(raw_holdout, len(holdout), numeric=True)
        for split, frame, values, labels in [
            ('train', train, train_values, train_labels),
            ('holdout', holdout, holdout_values, holdout_labels),
        ]:
            for cohort in ('all', 'common'):
                mask = np.ones(len(frame), dtype=bool) if cohort == 'all' else (
                    frame.station_id.isin(common_ids).to_numpy()
                )
                summary_rows.append({
                    'model': role, 'split': split, 'cohort': cohort,
                    **cluster_metrics(values[mask], labels[mask]),
                })
            grouped = frame[feature_columns].assign(cluster=labels).groupby('cluster')
            profiles = grouped.mean().reset_index()
            profiles.insert(0, 'n_stations', grouped.size().to_numpy())
            profiles.insert(0, 'split', split)
            profiles.insert(0, 'model', role)
            profile_rows.append(profiles)

        stations = train[['station_id', *feature_columns]].assign(
            train_cluster=train_labels
        ).merge(
            holdout[['station_id', *feature_columns]].assign(holdout_cluster=holdout_labels),
            on='station_id', how='outer', suffixes=('_train', '_holdout'), indicator=True,
        )
        stations['model'] = role
        stations['cohort'] = stations['_merge'].map({
            'both': 'common', 'left_only': 'train_only', 'right_only': 'holdout_only',
        })
        stations = stations.drop(columns='_merge')
        stations['cluster_changed'] = pd.Series(pd.NA, index=stations.index, dtype='boolean')
        common = stations.cohort == 'common'
        stations.loc[common, 'cluster_changed'] = (
            stations.loc[common, 'train_cluster'] != stations.loc[common, 'holdout_cluster']
        )
        for column in feature_columns:
            stations[f'{column}_change'] = (
                stations[f'{column}_holdout'] - stations[f'{column}_train']
            )
        station_tables.append(stations)

    assignments = pd.concat(station_tables, ignore_index=True)
    common = assignments.loc[(assignments.model == 'model') & (assignments.cohort == 'common')]
    stability = pd.DataFrame([{
        'n_common_stations': len(common),
        'n_train_only': len(set(train.station_id) - common_ids),
        'n_holdout_only': len(set(holdout.station_id) - common_ids),
        'ari_fixed_model': float(adjusted_rand_score(
            common.train_cluster, common.holdout_cluster
        )) if len(common) >= 2 else np.nan,
        'same_cluster_rate': float((~common.cluster_changed).mean()) if len(common) else np.nan,
    }])
    return {
        'summary': pd.DataFrame(summary_rows),
        'by_cluster': pd.concat(profile_rows, ignore_index=True),
        'stations': assignments, 'stability': stability,
    }
