import os
from functools import partial

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from fire import Fire
from tqdm import tqdm
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

from raiff.pipelines import fit_pipeline
from raiff.steps import read
from raiff.steps import preprocess
from raiff.steps import russia_only
from raiff.steps import rouble_only
from raiff.steps import with_columns
from raiff.steps import fix_mcc
from raiff.steps import with_transaction_location
from raiff.utils import train_validation_holdout_split
from raiff.utils import distance
from raiff.utils import has_columns
from raiff.utils import generate_submission_name

def as_clusters(target_columns, df):
    clusters = []

    for customer_id, transactions in tqdm(df.groupby('customer_id')):
        transactions = transactions.copy()
        transactions['cluster_id'] = DBSCAN(eps=0.005).fit_predict(transactions[['transaction_lat', 'transaction_lon']])

        for cluster_id, cluster in transactions.groupby('cluster_id'):
            if cluster_id == -1: continue

            cluster_median = cluster[['transaction_lat', 'transaction_lon']].median()

            amount_histogram = cluster.amount.round().value_counts(normalize=True)
            amount_histogram = amount_histogram.add_prefix('amount_hist_').to_dict()
            mcc_whitelist = [
                5411.0, 6011.0, 5814.0, 5812.0, 5499.0,
                5541.0, 5912.0, 4111.0, 5921.0, 5331.0,
                5691.0, 5261.0, 5977.0
            ]

            mcc_histogram = cluster.mcc.astype('float').astype(CategoricalDtype(categories=mcc_whitelist)).value_counts(normalize=True, dropna=False)
            mcc_histogram = mcc_histogram.add_prefix('mcc_hist_').to_dict()
            day_histogram = cluster.transaction_date.dt.dayofweek.value_counts(normalize=True).add_prefix('day_hist_').to_dict()

            try:
                area = ConvexHull(cluster[['transaction_lat', 'transaction_lon']]).area # pylint: disable=no-member
            except Exception as _:
                area = 0

            features = {
                'cluster_id': cluster_id,
                'customer_id': customer_id,
                'cluster_lat': cluster_median['transaction_lat'],
                'cluster_lon': cluster_median['transaction_lon'],
                'area': area,
                'transaction_ratio': len(cluster) / len(transactions),
                'amount_ratio': np.sum(np.exp(cluster.amount)) / np.sum(np.exp(transactions.amount)),
                'date_ratio': len(cluster.transaction_date.unique()) / len(transactions.transaction_date.unique()),
                'amount_hist_-2.0': 0,
                'amount_hist_-1.0': 0,
                'amount_hist_0.0': 0,
                'amount_hist_1.0': 0,
                'amount_hist_2.0': 0,
                'amount_hist_3.0': 0,
                'amount_hist_4.0': 0,
                'amount_hist_5.0': 0,
                'amount_hist_6.0': 0,
                **amount_histogram,
                'mcc_hist_5411.0': 0,
                'mcc_hist_6011.0': 0,
                'mcc_hist_5814.0': 0,
                'mcc_hist_5812.0': 0,
                'mcc_hist_5499.0': 0,
                'mcc_hist_4111.0': 0,
                'mcc_hist_5921.0': 0,
                'mcc_hist_5331.0': 0,
                'mcc_hist_5691.0': 0,
                'mcc_hist_5261.0': 0,
                'mcc_hist_5977.0': 0,
                'mcc_hist_nan': 0,
                **mcc_histogram,
                'day_hist_0': 0,
                'day_hist_1': 0,
                'day_hist_2': 0,
                'day_hist_3': 0,
                'day_hist_4': 0,
                'day_hist_5': 0,
                'day_hist_6': 0,
                **day_histogram
            }

            if has_columns(target_columns, df):
                target_lat, target_lon = transactions.iloc[0][target_columns]
                target_distance = distance(np.array([target_lat, target_lon]), np.array([cluster_median.values]))[0]
                features['distance'] = target_distance
                features['is_close'] = int(target_distance <= 0.02)
                features['target_lat'] = target_lat
                features['target_lon'] = target_lon

            clusters.append(features)

    return pd.DataFrame(clusters)

def fit(objective):
    if objective == 'work':
        target_columns = ['work_add_lat', 'work_add_lon']
    else:
        target_columns = ['home_add_lat', 'home_add_lon']

    train, validation, _ = train_validation_holdout_split(read('./data/train_set.csv'))

    steps = [
        preprocess,
        russia_only,
        rouble_only,
        with_transaction_location,
        partial(with_columns, target_columns),
        partial(as_clusters, target_columns)
    ]

    pipeline, train = fit_pipeline(steps, train)
    validation = pipeline(validation)

    feature_columns = [
        'amount_hist_-1.0', 'amount_hist_-2.0', 'amount_hist_0.0',
        'amount_hist_1.0', 'amount_hist_2.0', 'amount_hist_3.0',
        'amount_hist_4.0', 'amount_hist_5.0', 'amount_hist_6.0', 'amount_ratio',
        'area', 'cluster_id', 'date_ratio', 'day_hist_0', 'day_hist_1',
        'day_hist_2', 'day_hist_3', 'day_hist_4', 'day_hist_5', 'day_hist_6',
        'mcc_hist_4111.0', 'mcc_hist_5261.0', 'mcc_hist_5331.0',
        'mcc_hist_5411.0', 'mcc_hist_5499.0', 'mcc_hist_5541.0',
        'mcc_hist_5691.0', 'mcc_hist_5812.0', 'mcc_hist_5814.0',
        'mcc_hist_5912.0', 'mcc_hist_5921.0', 'mcc_hist_5977.0',
        'mcc_hist_6011.0', 'mcc_hist_nan', 'transaction_ratio'
    ]

    print(f'Train size: {len(train)}, Validation size: {len(validation)}')
    print(f'Features: {feature_columns}')
    model = LGBMClassifier()
    model.fit(train[feature_columns], train['is_close'])

    predictions = model.predict_proba(validation[feature_columns])
    accuracy_value = accuracy_score(validation['is_close'], np.argmax(predictions, axis=1))
    logloss_value = log_loss(validation['is_close'], predictions)
    print(f'Accuracy: {accuracy_value:.5f}, Logloss: {logloss_value:.5f}')
    print(classification_report(validation['is_close'], np.argmax(predictions, axis=1)))

    validation['probs'] = predictions[:, 1]
    top1_accuracy = validation.groupby('customer_id').apply(lambda group: group.sort_values('probs').tail(1).is_close.max()).mean()
    top5_accuracy = validation.groupby('customer_id').apply(lambda group: group.sort_values('probs').tail(5).is_close.max()).mean()
    top10_accuracy = validation.groupby('customer_id').apply(lambda group: group.sort_values('probs').tail(10).is_close.max()).mean()
    print(f'Top1: {top1_accuracy:.5f}')
    print(f'Top5: {top5_accuracy:.5f}')
    print(f'Top10: {top10_accuracy:.5f}')

    return model, pipeline, feature_columns

def make_submission():
    work_model, work_pipeline, work_columns = fit(objective='work')
    home_model, home_pipeline, home_columns = fit(objective='home')
    test = fix_mcc(read('./data/test_set.csv'))

    submission = pd.DataFrame(index=test.customer_id.unique())
    submission.index.name = 'customer_id'

    work_df = work_pipeline(test)
    work_df['probs'] = work_model.predict_proba(work_df[work_columns])[:, 1]

    work_locations = work_df.groupby('customer_id').apply(lambda group: group.sort_values('probs').tail(1)[['cluster_lat', 'cluster_lon']])
    work_locations = work_locations.reset_index(drop=True, level=1)

    submission['_WORK_LAT_'] = 0
    submission['_WORK_LON_'] = 0
    submission.loc[work_locations.index, ['_WORK_LAT_', '_WORK_LON_']] = work_locations.values

    home_df = home_pipeline(test)
    home_df['probs'] = home_model.predict_proba(home_df[home_columns])[:, 1]

    home_locations = home_df.groupby('customer_id').apply(lambda group: group.sort_values('probs').tail(1)[['cluster_lat', 'cluster_lon']])
    home_locations = home_locations.reset_index(drop=True, level=1)

    submission['_HOME_LAT_'] = 0
    submission['_HOME_LON_'] = 0
    submission.loc[home_locations.index, ['_HOME_LAT_', '_HOME_LON_']] = home_locations.values

    submission.index.name = '_ID_'
    submission_path = os.path.join('./data', 'submissions', generate_submission_name())
    submission.to_csv(submission_path)

    import pdb; pdb.set_trace()

if __name__ == '__main__':
    Fire()
