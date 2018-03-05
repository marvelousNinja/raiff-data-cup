from functools import partial

import numpy as np
from fire import Fire
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from raiff.pipelines import fit_pipeline
from raiff.steps import read
from raiff.steps import preprocess
from raiff.steps import russia_only
from raiff.steps import rouble_only
from raiff.steps import with_columns
from raiff.steps import with_transaction_location
from raiff.steps import cluster
from raiff.steps import merge_cluster_features
from raiff.steps import fit_categories
from raiff.steps import transform_categories
from raiff.steps import calc_is_close
from raiff.utils import train_validation_holdout_split

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
        cluster,
        merge_cluster_features,
        (partial(fit_categories, ['mcc']), transform_categories),
        partial(calc_is_close, ['transaction_lat', 'transaction_lon'], target_columns)
    ]

    pipeline, train = fit_pipeline(steps, train)
    validation = pipeline(validation)

    feature_columns = [
        # Original transaction features
        'amount', 'mcc',
        # Cluster features
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

    import pdb; pdb.set_trace()

    return model, pipeline, feature_columns

if __name__ == '__main__':
    Fire()
