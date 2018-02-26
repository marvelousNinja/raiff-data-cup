from itertools import combinations
from functools import partial
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool

import pandas as pd
import numpy as np
from fire import Fire
from tqdm import tqdm
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN

from raiff.utils import train_validation_holdout_split
from raiff.utils import distance

def get_encoding(averages, global_mean, group):
    try:
        return pd.Series(index=group.index, data=averages.loc[group.iloc[0]].iloc[0])
    except KeyError:
        return pd.Series(index=group.index, data=global_mean)

def target_encode(df, source_df, column_names, target_column):
    column_names = list(column_names)
    averages = source_df.groupby(column_names)[target_column].agg(['mean', 'count'])
    global_mean = source_df[target_column].mean()
    k = 100
    encoded_column_name = f'encoded_{"_".join(column_names)}'
    averages[encoded_column_name] = (global_mean * k + averages['mean'] * averages['count']) / (k + averages['count'])
    combined = pd.Series(index=df.index, data=list(map(tuple, df[column_names].values)), name=encoded_column_name)
    return combined.map(averages[encoded_column_name]).fillna(global_mean)

def calc_median_distance(group):
    return pd.Series(index=group.index, data=distance(
        group[['transaction_lat', 'transaction_lon']].median(),
        group[['transaction_lat', 'transaction_lon']]
    ))

def cluster(group):
    model = DBSCAN(eps=0.01)
    return pd.Series(index=group.index, data=model.fit_predict(
        group[['transaction_lat', 'transaction_lon']]
    ))

def cluster_size(group):
    return pd.Series(index=group.index, data=len(group))

def cluster_amount(group):
    return pd.Series(index=group.index, data=group.amount.sum())

def fit():
    train = pd.read_csv('./data/train_set.csv', parse_dates=['transaction_date'], dtype={
        'atm_address': 'object',
        'pos_address': 'object'
    })

    # 1. Common preprocessing
    train['transaction_lat'] = train['atm_address_lat'].fillna(train['pos_address_lat'])
    train['transaction_lon'] = train['atm_address_lon'].fillna(train['pos_address_lon'])
    train['transaction_address'] = train['atm_address'].fillna(train['pos_address'])
    train = train.drop([
        'atm_address_lat', 'atm_address_lon', 'atm_address',
        'pos_address_lat', 'pos_address_lon', 'pos_address'
    ], axis=1)

    # 2. Common filtering
    train = train[train['country'].isin(['RUS', 'RU '])]
    train = train[train['currency'] == 643.0]
    train = train[~train.transaction_lat.isnull()]
    train = train.drop(['country', 'currency'], axis=1)

    # 3. Task specific filtering
    train = train[~train.work_add_lat.isnull()]

    # grouped_by_customer = train.groupby('customer_id', sort=False, as_index=False, group_keys=False)
    # train['cluster_id'] = grouped_by_customer.apply(cluster)
    # train['median_distance'] = grouped_by_customer.apply(calc_median_distance)
    # train['rounded_median_distance'] = train['median_distance'].round(2)

    # grouped_by_cluster = train.groupby(['customer_id', 'cluster_id'], sort=False, as_index=False, group_keys=False)
    # train['cluster_median_distance'] = grouped_by_cluster.apply(calc_median_distance)
    # train['cluster_size'] = grouped_by_cluster.apply(cluster_size)
    # train['cluster_amount'] = grouped_by_cluster.apply(cluster_amount)
    # train['rounded_cluster_amount'] = train['cluster_amount'].round(2)
    # train['rounded_cluster_median_distance'] = train['cluster_median_distance'].round(2)

    # 4. Train, validation & holdout split
    train, validation, _ = train_validation_holdout_split(train)

    # [
    #     'amount', 'city', 'customer_id', 'home_add_lat', 'home_add_lon', 'mcc',
    #     'terminal_id', 'transaction_date', 'work_add_lat', 'work_add_lon',
    #     'transaction_lat', 'transaction_lon', 'transaction_address'
    # ]

    train['is_close'] = (distance(
        train[['transaction_lat', 'transaction_lon']].values,
        train[['work_add_lat', 'work_add_lon']].values
    ) <= 0.02).astype(int)

    validation['is_close'] = (distance(
        validation[['transaction_lat', 'transaction_lon']].values,
        validation[['work_add_lat', 'work_add_lon']].values
    ) <= 0.02).astype(int)

    train['rounded_amount'] = train['amount'].round(2)
    train['day_of_week'] = train.transaction_date.dt.dayofweek

    initial_columns = [
        'mcc', 'city', 'day_of_week', 'rounded_amount',
        'terminal_id',
        # 'cluster_id', 'cluster_size', 'rounded_cluster_amount', 'rounded_cluster_median_distance',
        # 'rounded_median_distance',
    ]

    column_combinations = list(combinations(initial_columns, 1))
    column_combinations += list(combinations(initial_columns, 2))
    column_combinations += list(combinations(initial_columns, 3))
    column_combinations += list(combinations(initial_columns, 4))

    pool = ThreadPool()
    train['n_fold'] = np.random.randint(0, 5, len(train))
    for n_fold in range(5):
        target_fold = train.loc[train.n_fold == n_fold]
        source_folds = train.loc[train.n_fold != n_fold]

        for encoded_column in tqdm(map(partial(target_encode, target_fold, source_folds, target_column='is_close'), column_combinations)):
            train.loc[train.n_fold == n_fold, encoded_column.name] = encoded_column

    validation['rounded_amount'] = validation['amount'].round(2)
    validation['day_of_week'] = validation.transaction_date.dt.dayofweek

    for encoded_column in tqdm(map(partial(target_encode, validation, train, target_column='is_close'), column_combinations)):
        validation[encoded_column.name] = encoded_column

    feature_columns = [c for c in train.columns if c.startswith('encoded')]

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

if __name__ == '__main__':
    Fire(fit)
