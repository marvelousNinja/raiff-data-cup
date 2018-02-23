from functools import partial

import numpy as np
import pandas as pd
from fire import Fire
from pandas.api.types import CategoricalDtype
from lightgbm.sklearn import LGBMClassifier, LGBMRanker

from raiff.utils import train_validation_holdout_split
from raiff.utils import distance

def order_by_dist(group):
    group = group.copy()
    group = group.sort_values('dist')
    group['rank'] = list(range(len(group)))
    return group['rank']

def as_ranking_task(df, feature_columns, rank_column, group_column):
    sorted_df = df.sort_values(group_column)
    return sorted_df[feature_columns], sorted_df[rank_column], sorted_df.groupby(group_column).size().values

def fit_pipeline(train_df):
    mcc_categories = train_df.mcc.astype('category').cat.categories
    mcc_dtype = CategoricalDtype(categories=mcc_categories)
    city_categories = train_df.city.astype('category').cat.categories
    city_dtype = CategoricalDtype(categories=city_categories)
    terminal_id_categories = train_df.terminal_id.astype('category').cat.categories
    terminal_dtype = CategoricalDtype(categories=terminal_id_categories)

    def transform(df):
        df = df.copy()
        df['mcc'] = df['mcc'].astype(mcc_dtype).cat.codes.astype('category')
        df['city'] = df['city'].astype(city_dtype).cat.codes.astype('category')
        df['terminal_id'] = df['terminal_id'].astype(terminal_dtype).cat.codes.astype('category')
        df['day'] = pd.to_datetime(df.transaction_date).dt.day.astype('category')
        df['month'] = pd.to_datetime(df.transaction_date).dt.month.astype('category')
        df['day_of_week'] = pd.to_datetime(df.transaction_date).dt.dayofweek.astype('category')

        df['dist'] = distance(df[['transaction_lat', 'transaction_lon']].values, df[['work_add_lat', 'work_add_lon']])
        df['rank'] = df.groupby('customer_id').apply(order_by_dist).reset_index(drop=True, level=0)
        return df

    return transform

def rank_transactions(model, feature_columns, transactions):
    transactions = transactions.copy()
    transactions['score'] = model.predict(transactions[feature_columns])
    return transactions['score']

def fit():
    # 1. Common preprocessing
    train = pd.read_csv('./data/train_set.csv')
    train['transaction_lat'] = train['atm_address_lat'].fillna(train['pos_address_lat'])
    train['transaction_lon'] = train['atm_address_lon'].fillna(train['pos_address_lon'])
    train['transaction_address'] = train['atm_address'].fillna(train['pos_address'])
    train = train.drop([
        'atm_address_lat', 'pos_address_lat',
        'atm_address_lon', 'pos_address_lon',
        'atm_address', 'pos_address'
    ], axis=1)

    # 2. Some generic filtering, orig len is 1,224,734
    train = train[train['country'].isin(['RUS', 'RU '])] # drops 4,000 samples
    train = train[train['currency'] == 643.0] # drops 7,143 samples
    train = train[~train.transaction_lat.isnull()] # drops 97,000 samples
    train = train.drop([
        'country', 'currency'
    ], axis=1)

    # 3. Task specific filtering
    train = train[~train.work_add_lat.isnull()] # drops 512,269 samples

    # 4. Train, validation, holdout split
    train, validation, _ = train_validation_holdout_split(train)
    transform = fit_pipeline(train)

    train = transform(train)
    validation = transform(validation)

    feature_columns = ['amount', 'mcc', 'terminal_id', 'day_of_week']

    train_x, train_y, train_groups = as_ranking_task(
        train,
        feature_columns=feature_columns,
        rank_column='rank',
        group_column='customer_id'
    )

    model = LGBMRanker(
        categorical_feature=train_x.select_dtypes(include='category').columns.values,
        label_gain=list(range(2000)))

    model.fit(train_x, train_y, group=train_groups)
    validation['score'] = validation.groupby('customer_id').apply(partial(rank_transactions, model, feature_columns)).reset_index(drop=True, level=0)
    validation['is_close'] = (validation['dist'] <= 0.02).astype(int)
    predictions = validation.groupby('customer_id').apply(lambda group: group.sort_values('score', ascending=True).head(1))
    score = predictions['is_close'].mean()
    print(score)

    import pdb; pdb.set_trace()

if __name__ == '__main__':
    Fire(fit)
