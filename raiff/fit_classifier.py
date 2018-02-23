import numpy as np
import reverse_geocoder as rg
import pandas as pd
from pandas.api.types import CategoricalDtype
from fire import Fire
from lightgbm.sklearn import LGBMClassifier
from sklearn.externals.joblib import dump

from raiff.utils import generate_model_name
from raiff.utils import distance
from raiff.utils import train_validation_holdout_split

def generate_customer_features(group):
    group['transaction_count'] = len(group)

    group['median_distance'] = distance(
        group[['transaction_lat', 'transaction_lon']].median(),
        group[['transaction_lat', 'transaction_lon']]
    )

    points = group[['transaction_lat', 'transaction_lon']].values
    points_sum = np.sum(points ** 2, axis=1)
    # TODO AS: np.clip solves issues with small negative values in sqrt, but degrades performance
    dists = np.sqrt(np.clip(-2 * (points @ points.T) + points_sum + points_sum.reshape(-1, 1), a_min=0.0000001, a_max=None))
    group['n_neighbors_001'] = (np.sum(dists <= 0.01, axis=1) - 1) / len(group)
    group['n_neighbors_002'] = (np.sum(dists <= 0.02, axis=1) - 1) / len(group)
    group['n_neighbors_004'] = (np.sum(dists <= 0.04, axis=1) - 1) / len(group)
    return group

def fit_pipeline(train):
    mcc_categories = train.mcc.astype('category').cat.categories
    mcc_dtype = CategoricalDtype(categories=mcc_categories)
    city_categories = train.city.astype('category').cat.categories
    city_dtype = CategoricalDtype(categories=city_categories)
    new_city_categories = train.new_city.astype('category').cat.categories
    new_city_dtype = CategoricalDtype(categories=new_city_categories)
    terminal_id_categories = train.terminal_id.astype('category').cat.categories
    terminal_dtype = CategoricalDtype(categories=terminal_id_categories)

    def transform(df):
        df = df.copy()

        df = df.groupby('customer_id').apply(generate_customer_features)
        df['mcc'] = df['mcc'].astype(mcc_dtype).cat.codes.astype('category')
        df['city'] = df['city'].astype(city_dtype).cat.codes.astype('category')
        df['new_city'] = df['new_city'].astype(new_city_dtype).cat.codes.astype('category')
        df['terminal_id'] = df['terminal_id'].astype(terminal_dtype).cat.codes.astype('category')
        df['day'] = pd.to_datetime(df.transaction_date).dt.day.astype('category')
        df['month'] = pd.to_datetime(df.transaction_date).dt.month.astype('category')
        df['day_of_week'] = pd.to_datetime(df.transaction_date).dt.dayofweek.astype('category')

        features = df[[
            'amount',
            'mcc',
            'day_of_week',
            'new_city',
            'terminal_id',
            'transaction_count',
            'median_distance',
            # 'n_neighbors_001',
            'n_neighbors_002',
            # 'n_neighbors_004'
        ]]

        df['dist'] = distance(
            df[['transaction_lat', 'transaction_lon']].values,
            df[['work_add_lat', 'work_add_lon']].values
        )

        labels = (df['dist'] <= 0.02).astype(int)
        weights = 1 / (np.clip(df['dist'], a_min=0.015, a_max=10))
        return features, labels, weights

    return transform

def fix_terminal_locations(terminal_transactions):
    terminal_transactions[['transaction_lat', 'transaction_lon']] = terminal_transactions[['transaction_lat', 'transaction_lon']].median().values
    return terminal_transactions

# TODO AS: Incorporate that
# df[df.mcc == 6011] = df[df.mcc == 6011].groupby('terminal_id').apply(fix_terminal_locations)
# df['new_city'] = list(map(lambda location: location['admin1'], rg.search(tuple(map(tuple, df[['transaction_lat', 'transaction_lon']].values)))))

def fit():
    train = pd.read_csv('./data/train_set.csv')
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

    # 3.1. Custom feature eng
    # TODO AS: Incorporate that
    # train[train.mcc == 6011] = train[train.mcc == 6011].groupby('terminal_id').apply(fix_terminal_locations)
    train['new_city'] = list(map(lambda location: location['admin1'], rg.search(tuple(map(tuple, train[['transaction_lat', 'transaction_lon']].values)))))

    # 4. Train, validation, holdout split
    train, validation, _ = train_validation_holdout_split(train)
    transform = fit_pipeline(train)

    train_x, train_y, _ = transform(train)
    validation_x, validation_y, _ = transform(validation)

    model = LGBMClassifier(categorical_feature=train_x.select_dtypes(include='category').columns.values)
    model.fit(train_x, train_y)

    validation['probs'] = model.predict_proba(validation_x)[:, 1]
    validation['is_close'] = validation_y

    predictions = validation.groupby('customer_id').apply(lambda group: group.sort_values('probs', ascending=False).head(1))
    score = predictions['is_close'].mean()
    print(score)

    name = generate_model_name(model, score)
    dump(model, f'./data/models/{name}')
    dump(model, f'./data/models/_latest.model')
    print(f'Model saved to ./data/models/{name}')

if __name__ == '__main__':
    Fire(fit)
