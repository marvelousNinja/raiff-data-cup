import numpy as np
import reverse_geocoder as rg
import pandas as pd
from pandas.api.types import CategoricalDtype
from fire import Fire
from lightgbm.sklearn import LGBMClassifier
from sklearn.externals.joblib import dump

from raiff.datasets import get_datasets
from raiff.utils import generate_model_name

def with_home_location(df):
    return df[~df.home_add_lat.isnull()]

def with_work_location(df):
    return df[~df.work_add_lat.isnull()]

def distance(first, second):
    return np.sqrt(np.sum((first - second) ** 2, axis=1))

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

def pipeline(full, location_columns, df):
    mcc_categories = full.mcc.astype('category').cat.categories
    mcc_dtype = CategoricalDtype(categories=mcc_categories)
    city_categories = full.city.astype('category').cat.categories
    city_dtype = CategoricalDtype(categories=city_categories)
    new_city_categories = full.new_city.astype('category').cat.categories
    new_city_dtype = CategoricalDtype(categories=new_city_categories)
    terminal_id_categories = full.terminal_id.astype('category').cat.categories
    terminal_dtype = CategoricalDtype(categories=terminal_id_categories)

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
        'n_neighbors_001',
        'n_neighbors_002',
        'n_neighbors_004'
    ]]

    df['dist'] = distance(
        df[['transaction_lat', 'transaction_lon']].values,
        df[location_columns].values
    )

    labels = (df['dist'] <= 0.02).astype(int)

    weights = 1 / (np.clip(df['dist'], a_min=0.015, a_max=10))

    return features, labels, weights

def fix_terminal_locations(terminal_transactions):
    terminal_transactions[['transaction_lat', 'transaction_lon']] = terminal_transactions[['transaction_lat', 'transaction_lon']].median().values
    return terminal_transactions

def preprocess(df):
    df = df.copy()
    df = df[df['country'] == 'RUS']
    df = df[df['currency'] == 643.0]
    df['transaction_lat'] = df['atm_address_lat'].fillna(df['pos_address_lat'])
    df = df[~df.transaction_lat.isnull()]
    df['transaction_lon'] = df['atm_address_lon'].fillna(df['pos_address_lon'])
    df['transaction_address'] = df['atm_address'].fillna(df['pos_address'])
    df[df.mcc == 6011] = df[df.mcc == 6011].groupby('terminal_id').apply(fix_terminal_locations)
    df['new_city'] = list(map(lambda location: location['admin1'], rg.search(tuple(map(tuple, df[['transaction_lat', 'transaction_lon']].values)))))

    return df.drop([
        'atm_address_lat', 'pos_address_lat',
        'atm_address_lon', 'pos_address_lon',
        'atm_address', 'pos_address',
        'country'
    ], axis=1)

def fit(target='home'):
    if target is 'home':
        custom_filter = with_home_location
        location_columns = ['home_add_lat', 'home_add_lon']
    else:
        custom_filter = with_work_location
        location_columns = ['work_add_lat', 'work_add_lon']

    train, validation, _ = get_datasets(preprocess)

    train = custom_filter(train)
    validation = custom_filter(validation)

    full = pd.concat([train, validation])
    train_x, train_y, train_weights = pipeline(full, location_columns, train)
    validation_x, validation_y, _ = pipeline(full, location_columns, validation)

    import pdb; pdb.set_trace()

    model = LGBMClassifier(categorical_feature=train_x.select_dtypes(include='category').columns.values)
    model.fit(train_x, train_y, train_weights)

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
