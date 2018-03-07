import requests
import numpy as np
import pandas as pd
import reverse_geocoder as rg
from pandas.api.types import CategoricalDtype
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from scipy.spatial import ConvexHull
from retry.api import retry_call
from joblib.memory import Memory

from raiff.utils import distance
from raiff.utils import has_columns

memory = Memory(cachedir='/tmp', verbose=0)

def read(path):
    return pd.read_csv(path, parse_dates=['transaction_date'], dtype={
        'atm_address': 'object',
        'pos_address': 'object'
    })

def preprocess(df):
    df['transaction_lat'] = df['atm_address_lat'].fillna(df['pos_address_lat'])
    df['transaction_lon'] = df['atm_address_lon'].fillna(df['pos_address_lon'])
    df['transaction_address'] = df['atm_address'].fillna(df['pos_address'])

    return df.drop([
        'atm_address_lat', 'atm_address_lon', 'atm_address',
        'pos_address_lat', 'pos_address_lon', 'pos_address'
    ], axis=1)

def russia_only(df):
    df = df[df['country'].isin(['RUS', 'RU '])]
    return df.drop(['country'], axis=1)

def rouble_only(df):
    df = df[df['currency'] == 643.0]
    return df.drop(['currency'], axis=1)

def with_transaction_location(df):
    return df[df.transaction_lat.notnull()]

def with_job(df):
    return df[df.work_add_lat.notnull()]

def with_home(df):
    return df[df.work_add_lat.notnull()]

def with_columns(columns, df):
    if has_columns(columns, df):
        return df.dropna(subset=columns)
    else:
        return df

def fix_mcc(df):
    df['mcc'] = df.mcc.astype('str').str.replace(',', '').astype('float')
    return df

def with_solution(df):
    mean_target = df.groupby('customer_id')['is_close'].mean()
    customer_ids = mean_target[mean_target > 0].index.values
    return df[df.customer_id.isin(customer_ids)]

def calc_is_close(location_columns, target_columns, df):
    df['is_close'] = (distance(
        df[location_columns].values,
        df[target_columns].values
    ) <= 0.02).astype(int)

    return df

def fit_categories(column_names, df):
    mapping = {}

    for column_name in column_names:
        mapping[column_name] = df[column_name].astype('category').dtype

    return mapping

def transform_categories(mapping, df):
    for column_name, dtype in mapping.items():
        df[column_name] = df[column_name].astype(dtype)

    return df

def one_hot_encode(column_names, df):
    encoded_df = pd.get_dummies(df[column_names], dummy_na=True)
    return pd.concat([df.drop(column_names, axis=1), encoded_df], axis=1)

def fix_terminal_locations(terminal_locations, df):
    terminal_locations = pd.concat([
        terminal_locations,
        df[df.mcc == 6011][['terminal_id', 'transaction_lat', 'transaction_lon']]
    ])

    transaction_lat = terminal_locations.groupby('terminal_id')['transaction_lat'].mean()
    transaction_lon = terminal_locations.groupby('terminal_id')['transaction_lon'].mean()

    df.loc[df.mcc == 6011, 'transaction_lat'] = df[df.mcc == 6011]['terminal_id'].map(transaction_lat).fillna(df[df.mcc == 6011]['transaction_lat'])
    df.loc[df.mcc == 6011, 'transaction_lon'] = df[df.mcc == 6011]['terminal_id'].map(transaction_lon).fillna(df[df.mcc == 6011]['transaction_lon'])
    return df

def collect_terminal_locations(df):
    return df[df.mcc == 6011][['terminal_id', 'transaction_lat', 'transaction_lon']]

def decrypt_amount(df):
    df['amount'] = np.power(10, df.amount)
    return df

def reverse_city(df):
    df['reversed_city'] = list(map(lambda location: location['name'], rg.search(tuple(map(tuple, df[['transaction_lat', 'transaction_lon']].values)))))
    return df

def add_week_day(df):
    df['week_day'] = df['transaction_date'].dt.dayofweek
    return df

def add_month_day(df):
    df['month_day'] = df['transaction_date'].dt.day
    return df

def get_cluster_ids(transactions):
    model = DBSCAN(eps=0.005)
    ids = model.fit_predict(transactions[['transaction_lat', 'transaction_lon']])
    return pd.Series(index=transactions.index, data=ids)

def cluster(df):
    grouped_by_customer = df.groupby('customer_id', sort=False, as_index=False, group_keys=False)
    df['cluster_id'] = grouped_by_customer.apply(get_cluster_ids)
    return df

def calculate_cluster_features(df):
    clusters = []

    for customer_id, transactions in tqdm(df.groupby('customer_id')):
        for cluster_id, cluster_transactions in transactions.groupby('cluster_id'):
            if cluster_id == -1: continue

            cluster_median = cluster_transactions[['transaction_lat', 'transaction_lon']].median()
            amount_histogram = cluster_transactions.amount.round().value_counts(normalize=True)
            amount_histogram = amount_histogram.add_prefix('amount_hist_').to_dict()
            mcc_whitelist = [
                5411.0, 6011.0, 5814.0, 5812.0, 5499.0,
                5541.0, 5912.0, 4111.0, 5921.0, 5331.0,
                5691.0, 5261.0, 5977.0
            ]

            mcc_histogram = cluster_transactions.mcc.astype('float').astype(CategoricalDtype(categories=mcc_whitelist)).value_counts(normalize=True, dropna=False)
            mcc_histogram = mcc_histogram.add_prefix('mcc_hist_').to_dict()
            day_histogram = cluster_transactions.transaction_date.dt.dayofweek.value_counts(normalize=True).add_prefix('day_hist_').to_dict()

            try:
                # pylint: disable=no-member
                area = ConvexHull(cluster_transactions[['transaction_lat', 'transaction_lon']]).area
            # pylint: disable=broad-except
            except Exception as _:
                area = 0

            # TODO AS: Might reconsider this later
            first_transaction = transactions.iloc[0]

            features = {
                'cluster_id': cluster_id,
                'customer_id': customer_id,
                'cluster_lat': cluster_median['transaction_lat'],
                'cluster_lon': cluster_median['transaction_lon'],
                'home_add_lat': first_transaction['home_add_lat'],
                'home_add_lon': first_transaction['home_add_lon'],
                'work_add_lat': first_transaction['work_add_lat'],
                'work_add_lon': first_transaction['work_add_lon'],
                'area': area,
                'transaction_ratio': len(cluster_transactions) / len(transactions),
                'amount_ratio': np.sum(np.exp(cluster_transactions.amount)) / np.sum(np.exp(transactions.amount)),
                'date_ratio': len(cluster_transactions.transaction_date.unique()) / len(transactions.transaction_date.unique()),
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

            clusters.append(features)

    return pd.DataFrame(clusters)

def merge_cluster_features(df):
    # TODO AS: -1 clusters are ignored
    clusters = calculate_cluster_features(df)
    df = pd.merge(df, clusters, how='left', on=['customer_id', 'cluster_id'])
    return df[df.cluster_id != -1].copy()

@memory.cache
def query_osm(query):
    # https://overpass.kumi.systems/api/interpreter
    # https://overpass-api.de/api/interpreter
    # http://localhost:12345/api/interpreter
    response = requests.post('http://localhost:12345/api/interpreter', data={'data': query})
    response.raise_for_status()
    return response.json()['elements']

def query_surrounding_map(location_columns, df):
    surroundings = ['atm', 'shop', 'apartment', 'industrial', 'natural']

    for column_name in surroundings:
        df[column_name] = 0

    box_size = 0.04
    half = box_size / 2

    def is_shop(record):
        tags = record.get('tags', {})
        return 'shop' in tags

    def is_natural(record):
        tags = record.get('tags', {})
        return 'natural' in tags

    def is_apartment(record):
        tags = record.get('tags', {})
        return ('building' in tags) and (tags['building'] == 'apartments')

    def is_industrial(record):
        tags = record.get('tags', {})
        return ('building' in tags) and (tags['building'] == 'industrial')

    def is_atm(record):
        tags = record.get('tags', {})
        if 'atm' in tags:
            return True

        if 'amenity' in tags:
            if tags['amenity'] == 'bank':
                return True

            if tags['amenity'] == 'atm':
                return True

        return False

    for index, row in tqdm(df.iterrows(), total=len(df)):
        lat, lon = row[location_columns]

        query = f"""
            [out:json][bbox:{lat - half},{lon - half},{lat + half},{lon + half}];
            (
                node["atm"];
                node["amenity"="atm"];
                node["amenity"="bank"];
                way["atm"];
                node["natural"];
                way["building"="apartments"];
                way["building"="industrial"];
                way["shop"];
            );
            (._;>;);
            out body geom;
        """

        map_objects = retry_call(query_osm, fargs=[query], tries=3, delay=10)

        df.loc[index, surroundings] = [
            len(list(filter(is_atm, map_objects))),
            len(list(filter(is_shop, map_objects))),
            len(list(filter(is_apartment, map_objects))),
            len(list(filter(is_industrial, map_objects))),
            len(list(filter(is_natural, map_objects)))
        ]

    return df
