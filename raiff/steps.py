import pandas as pd
from sklearn.cluster import DBSCAN

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
    df = df[df['currency'] == 643.0]
    df = df[df.transaction_lat.notnull()]
    return df.drop(['country', 'currency'], axis=1)

def with_job(df):
    return df[df.work_add_lat.notnull()]

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
