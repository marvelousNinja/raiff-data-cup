import numpy as np
import pandas as pd
import reverse_geocoder as rg

from raiff.utils import distance
from raiff.utils import has_columns

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

def calc_is_close(df):
    df['is_close'] = (distance(
        df[['transaction_lat', 'transaction_lon']].values,
        df[['work_add_lat', 'work_add_lon']].values
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
