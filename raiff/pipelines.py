import numpy as np

def train_pipeline(df):
    # Russia only
    df = df[df['country'] == 'RUS']
    # Russian Ruble only
    df = df[df['currency'] == 643.0]
    df = df.drop(['country', 'currency'], axis=1)

    # Unify columns for POS terminals and ATMs
    df['transaction_lat'] = (df['atm_address_lat'].fillna(0) + df['pos_adress_lat'].fillna(0)).replace({0: np.NaN})
    df['transaction_lon'] = (df['atm_address_lon'].fillna(0) + df['pos_adress_lon'].fillna(0)).replace({0: np.NaN})
    df['transaction_address'] = (df['atm_address'].fillna('') + df['pos_address'].fillna('')).replace({'': np.NaN})
    df['is_atm'] = ~df['atm_address_lon'].isnull()

    # Convert MCC to strings
    # df['mcc'] = df['mcc'].astype(str)
    # df.loc[df['mcc'] == '5411', 'mcc'] = 0
    # df.loc[df['mcc'] == '6011', 'mcc'] = 1
    # df.loc[df['mcc'] == '5814', 'mcc'] = 2
    # df.loc[df['mcc'] == '5812', 'mcc'] = 3
    # df.loc[df['mcc'] == '5541', 'mcc'] = 4
    # df.loc[df['mcc'] == '5499', 'mcc'] = 5
    # df.loc[df['mcc'] == '5912', 'mcc'] = 6
    # df.loc[df['mcc'] == '4111', 'mcc'] = 7
    # df.loc[df['mcc'] == '5921', 'mcc'] = 8
    # df.loc[~df['mcc'].isin([
    #     '5411', '6011', '5814',
    #     '5812', '5541', '5499',
    #     '5912', '4111', '5921'
    # ]), 'mcc'] = 100

    df['mcc'] = df['mcc'].astype('category')
    df['terminal_id'] = df['terminal_id'].astype('category')

    # df['amount'] = np.exp(7.9 - df['amount']) + 20

    # Filter entries with missing location data
    # TODO AS: Half transactions just disappeared
    df = df[~df.transaction_lat.isnull()]
    df = df[~df.work_add_lat.isnull()]
    df = df[~df.home_add_lat.isnull()]

    df['work_distance'] = np.sqrt(np.sum((df[['transaction_lat', 'transaction_lon']].values - df[['work_add_lat', 'work_add_lon']].values) ** 2, axis=1))
    df['home_distance'] = np.sqrt(np.sum((df[['transaction_lat', 'transaction_lon']].values - df[['home_add_lat', 'home_add_lon']].values) ** 2, axis=1))

    df = df.drop([
        'atm_address', 'atm_address_lat', 'atm_address_lon',
        'pos_address', 'pos_adress_lat', 'pos_adress_lon'
    ], axis=1)

    # Index(['amount', 'atm_address', 'atm_address_lat', 'atm_address_lon', 'city',
    #    'country', 'currency', 'customer_id', 'home_add_lat', 'home_add_lon',
    #    'mcc', 'pos_address', 'pos_adress_lat', 'pos_adress_lon', 'terminal_id',
    #    'transaction_date', 'work_add_lat', 'work_add_lon'],

    return df

def test_pipeline(df):
    # Russia only
    df = df[df['country'] == 'RUS']
    # Russian Ruble only
    df = df[df['currency'] == 643.0]
    df = df.drop(['country', 'currency'], axis=1)

    # Unify columns for POS terminals and ATMs
    df['transaction_lat'] = (df['atm_address_lat'].fillna(0) + df['pos_address_lat'].fillna(0)).replace({0: np.NaN})
    df['transaction_lon'] = (df['atm_address_lon'].fillna(0) + df['pos_address_lon'].fillna(0)).replace({0: np.NaN})
    df['transaction_address'] = (df['atm_address'].fillna('') + df['pos_address'].fillna('')).replace({'': np.NaN})
    df['is_atm'] = ~df['atm_address_lon'].isnull()

    # Convert MCC to the training data format
    df['mcc'] = df['mcc'].astype(str).map(lambda code: code.replace(',', ''))
    df.loc[df['mcc'] == '5411', 'mcc'] = 0
    df.loc[df['mcc'] == '6011', 'mcc'] = 1
    df.loc[df['mcc'] == '5814', 'mcc'] = 2
    df.loc[df['mcc'] == '5812', 'mcc'] = 3
    df.loc[df['mcc'] == '5541', 'mcc'] = 4
    df.loc[df['mcc'] == '5499', 'mcc'] = 5
    df.loc[df['mcc'] == '5912', 'mcc'] = 6
    df.loc[df['mcc'] == '4111', 'mcc'] = 7
    df.loc[df['mcc'] == '5921', 'mcc'] = 8
    df.loc[~df['mcc'].isin([
        '5411', '6011', '5814',
        '5812', '5541', '5499',
        '5912', '4111', '5921'
    ]), 'mcc'] = 100

    df['mcc'] = df['mcc'].astype('category')

    df = df.drop([
        'atm_address', 'atm_address_lat', 'atm_address_lon',
        'pos_address', 'pos_address_lat', 'pos_address_lon'
    ], axis=1)

    return df
