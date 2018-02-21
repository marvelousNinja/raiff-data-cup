import numpy as np
import pandas as pd
import reverse_geocoder as rg
from tqdm import tqdm
from fire import Fire
from sklearn.cluster import DBSCAN
from lightgbm.sklearn import LGBMClassifier

from raiff.datasets import get_datasets

def generate_customer_features(transactions):
    model = DBSCAN(eps=0.01)
    transactions['cluster_id'] = model.fit_predict(transactions[['transaction_lat', 'transaction_lon']])
    return transactions

def preprocess(df):
    df = df[df['country'].isin(['RUS', 'RU'])]
    df = df[df['currency'] == 643.0]

    df['transaction_lat'] = (df['atm_address_lat'].fillna(0) + df['pos_address_lat'].fillna(0)).replace({0: np.NaN})
    df['transaction_lon'] = (df['atm_address_lon'].fillna(0) + df['pos_address_lon'].fillna(0)).replace({0: np.NaN})
    df['transaction_address'] = (df['atm_address'].fillna('') + df['pos_address'].fillna('')).replace({'': np.NaN})
    df = df[~df.transaction_lat.isnull()]
    # df['new_city'] = list(map(lambda location: location['admin1'], rg.search(tuple(map(tuple, df[['transaction_lat', 'transaction_lon']].values)))))

    return df.drop([
        'atm_address_lat', 'pos_address_lat',
        'atm_address_lon', 'pos_address_lon',
        'atm_address', 'pos_address',
        'country'
    ], axis=1)

def distance(first, second):
    return np.sqrt(np.sum((first - second) ** 2))

def as_clusters(df):
    clusters = []
    for customer_id, customer_transactions in tqdm(df.groupby(['customer_id'])):
        customer_transactions = customer_transactions.copy()
        model = DBSCAN(eps=0.01)
        customer_transactions['cluster_id'] = model.fit_predict(customer_transactions[['transaction_lat', 'transaction_lon']])

        for cluster_id, cluster_transactions in customer_transactions.groupby(['cluster_id']):
            cluster_median = cluster_transactions[['transaction_lat', 'transaction_lon']].median().values
            home_location = cluster_transactions.iloc[0][['work_add_lat', 'work_add_lon']].values
            is_close = int(distance(cluster_median, home_location) <= 0.02)

            clusters.append({
                'customer_id': customer_id,
                'cluster_id': cluster_id,
                'top_mcc': cluster_transactions['mcc'].mode()[0],
                'amount_spent': cluster_transactions['amount'].sum(),
                'transaction_ratio': len(cluster_transactions) / len(customer_transactions),
                'date_ratio': len(cluster_transactions.transaction_date.unique()) / len(customer_transactions.transaction_date.unique()),
                'is_close': is_close
            })

    return pd.DataFrame(clusters)

def cluster():
    train, validation, _ = get_datasets(preprocess)
    train = train[~train.work_add_lat.isnull()]
    validation = validation[~validation.work_add_lat.isnull()]

    train_clusters = as_clusters(train[:40000])
    validation_clusters = as_clusters(validation[:40000])

    train_x = train_clusters[['amount_spent', 'date_ratio', 'top_mcc', 'transaction_ratio']]
    train_y = train_clusters['is_close']
    validation_x = validation_clusters[['amount_spent', 'date_ratio', 'top_mcc', 'transaction_ratio']]
    validation_y = validation_clusters['is_close']

    model = LGBMClassifier(categorical_feature=[2])
    model.fit(train_x, train_y)

    import pdb; pdb.set_trace()

    validation_clusters['probs'] = model.predict_proba(validation_x)[:, 1]
    predictions = validation_clusters.groupby('customer_id').apply(lambda group: group.sort_values('probs', ascending=False).head(1))
    score = predictions['is_close'].mean()
    print(score)

if __name__ == '__main__':
    Fire(cluster)
