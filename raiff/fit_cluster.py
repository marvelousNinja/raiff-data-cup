import numpy as np
import pandas as pd
from tqdm import tqdm
from fire import Fire
from sklearn.cluster import DBSCAN
from lightgbm.sklearn import LGBMClassifier

from raiff.utils import train_validation_holdout_split

# TODO AS: Make it work for both vectors and single values
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
            work_location = cluster_transactions.iloc[0][['work_add_lat', 'work_add_lon']].values
            is_close = int(distance(cluster_median, work_location) <= 0.02)

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
    train = train[train['country'].isin(['RUS'])] # drops 4,000 samples
    train = train[train['currency'] == 643.0] # drops 7,143 samples
    train = train[~train.transaction_lat.isnull()] # drops 97,000 samples
    train = train.drop([
        'country', 'currency'
    ], axis=1)

    # 3. Task specific filtering
    train = train[~train.work_add_lat.isnull()] # drops 512,269 samples

    # 4. Train, validation, holdout split
    train, validation, _ = train_validation_holdout_split(train)

    train_clusters = as_clusters(train)
    validation_clusters = as_clusters(validation)

    train_x = train_clusters[['amount_spent', 'date_ratio', 'top_mcc', 'transaction_ratio']]
    train_y = train_clusters['is_close']
    validation_x = validation_clusters[['amount_spent', 'date_ratio', 'top_mcc', 'transaction_ratio']]
    validation_y = validation_clusters['is_close']

    model = LGBMClassifier(categorical_feature=[2])
    model.fit(train_x, train_y)

    validation_clusters['probs'] = model.predict_proba(validation_x)[:, 1]
    predictions = validation_clusters.groupby('customer_id').apply(lambda group: group.sort_values('probs', ascending=False).head(1))
    score = predictions['is_close'].mean()
    print(score)

if __name__ == '__main__':
    Fire(fit)
