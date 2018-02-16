import numpy as np
import pandas as pd

def get_datasets(seed=11):
    np.random.seed(seed)
    grouped_by_customer = pd.read_csv('./data/train_set.csv').groupby('customer_id')
    groups = list(map(lambda record: record[1], grouped_by_customer))
    np.random.shuffle(groups)
    group_count = len(groups)

    train = pd.concat(groups[0:int(group_count * 0.7)])
    test = pd.concat(groups[int(group_count * 0.7):int(group_count * 0.85)])
    holdout = pd.concat(groups[int(group_count * 0.85):])
    return train, test, holdout
