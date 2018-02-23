from datetime import datetime

import numpy as np
import pandas as pd

def distance(first, second):
    return np.sqrt(np.sum((first - second) ** 2, axis=1))

def generate_model_name(model, score):
    timestr = datetime.utcnow().strftime('%Y%m%d_%H%M')
    model_name = type(model).__name__
    return f'{model_name}-{timestr}-{score:.5f}.model'

def train_validation_holdout_split(df, seed=11):
    np.random.seed(seed)
    grouped_by_customer = df.groupby('customer_id')
    groups = list(map(lambda record: record[1], grouped_by_customer))
    np.random.shuffle(groups)
    group_count = len(groups)
    train = pd.concat(groups[0:int(group_count * 0.7)])
    validation = pd.concat(groups[int(group_count * 0.7):int(group_count * 0.85)])
    holdout = pd.concat(groups[int(group_count * 0.85):])
    return train, validation, holdout
