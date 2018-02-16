import numpy as np
from fire import Fire
from lightgbm.sklearn import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals.joblib import dump
from sklearn.metrics import accuracy_score

from raiff.datasets import get_datasets
from raiff.pipelines import train_pipeline
from raiff.utils import generate_model_name

def fit():
    train, validation, _ = get_datasets()
    train = train_pipeline(train)
    validation = train_pipeline(validation)

    x = train[['amount', 'mcc']]
    y = np.column_stack([
        (train['work_distance'] <= 0.02).astype(int),
        (train['home_distance'] <= 0.02).astype(int)
    ])

    x_val = validation[['amount', 'mcc']]
    y_val = np.column_stack([
        (validation['work_distance'] <= 0.02).astype(int),
        (validation['home_distance'] <= 0.02).astype(int)
    ])

    model = MultiOutputClassifier(LGBMClassifier(categorical_feature=[1]))
    model.fit(x, y)

    predictions = model.predict(x_val)
    score = accuracy_score(y_val, predictions)

    name = generate_model_name(model, score)
    dump(model, f'./data/models/{name}')
    dump(model, f'./data/models/_latest.model')
    print(f'Model saved to ./data/models/{name}')

if __name__ == '__main__':
    Fire(fit)
