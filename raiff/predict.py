import numpy as np
from fire import Fire
from sklearn.externals.joblib import load

from raiff.datasets import get_datasets
from raiff.pipelines import train_pipeline

def predict(path='./data/models/_latest.model'):
    _, validation, _ = get_datasets()
    validation = train_pipeline(validation)

    x = validation[['amount', 'mcc']]

    model = load(path)
    predictions = model.predict_proba(x)

    validation['work_probs'] = predictions[0][:, 1]
    validation['home_probs'] = predictions[1][:, 1]

    correct = 0
    total = 0

    for _, group in validation.groupby('customer_id'):
        top_home_prediction = group.sort_values('home_probs', ascending=False).iloc[0]
        top_work_prediction = group.sort_values('work_probs', ascending=False).iloc[0]
        predicted_home = top_home_prediction[['transaction_lat', 'transaction_lon']].values
        predicted_work = top_work_prediction[['transaction_lat', 'transaction_lon']].values
        true_home = group.iloc[0][['home_add_lat', 'home_add_lon']].values
        true_work = group.iloc[0][['work_add_lat', 'work_add_lon']].values

        home_distance = np.sqrt(np.sum((predicted_home - true_home) ** 2))
        work_distance = np.sqrt(np.sum((predicted_work - true_work) ** 2))

        if home_distance <= 0.02:
            correct += 1

        if work_distance <= 0.02:
            correct += 1

        total += 2

    print(f'Acc: {correct / total}')


if __name__ == '__main__':
    Fire(predict)
