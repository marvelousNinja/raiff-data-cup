from functools import partial

import numpy as np
from fire import Fire
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from raiff.pipelines import fit_pipeline
from raiff.steps import read
from raiff.steps import preprocess
from raiff.steps import russia_only
from raiff.steps import rouble_only
from raiff.steps import with_job
from raiff.steps import with_transaction_location
from raiff.steps import fit_categories
from raiff.steps import transform_categories
from raiff.steps import calc_is_close
from raiff.utils import train_validation_holdout_split

def fit():
    train, validation, _ = train_validation_holdout_split(read('./data/train_set.csv'))

    steps = [
        preprocess,
        russia_only,
        rouble_only,
        with_transaction_location,
        with_job,
        (partial(fit_categories, ['mcc', 'city', 'terminal_id']), transform_categories),
        calc_is_close
    ]

    pipeline = fit_pipeline(steps, train)
    train = pipeline(train)
    validation = pipeline(validation)

    feature_columns = ['mcc', 'city', 'amount', 'terminal_id']
    print(f'Train size: {len(train)}, Validation size: {len(validation)}')
    print(f'Features: {feature_columns}')
    model = LGBMClassifier()
    model.fit(train[feature_columns], train['is_close'])

    predictions = model.predict_proba(validation[feature_columns])
    accuracy_value = accuracy_score(validation['is_close'], np.argmax(predictions, axis=1))
    logloss_value = log_loss(validation['is_close'], predictions)
    print(f'Accuracy: {accuracy_value:.5f}, Logloss: {logloss_value:.5f}')
    print(classification_report(validation['is_close'], np.argmax(predictions, axis=1)))

    validation['probs'] = predictions[:, 1]
    top1_accuracy = validation.groupby('customer_id').apply(lambda group: group.sort_values('probs').tail(1).is_close.max()).mean()
    top5_accuracy = validation.groupby('customer_id').apply(lambda group: group.sort_values('probs').tail(5).is_close.max()).mean()
    top10_accuracy = validation.groupby('customer_id').apply(lambda group: group.sort_values('probs').tail(10).is_close.max()).mean()
    print(f'Top1: {top1_accuracy:.5f}')
    print(f'Top5: {top5_accuracy:.5f}')
    print(f'Top10: {top10_accuracy:.5f}')

    # contributions = model._Booster.predict(validation[feature_columns], pred_contrib=True)
    # contributions_df = pd.DataFrame(
    #     index=validation.index,
    #     data=contributions,
    #     columns=list(map(lambda col: col + '_contr', feature_columns)) + ['expected_value']
    # )

    # debug_df = pd.concat([validation, contributions_df], axis=1)
    # debug_df.index.name = 'id'
    # debug_df.to_csv('./data/debug.csv')

    import pdb; pdb.set_trace()

if __name__ == '__main__':
    Fire(fit)
