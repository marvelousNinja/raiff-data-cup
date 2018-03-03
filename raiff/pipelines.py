from functools import partial

def pipeline(transforms, X):
    X = X.copy()

    for transform in transforms:
        X = transform(X)

    return X

def fit_pipeline(steps, X):
    X = X.copy()
    transforms = []

    for step in steps:
        if isinstance(step, tuple):
            fit, transform = step
            state = fit(X)
            transform = partial(transform, state)
            transforms.append(transform)
            X = transform(X)
        else:
            transforms.append(step)
            X = step(X)

    return partial(pipeline, transforms), X
