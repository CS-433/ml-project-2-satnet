from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np


def optimize_logistic_regression(X, y):
    param_grid = [
        {
            'penalty': ['l1', 'l2'],
            'C': np.logspace(-6, 6, 10),
            'solver': ['liblinear'],
            'max_iter': [100, 1000]
        },
        {
            'penalty': ['l2'],
            'C': np.logspace(-6, 6, 10),
            'solver': ['lbfgs', 'sag', 'newton-cg'],
            'max_iter': [100, 1000]
        },
        {
            'penalty': ['elasticnet'],
            'C': np.logspace(-6, 6, 10),
            'solver': ['saga'],
            'l1_ratio': [0.1, 0.5, 0.9],
            'max_iter': [100, 1000]
        },
        {
            'penalty': ['none'],
            'solver': ['lbfgs', 'sag', 'newton-cg'],
            'max_iter': [100, 1000]
        }
    ]
    clf = GridSearchCV(
        LogisticRegression(class_weight='balanced'),
        param_grid=param_grid,
        cv=10,
        scoring='f1',
        verbose=1,
        n_jobs=-1
    )

    clf.fit(X, y)

    best_model = clf.best_estimator_
    best_params = clf.best_params_
    best_f1 = clf.best_score_

    return best_model, best_params, best_f1