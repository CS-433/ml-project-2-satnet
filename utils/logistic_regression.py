"""
Helper functions which are linked to logistic regression
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, log_loss, f1_score
import numpy as np
from utils.helpers import *


def feature_extraction(img_patches):
    """
    Extracts features from image patches using multiple feature extraction methods.

    Parameters:
    img_patches (list or np.ndarray): A list or array of image patches (each patch is a sub-image).
                                        The patches should have been extracted from a larger image.

    Returns:
    tuple: A tuple containing three elements:
        - X2d (np.ndarray): A 2D numpy array of features extracted using the 2D method.
        - X6d (np.ndarray): A 2D numpy array of features extracted using the 6D method.
        - X8d (np.ndarray): A 2D numpy array where each feature vector is a concatenation of the 2D and 6D features.
    """
    X2d = np.asarray(
        [extract_features_2d(img_patches[i]) for i in range(len(img_patches))]
    )
    X6d = np.asarray(
        [extract_features(img_patches[i]) for i in range(len(img_patches))]
    )
    X8d = np.asarray([np.concatenate((X2d[i], X6d[i])) for i in range(len(X2d))])
    print(
        "Dimensions of the three different feature extraction methods (2D,6D,8D): {},{} and {}\n".format(
            X2d.shape, X6d.shape, X8d.shape
        )
    )
    return X2d, X6d, X8d


def label_extraction(gt_patches, foreground_threshold=0.25):
    """
    Extracts binary labels for image patches based on the foreground threshold.

    Parameters:
    gt_patches (list or np.ndarray): A list or array of ground truth patches. Each patch is a sub-region
                                      of a larger image, with pixel values representing the ground truth mask.
    foreground_threshold (float, optional): The threshold value used to classify patches as foreground
                                            or background based on the mean pixel value of the patch.
                                            Defaults to 0.25.

    Returns:
    np.ndarray: A 1D numpy array of binary labels (`0` or `1`) for each patch, where `1` represents foreground
                and `0` represents background.
    """
    y = np.asarray(
        [
            value_to_class(np.mean(gt_patches[i]), foreground_threshold=0.25)
            for i in range(len(gt_patches))
        ]
    )
    return y


def optimize_logistic_regression(X, y):
    """
    Extracts binary labels for image patches based on the foreground threshold.

    Parameters:
    gt_patches (list or np.ndarray): A list or array of ground truth patches. Each patch is a sub-region
                                      of a larger image, with pixel values representing the ground truth mask.
    foreground_threshold (float, optional): The threshold value used to classify patches as foreground
                                            or background based on the mean pixel value of the patch.
                                            Defaults to 0.25.

    Returns:
    np.ndarray: A 1D numpy array of binary labels (`0` or `1`) for each patch, where `1` represents foreground
                and `0` represents background.
    """
    param_grid = [
        {
            "penalty": ["l1", "l2"],
            "C": np.logspace(-6, 6, 10),
            "solver": ["liblinear"],
            "max_iter": [100, 1000],
        },
        {
            "penalty": ["l2"],
            "C": np.logspace(-6, 6, 10),
            "solver": ["lbfgs", "sag", "newton-cg"],
            "max_iter": [100, 1000],
        },
        {
            "penalty": ["elasticnet"],
            "C": np.logspace(-6, 6, 10),
            "solver": ["saga"],
            "l1_ratio": [0.1, 0.5, 0.9],
            "max_iter": [100, 1000],
        },
        {
            "penalty": ["none"],
            "solver": ["lbfgs", "sag", "newton-cg"],
            "max_iter": [100, 1000],
        },
    ]
    log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    clf = GridSearchCV(
        LogisticRegression(class_weight="balanced"),
        param_grid=param_grid,
        cv=10,
        scoring=log_loss_scorer,
        verbose=1,
        n_jobs=-1,
    )

    clf.fit(X, y)

    best_model = clf.best_estimator_
    best_params = clf.best_params_
    best_f1 = clf.best_score_

    return best_model, best_params, best_f1


def find_best_treshold(model, x, y, tresholds):
    """
    Find the best classification threshold for a given model by maximizing the F1 score.

    Parameters:
    model (sklearn.model): A trained classifier with a `predict_proba` method.
    x (numpy.ndarray or pd.DataFrame): The input features to make predictions on (shape: n_samples, n_features).
    y (numpy.ndarray or pd.Series): The true target labels (shape: n_samples).
    tresholds (list or numpy.ndarray): A list or array of thresholds to test for classification.

    Returns:
    float: The threshold value that maximizes the F1 score.
    """
    best_threshold = None
    best_f1 = -1

    probabilities = model.predict_proba(x)[:, 1]

    for threshold in tresholds:
        y_pred = (probabilities > threshold).astype(int)
        f1 = f1_score(y, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
