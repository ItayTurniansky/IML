from typing import Tuple
import numpy as np
from base_estimator import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data. Has functions: fit, predict, loss

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    n_samples = X.shape[0]
    fold_sizes = np.full(cv, n_samples // cv)
    fold_sizes[:n_samples % cv] +=1
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    current = 0
    train_scores = []
    val_scores = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        estimator.fit(X_train, y_train)
        train_scores.append(estimator.loss(X_train, y_train))
        val_scores.append(estimator.loss(X_val, y_val))
        current = stop
    return float(np.mean(train_scores)), float(np.mean(val_scores))