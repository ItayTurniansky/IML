import numpy as np
from typing import Callable, NoReturn

from IML_ex3_files.decision_stump import DecisionStump
from base_estimator import BaseEstimator
from loss_functions import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations

    self.weights_: List[float]
        List of weights for each fitted estimator, fitted along the boosting iterations

    self.D_: List[np.ndarray]
        List of weights for each sample, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = [], [], []

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples = X.shape[0]
        D = np.ones(n_samples) / n_samples  # uniform init

        for t in range(self.iterations_):
            #Resample according to D
            indices = np.random.choice(n_samples, size=n_samples, replace=True, p=D)
            X_resampled = X[indices]
            y_resampled = y[indices]

            # Train weak learner
            stump = DecisionStump()
            stump.fit(X_resampled, y_resampled)

            #Compute error on resampled data
            y_pred_resampled = stump.predict(X_resampled)
            error = misclassification_error(y_resampled, y_pred_resampled)

            if error == 0:
                alpha = np.inf
            else:
                alpha = 0.5 * np.log((1 - error) / error)

            # Predict on full data to update D
            y_pred_full = stump.predict(X)
            D *= np.exp(-alpha * y * y_pred_full)
            D /= np.sum(D)  # normalize

            # Save model, weight, distribution
            self.models_.append(stump)
            self.weights_.append(alpha)
            self.D_.append(D.copy())

            print(f"Fitting progress: {100 * (t + 1) / self.iterations_:.1f}%", flush=True)

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        n_samples = X.shape[0]
        agg_preds = np.zeros(n_samples)

        for model, weight in zip(self.models_, self.weights_):
            agg_preds += weight * model.predict(X)

        return np.sign(agg_preds)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        n_samples = X.shape[0]
        agg_preds = np.zeros(n_samples)

        for i, (model, weight) in enumerate(zip(self.models_[:T], self.weights_[:T])):
            agg_preds += weight * model.predict(X)
            print(f"Partial predict progress: {100 * (i + 1) / T:.1f}%")

        return np.sign(agg_preds)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        print(f"Evaluating loss at T={T}")
        return misclassification_error(y, self.partial_predict(X, T))
