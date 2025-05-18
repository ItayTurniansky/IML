import numpy as np
import plotly.graph_objects as go
from sklearn import datasets
from cross_validate import cross_validate
from estimators import RidgeRegression, Lasso, LinearRegression
import matplotlib.pyplot as plt


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, y_train = X[:n_samples], y[:n_samples]
    X_test, y_test = X[n_samples:], y[n_samples:]

    # Question 2 - Perform Cross Validation for different values of the regularization parameter for Ridge and
    # Lasso regressions
    lambda_values = np.logspace(-5, 1, n_evaluations)

    ridge_train_errors = []
    ridge_val_errors = []
    lasso_train_errors = []
    lasso_val_errors = []

    for lam in lambda_values:
        ridge = RidgeRegression(lam=float(lam))
        ridge_train, ridge_val = cross_validate(ridge, X_train, y_train)
        ridge_train_errors.append(ridge_train)
        ridge_val_errors.append(ridge_val)

        lasso = Lasso(alpha=float(lam))
        lasso_train, lasso_val = cross_validate(lasso, X_train, y_train)
        lasso_train_errors.append(lasso_train)
        lasso_val_errors.append(lasso_val)

    best_ridge_idx = np.argmin(ridge_val_errors)
    best_lasso_idx = np.argmin(lasso_val_errors)
    best_ridge_lambda = float(lambda_values[best_ridge_idx])
    best_lasso_lambda = float(lambda_values[best_lasso_idx])

    ridge = RidgeRegression(lam=best_ridge_lambda)
    ridge.fit(X_train, y_train)
    ridge_test_error = ridge.loss(X_test, y_test)

    lasso = Lasso(alpha=best_lasso_lambda)
    lasso.fit(X_train, y_train)
    lasso_test_error = lasso.loss(X_test, y_test)

    ls = LinearRegression()
    ls.fit(X_train, y_train)
    ls_test_error = ls.loss(X_test, y_test)

    print(f"Best Ridge λ = {best_ridge_lambda:.5f}, Test Error = {ridge_test_error:.2f}")
    print(f"Best Lasso λ = {best_lasso_lambda:.5f}, Test Error = {lasso_test_error:.2f}")
    print(f"Least Squares Test Error = {ls_test_error:.2f}")

    # Plot Ridge
    plt.figure(figsize=(8, 5))
    plt.plot(lambda_values, ridge_train_errors, label="Ridge Train")
    plt.plot(lambda_values, ridge_val_errors, label="Ridge Validation")
    plt.axvline(best_ridge_lambda, color='gray', linestyle='--', label=f"Best λ = {best_ridge_lambda:.5f}")
    plt.xscale("log")
    plt.xlabel("λ (log scale)")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Ridge CV Errors (n_samples={n_samples})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"ridge_cv_log_nsamples{n_samples}.png")
    plt.close()

    # Plot Lasso
    plt.figure(figsize=(8, 5))
    plt.plot(lambda_values, lasso_train_errors, label="Lasso Train")
    plt.plot(lambda_values, lasso_val_errors, label="Lasso Validation")
    plt.axvline(best_lasso_lambda, color='gray', linestyle='--', label=f"Best λ = {best_lasso_lambda:.5f}")
    plt.xscale("log")
    plt.xlabel("λ (log scale)")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Lasso CV Errors (n_samples={n_samples})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"lasso_cv_log_nsamples{n_samples}.png")
    plt.close()


    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge = RidgeRegression(lam=float(best_ridge_lambda))
    ridge.fit(X_train, y_train)
    ridge_test_error = ridge.loss(X_test, y_test)

    lasso = Lasso(alpha=float(best_lasso_lambda))
    lasso.fit(X_train, y_train)
    lasso_test_error = lasso.loss(X_test, y_test)

    ls = LinearRegression()
    ls.fit(X_train, y_train)
    ls_test_error = ls.loss(X_test, y_test)

    print(f"Best Ridge λ = {best_ridge_lambda:.4f}, Test Error = {ridge_test_error:.4f}")
    print(f"Best Lasso λ = {best_lasso_lambda:.4f}, Test Error = {lasso_test_error:.4f}")
    print(f"Least Squares Test Error = {ls_test_error:.4f}")

if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()