import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from base_learning_rate import  BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from cross_validate import cross_validate



# from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test

import plotly.graph_objects as go

def save_simple_descent_plot(weights: List[np.ndarray], title: str):
    if len(weights) == 0:
        print(f"[Warning] No weights recorded for {title}")
        return

    weights = np.array(weights)
    if weights.ndim == 1:
        weights = np.expand_dims(weights, axis=0)

    plt.figure(figsize=(6, 6))
    plt.plot(weights[:, 0], weights[:, 1], marker="o", markersize=3, linewidth=1)
    plt.title(title)
    plt.xlabel("w[0]")
    plt.ylabel("w[1]")
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(title + ".png")



def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []
    def callback(**kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"].copy())

    return callback, values, weights


def plot_convergence_curves(convergence_dict, title, filename):
    plt.figure(figsize=(10, 6))
    for eta, values in convergence_dict.items():
        plt.plot(range(len(values)), values, label=f"η = {eta}")
    plt.yscale("linear")
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title(f"{title} Module - Convergence Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):

    l1_convergence = {}
    l2_convergence = {}

    ##1+2##
    for eta in etas:
        learning_rate = FixedLR(eta)
        module1 = L1(init.copy())
        callback1, values1, weights1 = get_gd_state_recorder_callback()
        gd1 = GradientDescent(learning_rate = learning_rate, callback = callback1)
        gd1.fit(f = module1, X= None, y=None)
        l1_convergence[eta] = values1
        fig1 = plot_descent_path(L1, np.array(weights1), title=f"L1 Descent Path (eta={eta})")
        fig1.write_html(f"L1_descent_eta_{eta}_fancy.html")

        module2= L2(init.copy())
        callback2, values2, weights2 = get_gd_state_recorder_callback()
        gd2 = GradientDescent(learning_rate=learning_rate, callback=callback2)
        gd2.fit(f=module2, X=None, y=None)
        l2_convergence[eta] = values2
        fig2 = plot_descent_path(L2, np.array(weights2), title=f"L2 Descent Path (eta={eta})")
        fig2.write_html(f"L2_descent_eta_{eta}_fancy.html")

    ##3##
    plot_convergence_curves(l1_convergence, "L1", "L1_convergence.png")
    plot_convergence_curves(l2_convergence, "L2", "L2_convergence.png")

    ##4##
    print("Part 1 === GD ===")
    print("L1:")
    for eta, vals in l1_convergence.items():
        print(f"  η = {eta}: min loss = {min(vals):.9f}")

    print("L2:")
    for eta, vals in l2_convergence.items():
        print(f"  η = {eta}: min loss = {min(vals):.9f}")


def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def misclassification_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


def fit_logistic_regression():
    print("Part 2 === Logistic Regression ===")

    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Plotting convergence rate of logistic regression over SA heart disease data
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    lr = LogisticRegression(include_intercept=True, penalty="none")
    lr._fit(X_train, y_train)
    y_pred_proba = lr.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    alpha_star = thresholds[best_idx]
    print(f"Optimal α* = {alpha_star:.4f}")
    y_pred = (y_pred_proba >= alpha_star).astype(int)
    test_error = np.mean(y_pred != y_test)
    print(f"Test error with α* = {alpha_star:.4f}: {test_error:.4f}")


    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC Curve", color='blue')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression (No Regularization)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logistic_regression_roc.png")
    plt.show()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    best_lambda = None
    best_val_score = float('inf')
    for lam in lambdas:
        solver = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
        estimator = LogisticRegression(
            include_intercept=True,
            solver=solver,
            penalty="l1",
            lam=lam,
            alpha=0.5
        )
        train_score, val_score = cross_validate(estimator, X_train, y_train,
                                                scoring=misclassification_error, cv=5)
        print(f"λ = {lam}: train error = {train_score:.4f}, validation error = {val_score:.4f}")
        if val_score < best_val_score:
            best_val_score = val_score
            best_lambda = lam

    print(f"Best λ : {best_lambda}")

    final_solver = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
    final_model = LogisticRegression(
        include_intercept=True,
        solver=final_solver,
        penalty="l1",
        lam=best_lambda,
        alpha=0.5
    )
    final_model._fit(X_train, y_train)
    test_error = final_model._loss(X_test, y_test)
    print(f"Test error using best λ = {best_lambda}: {test_error:.4f}")

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
