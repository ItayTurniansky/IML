import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from plotly.graph_objects import Figure
from adaboost import AdaBoost
from decision_stump import DecisionStump
from matplotlib.colors import ListedColormap


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, iterations=n_learners)
    model.fit(train_X, train_y)

    train_errors = []
    for t in range(1, n_learners + 1):
        error = model.partial_loss(train_X, train_y, t)
        train_errors.append(error)

    test_errors = []
    for t in range(1, n_learners + 1):
        error = model.partial_loss(test_X, test_y, t)
        test_errors.append(error)


    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_learners + 1), train_errors, label='Train Error')
    plt.plot(range(1, n_learners + 1), test_errors, label='Test Error')
    plt.xlabel("Number of Learners")
    plt.ylabel("Misclassification Error")
    plt.title(f"Train and Test Error vs. Number of AdaBoost Iterations (Noise={noise})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"question1_train_test_error_noise_{noise}.png")
    plt.close()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    xx, yy = np.meshgrid(
        np.linspace(lims[0][0], lims[0][1], 500),
        np.linspace(lims[1][0], lims[1][1], 500)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, t in enumerate(T):
        zz = model.partial_predict(np.c_[xx.ravel(), yy.ravel()], t).reshape(xx.shape)

        axes[i].contourf(xx, yy, zz, levels=1, alpha=0.3, cmap=ListedColormap(['red', 'blue']))
        axes[i].scatter(test_X[:, 0], test_X[:, 1], c=test_y, cmap=ListedColormap(['red', 'blue']), s=10, edgecolor='k')
        axes[i].set_title(f"Decision Boundary with {t} Learners")
        axes[i].set_xlim(lims[0])
        axes[i].set_ylim(lims[1])
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.suptitle("AdaBoost Decision Boundaries at Different Iterations", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"question2_boundaries_noise_{noise}.png")
    plt.close()

    # Question 3: Decision surface of best performing ensemble
    best_T = np.argmin(test_errors) + 1
    best_accuracy = 1 - test_errors[best_T - 1]
    xx, yy = np.meshgrid(
        np.linspace(lims[0][0], lims[0][1], 500),
        np.linspace(lims[1][0], lims[1][1], 500)
    )
    zz = model.partial_predict(np.c_[xx.ravel(), yy.ravel()], best_T).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, levels=1, alpha=0.3, cmap=ListedColormap(['red', 'blue']))
    plt.scatter(test_X[:, 0], test_X[:, 1], c=test_y, cmap=ListedColormap(['red', 'blue']), s=10, edgecolor='k')
    plt.title(f"Best Ensemble (T={best_T})\nAccuracy = {best_accuracy:.2%}")
    plt.xticks([])
    plt.yticks([])
    plt.xlim(lims[0])
    plt.ylim(lims[1])
    plt.tight_layout()
    plt.savefig(f"question3_best_boundary_T{best_T}_acc_{best_accuracy:.2f}_noise_{noise}.png")
    plt.close()

    # Question 4: Decision surface with weighted samples
    final_D = model.D_[-1]
    D_scaled = final_D / np.max(final_D) * 5

    x_min, x_max = lims[0]
    y_min, y_max = lims[1]
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, levels=1, cmap=ListedColormap(['red', 'blue']), alpha=0.3)
    plt.scatter(train_X[:, 0], train_X[:, 1],
                c=train_y,
                cmap=ListedColormap(['red', 'blue']),
                s=D_scaled,
                edgecolor='k', linewidth=0.5)

    plt.title("Training Sample Weights (Final Iteration)\nSize ∝ Difficulty")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"question4_weighted_points_noise_{noise}.png")
    plt.close()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)