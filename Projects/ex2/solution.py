import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


### HELPER FUNCTIONS ###
# Add here any helper functions which you think will be useful
def generate_data_svm(n_samples):
    mean = [0, 0]
    cov = [[1,0.5], [0.5,1]]
    X = np.random.multivariate_normal(mean, cov, n_samples)
    w = np.array([-0.6, 0.4])
    y = np.sign(X @ w)
    return X, y

def create_plot_svm(X, y, clf, m, C, save_path):
    plt.figure(figsize=(6,6))

    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette="coolwarm", s=50, edgecolor='k')
    # TRUE
    w_true = np.array([-0.6, 0.4])
    slope_true = -w_true[0]/w_true[1]
    x_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
    plt.plot(x_vals, slope_true * x_vals, 'k--', label="True boundary")

    # SVM
    w = clf.coef_[0]
    b = clf.intercept_[0]
    slope_svm = -w[0]/w[1]
    intercept_svm = -b/w[1]
    plt.plot(x_vals, slope_svm * x_vals + intercept_svm, 'r-', label="SVM boundary")

    plt.title(f"SVM C={C}, m={m}")
    plt.legend()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'svm_C{C}_m{m}.png'))
    else:
        plt.show()

    plt.close()

def generate_two_gaussians(n_samples=200):
    mean1 = [-1, -1]
    mean2 = [1, 1]
    cov = [[0.5, 0.2], [0.2, 0.5]]
    n_half = n_samples // 2
    X1 = np.random.multivariate_normal(mean1, cov, n_half)
    y1 = np.zeros(n_half)
    X2 = np.random.multivariate_normal(mean2, cov, n_half)
    y2 = np.ones(n_half)
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y


def plot_decision_boundary(X_train, y_train, X_test, y_test, model, model_name, dataset_name, save_path=None):
    import matplotlib.patches as mpatches

    plt.figure(figsize=(6,6))

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

    for label, marker, color in zip([0, 1], ['o', 'x'], ['blue', 'red']):
        idx = np.where(y_train == label)
        if marker == 'o':
            plt.scatter(X_train[idx, 0], X_train[idx, 1], c=color, marker=marker, edgecolor='k', s=70)
        else:
            plt.scatter(X_train[idx, 0], X_train[idx, 1], c=color, marker=marker, s=70)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    correct = y_pred == y_test
    colors = np.where(correct, 'lime', 'purple')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, edgecolor='k', s=100, alpha=0.8)

    legend_elements = [
        mpatches.Patch(color='blue', label='Train class 0'),
        mpatches.Patch(color='red', label='Train class 1'),
        mpatches.Patch(color='lime', label='Correct test'),
        mpatches.Patch(color='purple', label='Wrong test')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title(f"{model_name} on {dataset_name}\nTest Accuracy: {acc:.2f}", fontsize=13)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{model_name}_{dataset_name}.png"))
    else:
        plt.show()

    plt.close()




### Exercise Solution ###

def pratical_1_runner(save_path=None):
    M = [5, 10, 20, 100]
    C = [0.1, 1, 5, 10, 100]

    for m in M:
        for c in C:
            while True:
                X_train, y_train = generate_data_svm(m)
                if len(np.unique(y_train)) > 1:
                    break
            clf = SVC(C=c, kernel='linear')
            clf.fit(X_train, y_train)
            create_plot_svm (X_train, y_train, clf, m, c, save_path)


def practical_2_runner(save_path=None):
    moons_X, moons_y = make_moons(200, noise=0.2, random_state=42)
    circles_X, circles_y = make_circles(200, noise=0.1, factor=0.5, random_state=42)
    gaussians_X, gaussians_y = generate_two_gaussians(200)

    datasets = {
        "Moons": (moons_X, moons_y),
        "Circles": (circles_X, circles_y),
        "Gaussians": (gaussians_X, gaussians_y)
    }

    for dataset_name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #SVM
        svm = SVC(C=0.2, kernel='linear')
        svm.fit(X_train, y_train)
        plot_decision_boundary(X_train, y_train, X_test, y_test, svm, "SVM", dataset_name, save_path)

        # DECISION TREE
        tree = DecisionTreeClassifier(max_depth=7)
        tree.fit(X_train, y_train)
        plot_decision_boundary(X_train, y_train, X_test, y_test, tree, "Decision Tree", dataset_name, save_path)

        # KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        plot_decision_boundary(X_train, y_train, X_test, y_test, knn, "KNN", dataset_name, save_path)



if __name__ == "__main__":
    #SVM
    path = None
    #pratical_1_runner(save_path=path)
    practical_2_runner(save_path=path)