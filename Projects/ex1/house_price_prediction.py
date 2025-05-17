import numpy as np
import pandas as pd
import plotly.express as px
import os
from typing import NoReturn
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

def prepare_data (cd: pd.DataFrame):
    cd['house_age'] = 2025 - cd['yr_built']
    cd['years_since_renovation'] = 2025 - cd['yr_renovated']
    cd['sqft_living_difference'] = cd['sqft_living'] - cd['sqft_living15']
    cd['sqft_lot_difference'] = cd['sqft_lot'] - cd['sqft_lot15']
    cd = cd.drop(columns=['id', 'date', 'yr_built', 'yr_renovated'])
    cd.drop_duplicates(inplace=True)
    return cd

def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    cd = pd.concat([X,y], axis=1)
    # filter illogical data
    cd = cd[(cd['yr_renovated'] == 0) | (cd['yr_renovated'] >= cd['yr_built'])]
    cd = cd[cd['sqft_lot'] >= cd['sqft_living']]
    cd = cd[(cd['sqft_basement'] + cd['sqft_above']) == cd['sqft_living']]
    cd = cd[(cd['long'] <= -121) & (cd['long'] >= -123)]
    cd = cd[(cd['lat'] >= 47) & (cd['lat'] <= 48)]
    cd = cd[cd['bedrooms']!=0]

    cd = prepare_data(cd)

    #split back to features and price
    X_clean = cd.drop(columns=['price'])
    y_clean = cd['price']

    return X_clean, y_clean


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    cd = X
    cd = prepare_data(cd)
    return cd


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        feat_values = X[feature]
        y_values = y
        if np.std(feat_values) == 0 or np.std(y_values) == 0:
            pearson_corr = 0
        else:
            cov = feat_values.cov(y_values)
            pearson_corr = cov / (np.std(feat_values) * np.std(y_values))
            pearson_corr = np.clip(pearson_corr, -1, 1)

        # create graph and save it.
        plt.figure(figsize=(8, 6))
        plt.scatter(feat_values, y_values, alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel("price")
        plt.title(f"{feature} against price \n Pearson Correlation: {pearson_corr:.3f}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{feature}.png"))
        plt.close()



if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price
    # Question 2 - split train test

    # reconnect X, y for shuffling
    data = np.column_stack([X,y])
    np.random.seed(54)
    np.random.shuffle(data)

    #spliting to training and testing
    train_count = int(0.75*data.shape[0])
    train_data = data[:train_count]
    test_data = data[train_count:]

    # split training and testing back to X,y
    X_train, y_train = train_data[:,:-1], train_data[:,-1]
    columns = X.columns
    columns_test = df.columns

    # Question 3 - preprocessing of housing prices train dataset
    X_clean, y_clean = preprocess_train(pd.DataFrame(X_train,columns=columns), pd.Series(y_train,name='price'))

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_clean, y_clean)

    # Question 5 - preprocess the test data
    test_pre_data = preprocess_test(pd.DataFrame(test_data,columns=columns_test))


    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    X_test_np = X_clean.to_numpy(dtype=np.float64)
    y_test_np = y_clean.to_numpy(dtype=np.float64)
    perncentages = np.arange(10, 101)
    mean_losses = []
    std_losses = []
    #run the model for all percentages
    for p in perncentages:
        losses = []
        for j in range(10):
            sample_X = X_clean.sample(frac=p / 100, random_state=None)
            sample_X = sample_X.apply(pd.to_numeric, errors='coerce')
            sample_X = sample_X.fillna(0)
            sample_y = y_clean.loc[sample_X.index]
            sample_y = pd.to_numeric(sample_y, errors='coerce')
            sample_y = sample_y.fillna(0)
            model = LinearRegression(include_intercept=True)
            model.fit(sample_X.to_numpy(), sample_y.to_numpy())
            loss = model.loss(X_test_np, y_test_np)
            losses.append(loss)
        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))

    #create graph
    plt.figure(figsize=(14, 10))
    plt.plot(perncentages, mean_losses, marker='o', label='Mean Loss', color='red')
    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)
    plt.fill_between(perncentages, mean_losses - 2 * std_losses, mean_losses + 2 * std_losses, alpha=0.5, color='blue',
                     label='Mean +- 2*STD')
    plt.title('Mean Loss vs. Training Percentage')
    plt.xlabel('Training Set Percentage')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_loss_vs_training_size.png")

