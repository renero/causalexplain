from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame
from pygam import LinearGAM
from sklearn.gaussian_process import GaussianProcessRegressor as gpr

from causalexplain.independence.hsic import HSIC


def _fit_GPR(x, y, x_test, y_test):
    model = gpr(normalize_y=False)
    model.fit(x, y)
    prediction = model.predict(x_test).reshape(-1, 1)
    residuals = y_test - prediction
    return residuals


def _fit_GAM(x, y, x_test, y_test):
    model = LinearGAM()
    model.gridsearch(x, y, progress=False)
    pred = model.predict(x_test)
    residuals = np.array([x_test[i] - pred[i] for i in range(x_test.shape[0])])
    return residuals


def fit_and_get_residuals(
    X: np.ndarray,
    Y: np.ndarray,
    X_test: np.ndarray| None = None,
    Y_test: np.ndarray| None = None,
    method="gpr"
):
    """
    Fit a model y ~ f(X), where X is an independent variable and Y is a
    dependent one. The model is passed as argument, together with the
    training and test sets.

    Args:
        X: (np.ndarray) The feature to be used as input to predict Y_train
        Y: (np.ndarray) The feature to be predicted
        X_test: (np.ndarray) The feature to be used as input to predict Y_test
        Y_test: (np.ndarray) The feature to be predicted
        method: (str) Either "gpr" or "gam"

    Returns:
        The method returns the residuals and the RMS error.
    """
    # Fix dimensions if hasn't been done already.
    if np.ndim(X) == 1:
        X = X.reshape(-1, 1)
    if np.ndim(Y) == 1:
        Y = Y.reshape(-1, 1)
    if np.ndim(X_test) == 1:
        X_test = X_test.reshape(-1, 1)
    if np.ndim(Y_test) == 1:
        Y_test = Y_test.reshape(-1, 1)

    noise = np.random.normal(0, .1, X.shape[0]).reshape(-1, 1)
    Y = Y + noise

    if X_test is None or Y_test is None:
        X_test = X
        Y_test = Y

    if method == "gpr":
        # residuals, XX, YY = _fit_GPR(X, Y, X_test, Y_test)
        residuals = _fit_GPR(X, Y, X_test, Y_test)
    elif method == "gam":
        # residuals, XX, YY = _fit_GAM(X, Y, X_test, Y_test)
        residuals = _fit_GAM(X, Y, X_test, Y_test)
    else:
        raise ValueError(f"Invalid method: {method}")

    return residuals


def run_feature_selection(X: DataFrame, y: str) -> List:
    """
    Extracts 'y' from the list of features of "X" and call the prediction
    method passed to asses the predictive influence of each variable in X
    to obtain "y".

    Args:
         X: Dataframe with ALL continous variables
         y: the name of the variable in X to be used as target.
         predict_method: the method used to predict "y" from "X".
            "hsiclasso" or "block_hsic_lasso"

    Return:
        List: with the predictive score for each variable.
    """
    feature_names = list(X.columns.values)
    feature_names.remove(y)
    df_target = pd.DataFrame(X[y], columns=[y])
    df_features = X[feature_names]

    y = np.transpose(df_target.values)
    X = np.transpose(df_features.values)
    hsic = HSIC().fit(X, y)
    return hsic.stat
