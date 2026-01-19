"""Lasso selection helper translated from the original R code."""
import numpy as np

from .train_lasso import train_lasso


def selLasso(X, pars=None, output=False, k=None):
    """Select candidate parents using Lasso regression."""
    if output:
        print(f"Performing variable selection for variable {k}:")

    result = {}
    X = np.asarray(X)
    p = X.shape

    if p[1] > 1:
        selVec = [False] * p[1]
        modfitGam = train_lasso(
            X[:, :k].tolist() + X[:, k+1:].tolist(), X[:, k].tolist(), pars)
        selVecTmp = [False] * (p[1] - 1)

        # Access the coefficients from the Lasso model
        coefficients = modfitGam['model'].coef_
        for i, coef in enumerate(coefficients):
            if coef != 0:
                selVecTmp[i] = True

        selVec[:k] = selVecTmp[:k]
        selVec[k+1:] = selVecTmp[k:]
    else:
        selVec = []

    return selVec
