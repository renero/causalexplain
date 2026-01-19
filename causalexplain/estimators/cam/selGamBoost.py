"""GAM boost selection helper translated from the original R code."""
import numpy as np
from .train_GAMboost import train_GAMboost


def selGamBoost(X, pars=None, output=False, k=None):
    """Select candidate parents using boosted GAMs."""
    if pars is None:
        pars = {'atLeastThatMuchSelected': 0.02, 'atMostThatManyNeighbors': 10}

    if output:
        print(f"Performing variable selection for variable {k}:")

    result = {}
    X = np.array(X)
    p = X.shape

    if p[1] > 1:
        selVec = [False] * p[1]
        X_without_k = np.delete(X, k, axis=1)
        X_k = X[:, k]
        modfitGam = train_GAMboost(X_without_k, X_k, pars)

        # Replace the xselect() call with feature_importances_
        feature_importances = modfitGam['model'].feature_importances_
        # Indices of features sorted by importance
        cc = np.argsort(feature_importances)[::-1]
        if output:
            print("The following variables")
            print(cc)

        nstep = len(feature_importances)
        howOftenSelected = feature_importances[cc]

        if output:
            print("... have been selected that many times:")
            print(howOftenSelected)

        howOftenSelectedSorted = sorted(howOftenSelected, reverse=True)

        if sum(np.array(howOftenSelected) > pars['atLeastThatMuchSelected']) \
                > pars['atMostThatManyNeighbors']:
            cc = cc[np.array(howOftenSelected) >
                    howOftenSelectedSorted[pars['atMostThatManyNeighbors']]]
        else:
            cc = cc[np.array(howOftenSelected) >
                    pars['atLeastThatMuchSelected']]

        if output:
            print("We finally choose as possible parents:")
            print(cc)
            print()

        tmp = [False] * (p[1] - 1)
        for i in cc:
            tmp[i] = True
        selVec[:k] + tmp + selVec[k+1:]
    else:
        selVec = []

    return selVec
