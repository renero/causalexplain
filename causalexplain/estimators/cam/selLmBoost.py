"""Linear-model boost selection helper translated from the original R code."""
import numpy as np

from .train_LMboost import train_LMboost


def selLmBoost(X, pars=None, output=False, k: int = None):
    """Select candidate parents using boosted linear models."""
    if pars is None:
        pars = {'atLeastThatMuchSelected': 0.02, 'atMostThatManyNeighbors': 10}

    if output:
        print(f"Performing variable selection for variable {k}:")

    X = np.array(X)
    p = X.shape

    if p[1] > 1:
        selVec = [False] * p[1]

        modfit_lm = train_LMboost(np.delete(X, k, axis=1), X[:, k], pars)
        # Get the indices of the important features
        cc = np.argsort(modfit_lm['model'].feature_importances_)[::-1]

        if output:
            print("The following variables")
            print(cc)

        # Create a list to store the importance of each feature in 'cc'
        how_often_selected = modfit_lm['model'].feature_importances_

        if output:
            print("... have been selected that many times:")
            print(how_often_selected)

        how_often_selected_sorted = sorted(how_often_selected, reverse=True)

        if sum(np.array(how_often_selected) > pars['atLeastThatMuchSelected']) \
                > pars['atMostThatManyNeighbors']:
            cc = cc[np.array(how_often_selected) >
                    how_often_selected_sorted[pars['atMostThatManyNeighbors']]]
        else:
            cc = cc[np.array(how_often_selected) >
                    pars['atLeastThatMuchSelected']]

        if output:
            print("We finally choose as possible parents:")
            print(cc)
            print()

        tmp = [False] * (p[1] - 1)
        for idx in cc:
            tmp[idx] = True

        selVec[:k] = tmp[:k]
        selVec[k+1:] = tmp[k:]
    else:
        selVec = []

    return selVec
