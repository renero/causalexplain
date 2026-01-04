# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

from copy import copy
from typing import List

import colorama
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.metrics import euclidean_distances

BLACK = colorama.Fore.BLACK
RED = colorama.Fore.RED
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Style.RESET_ALL


def select_features(
        values,
        feature_names,
        return_shaps=False,
        min_impact: float = 1e-06,
        exhaustive=False,
        threshold: float|None = None,
        verbose=False) -> List[str]:
    """
    Sort the values and select those before (strict) the point of max. curvature,
    according to the selected algorithm. If strict is False, the point of max curv.
    is also selected. When the method is 'abrupt' the selection method is based on
    taking only those feature up-to (down-to) a certain percentage of change in their
    values.

    Arguments:
        - values (np.ndarray): The values for each of the features. This can
            be anything that should be used to determine what features are more
            important than others.
        - feature_names (list): Names of the variables corresponding to the shap values
        - return_shaps (bool): Whether returning the mean shap values together with
            order of the features.
        - min_impact (float): Default 1e-06. The minimum impact of a feature to be
            selected. If all features are below this value, none are selected.
        - exhaustive (bool): Default False. Whether to use the exhaustive method or
            not. If True, the threshold is used to find all possible clusters above
            the given threshold, not only the first one.
        - threshold (float): Default None. The threshold to use when exahustive is
            True. If None, exception is raised.
        - verbose: guess what.

    """
    if (exhaustive is True) and (threshold is None):
        raise ValueError("If exhaustive is True, threshold must be provided.")
    threshold = 0.0 if threshold is None else threshold

    if len(values.shape) > 1:
        feature_order = np.argsort(np.sum(np.abs(values), axis=0))
        mean_values = np.abs(values).mean(0)
    else:
        feature_order = np.argsort(np.abs(values))
        mean_values = np.abs(values)
    sorted_shap_values = np.array([mean_values[idx] for idx in feature_order])
    if verbose:
        print(f"  > Feature order......: {feature_order}")

    # In some cases, the mean SHAP values are 0. We return an empty list in that case.
    if np.all(mean_values < min_impact):
        return []

    if verbose:
        print("  > Sum values.........: ", end="")
        if len(values.shape) > 1:
            print(','.join([f"({f}:{s:.03f})" for f, s in zip(
                feature_names, np.sum(np.abs(values), axis=0))]))
        else:
            print(','.join([f"({f}:{s:.03f})" for f, s in zip(
                feature_names, np.abs(values))]))
        print(
            f"  > Feature_order......: "
            f"{','.join([f'{feature_names[i]}' for i in feature_order if i < len(feature_names)])}\n"
            f"  > sorted_mean_values.: "
            f"{','.join([f'{x:.6f}' for x in sorted_shap_values])}\n"
            f"  > threshold..........: {threshold:.6f}")

    sorted_impact_values = copy(sorted_shap_values)
    selected_features = []
    max_iterations = len(sorted_impact_values)
    iteration = 0
    limit_idx = 0
    while np.any(sorted_impact_values > threshold):
        if iteration >= max_iterations:
            break

        limit_idx = find_cluster_change_point(sorted_impact_values, verbose=verbose)

        if verbose:
            print("  > Clustering iteration Details:")
            print(
                f"     > Iteration..........: {iteration}\n"
                f"     > Limit_idx..........: {limit_idx}\n"
                f"     > feature_names......: {feature_names}\n"
                f"     > feature_order......: "
                f"{','.join([str(i) for i in feature_order])}\n"
                f"     > Sorted_impact_vals.: "
                f"{','.join([f'{x:.6f}' for x in sorted_impact_values])}")

        selected_features = list(reversed(
            [feature_names[i] for i in feature_order[limit_idx:] if i < len(feature_names)]))

        if not exhaustive:
            break

        sorted_impact_values = sorted_impact_values[:limit_idx]
        iteration += 1

    if verbose:
        print(f"  > Limit_idx(cut-off).: {limit_idx}")
        print(f"  > Selected_features..: {selected_features}")
    if return_shaps:
        return selected_features, list(reversed(sorted(mean_values)[limit_idx:]))

    return selected_features


def find_cluster_change_point(X: List, verbose: bool = False) -> int|None:
    """
    Given an array of values in increasing or decreasing order, detect what are the
    elements that belong to the same cluster. The clustering is done using DBSCAN
    with a distance computed as the max. difference between consecutive elements.

    Arguments:
        - X (np.array): the series of values to detect the abrupt change.
        - verbose (bool): Verbose output.

    Returns:
        The position in the array where an abrupt change is produced. If there's
            no change in consecutive values greater than the tolerance passed then
            the last element of the array is returned.
    """
    if len(X) <= 1:
        return None

    X = np.array(X).reshape(-1, 1)
    X_safe = X.copy()
    X_safe[X_safe == 0.0] = 1e-06

    pairwise_distances = euclidean_distances(X)[0,]
    pairwise_distances = np.diff(pairwise_distances)
    pairwise_distances = np.sort(pairwise_distances)[::-1]

    # Safety check: if any of the values in pairwise_distances is 0, add 1e-06
    pairwise_distances[pairwise_distances == 0.0] = 1e-06

    n_clusters_ = 0
    while n_clusters_ <= 1 and len(pairwise_distances) > 0:
        max_distance = pairwise_distances.max() + 1e-06

        # Safety check
        max_distance = 0.001 if max_distance <= 0.0 else max_distance

        if verbose:
            print(f"  > pairwise_distances.: {pairwise_distances}")
            print(f"  > max_distance.......: {max_distance:.4f}")

        db = DBSCAN(eps=max_distance, min_samples=1).fit(X)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        if n_clusters_ <= 1:
            if verbose:
                print("      ↳ Only 1 cluster generated. Decreasing max_distance.")
            pairwise_distances = pairwise_distances[1:]

    if pairwise_distances.size == 0:
        print("** No clusters generated") if verbose else None
        return None

    if verbose:
        print(f"  > Est.clusters/noise.: {n_clusters_}/{n_noise_}")
        if (len(labels) > 3) and (len(labels) < (X.shape[0]-1)):
            print(
                f"  Silhouette Coeff...: {metrics.silhouette_score(X, labels):.3f}\n"
                f"    ↳ Labels: {labels}")

    winner_label = n_clusters_ - 1
    samples_in_winner_cluster = np.argwhere(X_safe[labels != winner_label])

    return samples_in_winner_cluster[:, 0][-1]+1


def main():
    # Display Options
    np.set_printoptions(precision=4, linewidth=100)
    pd.set_option('display.precision', 4)
    pd.set_option('display.float_format', '{:.4f}'.format)

    toy_values = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.959290610692456]
    )
    # values = np.array(
    #     [0.083, 0.175, 0.353, 0.204, 0.081, 0.116, 0.088, 0.451, 0.152]
    # )
    names = [f"V{i}" for i in range(len(toy_values))]
    select_features(toy_values, names, verbose=True)


def test():
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    feature_names = ['A', 'B', 'C', 'D', 'E']
    result = select_features(values, feature_names)
    assert result == ['A', 'B', 'C', 'D', 'E']


if __name__ == "__main__":
    test()
