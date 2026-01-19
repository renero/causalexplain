"""
This module contains functions to extract and visualize the weights of a neural
network model. The weights are extracted from the model and then visualized in
different ways to help understand the relationships between the input features
and the target variable. The functions in this module are used to visualize the
weights of a neural network model and to identify relationships between the input
features and the target variable.

"""

from collections import Counter
from typing import Dict, List, Tuple

import kneed
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from mlforge.progbar import ProgBar
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from causalexplain.common.utils import graph_from_dictionary
from causalexplain.models._models import MLPModel


# pylint: disable=too-many-locals, consider-using-enumerate


def extract_weights(model: torch.nn.Module, verbose: bool = False) -> List[np.ndarray]:
    """
    Extracts the weights from a given model.

    Parameters:
    - model: The model from which to extract the weights.
    - verbose: If True, prints the names of the weights being extracted.

    Returns:
    - weights: A list of the extracted weights.

    """
    weights = []
    for name, prm in model.named_parameters():
        if verbose:
            print(name)
        if "weight" in name:
            w = prm.detach().numpy()
            weights.append(w)

    return weights


def see_weights_to_hidden(
        weights_matrix: np.ndarray,
        input_names: List[str],
        target: str) -> None:
    """
    Visualizes the weights connecting the input layer to the hidden layer.

    Args:
        W (np.ndarray): The weight matrix of shape (num_hidden, num_inputs) representing
        the connections between the input layer and the hidden layer.
        input_names (List[str]): A list of input names corresponding to each input
            feature.
        target (str): The target variable.

    Returns:
        None

    """
    num_rows = 4
    num_cols = 3
    num_plots = weights_matrix.shape[0]
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 8))
    x_range = range(weights_matrix.shape[1])
    hidden_names = [f"h{i:02d}" for i in range(weights_matrix.shape[0])]
    for idx, i in enumerate(range(num_plots)):
        row = int((i) / num_cols)
        col = np.round(((i / num_cols) - row) * num_cols).astype(int)
        ax[row, col].axhline(0, color="black", linewidth=0.5)
        ax[row, col].bar(x_range, weights_matrix[i], alpha=0.6)
        ax[row, col].set_xticks(x_range)
        ax[row, col].set_xticklabels(input_names, fontsize=7)
        ax[row, col].set_title(hidden_names[i])

    fig.suptitle(f"target: {target}", fontsize=14)
    plt.tight_layout()
    plt.show()


def see_weights_from_input(W, input_names, target):
    num_rows = 4
    num_cols = 3
    num_plots = W.shape[1]
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 8))
    x_range = range(W.shape[0])
    hidden_names = [f"h{i:02d}" for i in range(W.shape[0])]
    for idx, i in enumerate(range(num_plots)):
        row = int((i) / num_cols)
        col = np.round(((i / num_cols) - row) * num_cols).astype(int)
        ax[row, col].axhline(0, color="black", linewidth=0.5)
        ax[row, col].bar(x_range, W[:, i], alpha=0.6)
        ax[row, col].set_xticks(x_range)
        ax[row, col].set_xticklabels(hidden_names, fontsize=7)
        ax[row, col].set_title(input_names[idx])

    fig.suptitle(f"target: {target}", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_feature(result, axis=None):
    if axis is None:
        axis = plt.gca()
    all_weights = extract_weights(result.model)
    W = all_weights[0]
    target_name = list(set(result.all_columns) - set(result.columns))[0]
    weights = pd.DataFrame(
        W, columns=result.columns, index=[f"h{i}" for i in range(W.shape[0])]
    )
    transformations = pd.DataFrame(
        data={
            "psd": (np.power(weights, 2.0) / 2.0).mean(),
            "avg": weights.mean(),
            "med": weights.median(),
        }
    )
    transformations.plot(kind="bar", width=0.8, title=target_name, ax=axis)


def plot_features(
        results: Dict[str, MLPModel],
        n_rows, n_cols, all_columns
):
    sns.set_style("whitegrid")
    num_rows = n_rows
    num_cols = n_cols
    row, col = 0, 0
    _, ax = plt.subplots(num_rows, num_cols, figsize=(12, 14))
    for target in all_columns:
        plot_feature(results[target], axis=ax[row, col])
        col += 1
        if col == num_cols:
            row += 1
            col = 0

    plt.tight_layout()
    plt.show()


def layer_weights(dff_net, target, layer=0):
    W = extract_weights(dff_net.model)[layer]
    weights = pd.DataFrame(
        W, columns=dff_net.columns, index=[f"h{i}" for i in range(W.shape[0])]
    )
    transformations = pd.DataFrame(
        data={
            f"psd_{target}": (np.power(weights, 2.0) / 2.0).mean(),
            f"avg_{target}": weights.mean(),
            f"med_{target}": weights.median(),
        }
    )
    return transformations


def _plot_clusters(X: pd.DataFrame, K: DBSCAN, target: str, counts: Dict, min_counts=2):
    colors = [mpl.cm.tab20(i) for i in range(20)]
    fig, ax = plt.subplots(1, 3, figsize=(8, 2), sharey=True)
    minX, maxX = X.min().min(), X.max().max()
    minX -= (maxX - minX) * 0.1
    maxX += (maxX - minX) * 0.1
    for i in range(X.shape[0]):
        ax[0].scatter(X.iloc[i, 0], X.iloc[i, 1],
                      color=colors[K.labels_[i]], alpha=0.5)
        ax[1].scatter(X.iloc[i, 0], X.iloc[i, 2],
                      color=colors[K.labels_[i]], alpha=0.5)
        ax[2].scatter(X.iloc[i, 1], X.iloc[i, 2],
                      color=colors[K.labels_[i]], alpha=0.5)
        if counts[K.labels_[i]] <= min_counts:
            ax[0].text(
                X.iloc[i, 0],
                X.iloc[i, 1],
                f"{X.iloc[i,:].name}  ",
                horizontalalignment="right",
                verticalalignment="center",
            )
            ax[1].text(
                X.iloc[i, 0],
                X.iloc[i, 2],
                f"{X.iloc[i,:].name}  ",
                horizontalalignment="right",
                verticalalignment="center",
            )
            ax[2].text(
                X.iloc[i, 1],
                X.iloc[i, 2],
                f"{X.iloc[i,:].name}  ",
                horizontalalignment="right",
                verticalalignment="center",
            )
        ax[0].set_xlim((minX, maxX))
        ax[0].set_ylim((minX, maxX))
        ax[1].set_xlim((minX, maxX))
        ax[1].set_ylim((minX, maxX))
        ax[2].set_xlim((minX, maxX))
        ax[2].set_ylim((minX, maxX))
    fig.suptitle(target)


def summarize_weights(weights, feature_names, layer=0, scale=True):
    """
    Summarize the weights of a neural network model by calculating the mean,
    median, and positive semidefinite values of the weights for each feature.

    Parameters:
    - weights: The weights of the neural network model.
    - feature_names: A list of feature names.
    - layer: The layer of the neural network model from which to extract the weights.
    - scale: If True, scale the summary values.

    Returns:
    - psd: A DataFrame containing the summary values of the weights for each feature.

    """
    l1 = dict()
    psd = pd.DataFrame()
    for feature in feature_names:
        l1[feature] = layer_weights(weights[feature], feature, layer)
        psd = pd.concat(
            (psd, l1[feature]
             [[f"psd_{feature}", f"med_{feature}", f"avg_{feature}"]]),
            axis=1,
        )

    psd = psd.fillna(0)
    if scale:
        scaler = StandardScaler()
        psd = pd.DataFrame(scaler.fit_transform(psd),
                           index=psd.index, columns=psd.columns)
    return psd


def identify_relationships(weights, feature_names, eps=0.5, min_counts=2, plot=True):
    """Identify feature relationships using clustered weight summaries."""
    rels = {}
    for target in feature_names:
        X = weights[[f"psd_{target}", f"med_{target}",
                     f"avg_{target}"]].drop(target)
        K = DBSCAN(eps=eps, min_samples=1).fit(X)  # do_cluster(X, eps=eps)

        # Pairs with cluster_id: num_elements_per_cluster_id
        counts = Counter(K.labels_)

        rels[target] = []
        for i in range(X.shape[0]):
            if counts[K.labels_[i]] <= min_counts and X.iloc[i, :].name != "Noise":
                rels[target].append(X.iloc[i, :].name)
        if plot:
            _plot_clusters(X, K, target, counts)

    return rels


def _get_shap_values(model: MLPModel) -> np.ndarray:
    """
    Get the SHAP values for the given model.

    Parameters:
    - model: The model for which to compute the SHAP values.

    Returns:
    - shap_values: The SHAP values for the given model.

    """
    explainer = shap.DeepExplainer(
        model.model, model.train_loader.dataset.features)
    shap_values = explainer.shap_values(
        model.train_loader.dataset.features)

    return shap_values


def _average_shap_values(
        shap_values: Dict[str, np.ndarray],
        column_names: List[str],
        abs: bool = False) -> np.array:
    """
    Compute the average SHAP values for the given model.

    Parameters:
    - shap_values: The SHAP values for the given model.
    - column_names: The names of the columns in the dataset.
    - abs: If True, compute the absolute values of the SHAP values.

    Returns:
    - avg_shaps: The average SHAP values for the given model.

    """
    avg_shaps = []
    for i in range(len(column_names)):
        target_name = column_names[i]
        if abs:
            y = np.mean(np.abs(shap_values[target_name].T), axis=1) * 2.0
        else:
            y = np.mean(shap_values[target_name].T, axis=1) * 2.0
        avg_shaps.append(y)

    avg_shaps = np.array(avg_shaps)
    return avg_shaps


def _find_shap_elbow(avg_shaps: np.array, plot=False, verbose=False) -> float:
    n_bins = 20
    if avg_shaps.shape[0] < 20:
        n_bins = avg_shaps.shape[0]
    cutoff = 0
    # Try to find the elbow of the histogram counts unless the cutoff selected
    # is zero, in which case, the `knee` algorithm simply failed because
    # values in histogram are not monotonically decreasing uniformly.
    while cutoff == 0:
        histogram = np.histogram(avg_shaps, bins=n_bins)
        knee = kneed.KneeLocator(range(len(histogram[0])), histogram[0], S=1.0,
                                 curve="convex", direction="decreasing")
        threshold = histogram[1][knee.elbow]
        cutoff = knee.elbow
        if cutoff == 0:
            n_bins -= 1

    if plot:
        knee.plot_knee(figsize=(4, 3))
    if verbose:
        print(f"Cutoff pos.: {knee.elbow}; Threshold: {threshold:.4f}")

    return threshold


def _identify_edges(avg_shaps: np.array,
                    feature_names: List[str],
                    threshold: Dict[str, float]) -> Dict[str, List[Tuple[str, float]]]:
    rels = {}
    for i, target_name in enumerate(feature_names):
        labels = [f for f in feature_names if f != target_name]
        candidate_positions = np.argwhere(
            avg_shaps[i] > threshold[target_name]).flatten().tolist()
        rels[target_name] = [(labels[position], avg_shaps[i][position])
                             for position in candidate_positions]

    return rels


def _orient_edges_based_on_shap(G_shap: nx.DiGraph, verbose=False) -> nx.DiGraph:
    g = nx.DiGraph()
    already_checked = []
    if verbose:
        print("Orienting edges based on SHAP values ratio")
    for u, v, data in G_shap.edges(data=True):
        if G_shap.has_edge(v, u):
            if f"{u}->{v}" in already_checked:
                continue
            already_checked.append(f"{v}->{u}")
            reverse = G_shap.get_edge_data(v, u)
            diff = reverse['weight'] / data['weight']
            if verbose:
                print(f"{u}->{v}: {data['weight']:.3f} | ", end="")
                print(f"{v}->{u}: {reverse['weight']:.3f} | ")
                print(f"ratio: {diff:.2f}")
            if diff < 0.95:
                g.add_edge(u, v, weight=data['weight'])
            elif diff > 1.05:
                g.add_edge(v, u, weight=reverse['weight'])
            else:
                g.add_edge(u, v, weight=data['weight'])
                g.add_edge(v, u, weight=reverse['weight'])
        else:
            g.add_edge(u, v, weight=data['weight'])

    return g


def _remove_asymmetric_shap_edges(
    relationships: Dict[str, List[Tuple[str, float]]], verbose=False
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Removes those edges who do not have a symmetric relationship found across the SHAP
    process.
    """
    shap_relationships = relationships.copy()
    asymmetric_edges = {}
    for key, values in shap_relationships.items():
        relation_names = [f"{relation[0]}" for relation in values]
        for potential_relation in relation_names:
            names = [e[0] for e in shap_relationships[potential_relation]]
            if key not in names:
                if key not in asymmetric_edges:
                    asymmetric_edges[key] = []
                asymmetric_edges[key].append(potential_relation)
                if verbose:
                    print(f"{key} -X-> {potential_relation} has NO symmetric")

    # Now remove those who are asymmetric.
    for target in shap_relationships.keys():
        if target in asymmetric_edges.keys():
            shap_relationships[target] = [
                tup
                for tup in shap_relationships[target]
                if tup[0] not in asymmetric_edges[target]
            ]

    return shap_relationships


def infer_causal_relationships(
        trained_models: Dict[str, MLPModel],
        feature_names: List[str],
        prune: bool = False,
        verbose=False,
        plot=False,
        prog_bar=True,
        silent=False):
    """Infer causal relationships from SHAP values of trained models."""
    # pbar = tqdm(total=len(feature_names),
    #             **tqdm_params("Computing SHAPLEY values", prog_bar, silent=silent))
    if prog_bar and not silent:
        pbar = ProgBar().start_subtask(len(feature_names))

    shap_values = {}

    # desc="Computing SHAPLEY values",
    # disable=not prog_bar, position=1, leave=False)
    for target_name in feature_names:
        model = trained_models[target_name]
        shap_values[target_name] = _get_shap_values(model)
        pbar.update_subtask(1)

    avg_shaps = _average_shap_values(shap_values, feature_names)
    feature_threshold = {}
    for i, target in enumerate(feature_names):
        feature_threshold[target] = _find_shap_elbow(
            avg_shaps[i], plot, verbose)
    edges = _identify_edges(avg_shaps, feature_names, feature_threshold)
    if prune:
        edges = _remove_asymmetric_shap_edges(edges)
    G_shap = graph_from_dictionary(edges)
    G_shap_oriented = _orient_edges_based_on_shap(G_shap, verbose=False)

    return {
        'shap_values': shap_values,
        'avg_shaps': avg_shaps,
        'thresholds': feature_threshold,
        'raw_graph': G_shap,
        'graph': G_shap_oriented
    }
