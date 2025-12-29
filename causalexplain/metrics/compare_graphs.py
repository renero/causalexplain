#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This method computes metrics between a pair of graphs.
    To call this method, simply pass the reference graph, and the
    predicted graph (the one you want to make as much similar to the first one as
    possible), and all metrics will be computed.

    Use:
    >>> from random import random
    >>> target = nx.DiGraph()
    >>> target.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
    >>> target.add_weighted_edges_from([\
        ('A', 'B', random()), ('B', 'D', random()),('C', 'B', random()),\
        ('D', 'E', random()), ('C', 'E', random())])

    >>> predicted = nx.DiGraph()
    >>> predicted.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
    >>> predicted.add_weighted_edges_from([\
        ('A', 'B', random()), ('A', 'C', random()), ('E', 'A', random()),\
        ('E', 'B', random()), ('C', 'D', random())])

    >>> result = evaluate_graph(target, predicted)
"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
from networkx.linalg import adjacency_matrix
from sklearn.metrics import auc, precision_recall_curve

from causalexplain.metrics.SID import SID
from causalexplain.common import utils

# Code for metrics...
AnyGraph = Union[nx.Graph, nx.DiGraph]


@dataclass
class Metrics:
    """
        This class contains all the metrics computed by the evaluate_graph method.
    """

    Tp: int = 0
    Tn: int = 0
    Fn: int = 0
    Fp: int = 0
    precision: float = 0.0
    recall: float = 0.0
    aupr: float = 0.0
    f1: float = 0.0
    shd: float = 0.0
    sid: float = 0.0

    def __init__(
            self,
            Tp: int,
            Tn: int,
            Fn: int,
            Fp: int,
            precision: float,
            recall: float,
            aupr: float,
            f1: float,
            shd: int,
            sid: Dict[str, float]
    ):
        self.Tp = Tp
        self.Tn = Tn
        self.Fn = Fn
        self.Fp = Fp
        self.precision = precision
        self.recall = recall
        self.aupr = aupr
        self.f1 = f1
        self.shd = shd
        if self.Fn + self.Fp == 0:
            self.ishd = 0.0
        else:
            self.ishd = 1 - (self.shd / (self.Fn + self.Fp))
        self.sid = sid['sid']
        self.sid_lower = sid['sidLowerBound']
        self.sid_upper = sid['sidUpperBound']

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """
        Convert the metrics to a dictionary.
        """
        dictionary = {}
        dictionary["Tp"] = self.Tp
        dictionary["Tn"] = self.Tn
        dictionary["Fn"] = self.Fn
        dictionary["Fp"] = self.Fp
        dictionary["precision"] = self.precision
        dictionary["recall"] = self.recall
        dictionary["f1"] = self.f1
        dictionary["aupr"] = self.aupr
        dictionary["shd"] = self.shd
        dictionary["sid"] = self.sid

        return dictionary

    def __str__(self):
        s = ""
        s += f"Predicted    1     0      Precision: {self.precision:.3g}\n"
        s += f"Actual     -----------    Recall...: {self.recall:.3g}\n"
        s += f"  1       |{self.Tp:^5d} {self.Fn:^5d}|   F1.......: {self.f1:.3g}\n"
        s += f"  0       |{self.Fp:^5d} {self.Tn:^5d}|   AuPR.....: {self.aupr:.3g}\n"
        s += f"           -----------    SHD......: {self.shd:.3g}\n"
        if self.sid_upper == self.sid_lower:
            s += f"                          SID......: {self.sid:.3g}\n"
        else:
            s += f"                          SID......: {self.sid:.3g} [{self.sid_lower}..{self.sid_upper}]\n"
        return s

    def matplotlib_repr(self, usetex: bool = True):
        """
        Generates a formatted string representation of the metrics for display
        in a matplotlib plot.

        Returns:
            str: The formatted string representation of the metrics.
        """
        s_Tp = f"{self.Tp:^5d}"
        s_Tn = f"{self.Tn:^5d}"
        s_Fp = f"{self.Fp:^5d}"
        s_Fn = f"{self.Fn:^5d}"

        if usetex:
            s_Tp = s_Tp.replace(" ", r"\ ")
            s_Tn = s_Tn.replace(" ", r"\ ")
            s_Fp = s_Fp.replace(" ", r"\ ")
            s_Fn = s_Fn.replace(" ", r"\ ")

            s = r"> Predicted\ \ \ 1\ \ \ \ \ 0" + "\n"
            s += r"> Actual\ \ \ \ -----------" + "\n"
            s += r">\ \ 1\ \ \ \ \ \ \ |" + s_Tp + r"\ " + s_Fn + "|" + "\n"
            s += r">\ \ 0\ \ \ \ \ \ \ |" + s_Fp + r"\ " + s_Tn + "|" + "\n"
            s += r">\ \ \ \ \ \ \ \ \ \ \ -----------" + "\n"
        else:
            s = "> Predicted    1     0\n"
            s += "> Actual     -----------\n"
            s += f">   1       |{s_Tp} {s_Fn}|\n"
            s += f">   0       |{s_Fp} {s_Tn}|\n"
            s += ">            -----------\n"

        s += f"> Precision: {self.precision:.3g}" + "\n"
        s += f"> Recall...: {self.recall:.3g}" + "\n"
        s += f"> F1.......: {self.f1:.3g}" + "\n"
        s += f"> AuPR.....: {self.aupr:.3g}" + "\n"
        s += f"> SHD......: {self.shd:.1f}" + "\n"
        s += f"> SID......: {self.sid}"

        return s


def _shallow_copy(G: AnyGraph):
    """
    This method returns a shallow copy of the graph G.

    Parameters
    ----------
    G : AnyGraph
        The graph to copy.

    Returns
    -------
    AnyGraph
        The shallow copy of the graph.
    """
    G_new = nx.DiGraph()
    G_new.add_edges_from(G.edges())
    return G_new


def evaluate_graph(
    ground_truth: AnyGraph,
    predicted_graph: AnyGraph,
    feature_names: Optional[List[str]] = None,
    threshold: float = 0.0,
    absolute: bool = False,
    double_for_anticausal: bool = True,
) -> Optional[Metrics]:
    """
        This method computes metrics between a pair of graphs: the ground truth and
        the predicted graph. To call this method, simply pass the reference graph, and
        the predicted graph (the one you want to make as much similar to the first one
        as possible), and all metrics will be computed.

        Parameters
        ----------
        ground_truth : AnyGraph
            The ground truth graph.
        predicted_graph : AnyGraph
            The predicted graph.
        feature_names : Optional[List[str]], optional
            The list of feature names, by default None
        threshold : float, optional
            The threshold to use for the precision-recall curve, by default 0.0
        absolute : bool, optional
            Whether to use the absolute value of the weights, by default False
        double_for_anticausal : bool, optional
            Whether to double the weights of anticausal edges, by default True

        Returns
        -------
        Metrics
            The metrics object containing all the metrics: Tp, Tn, Fn, Fp, precision,
            recall, AuPR, F1, SHD, ISHD, UMI, OMI.
    """
    if ground_truth is None:
        return None

    # If one graphs is directed and the other is not, then consider them both as
    # undirected.
    if (nx.is_directed(ground_truth) and not nx.is_directed(predicted_graph)) or (
            not nx.is_directed(ground_truth) and nx.is_directed(
                predicted_graph)
    ):
        truth = ground_truth.to_undirected() if nx.is_directed(
            ground_truth) else _shallow_copy(ground_truth)
        predicted = predicted_graph.to_undirected() if nx.is_directed(
            predicted_graph) else _shallow_copy(ground_truth)
    else:
        truth = ground_truth.copy()
        predicted = predicted_graph.copy()

    # Checking whether both contain the same nr of nodes.
    # If not, add isolated nodes to the 'g' graph
    missing_nodes = list(set(ground_truth.nodes()) -
                         set(predicted_graph.nodes()))
    if len(missing_nodes) != 0:
        predicted.add_nodes_from(missing_nodes)

    if feature_names is None:
        feature_names = list(predicted_graph.nodes())

    metrics = dict()
    metrics["_Gm"] = _adjacency(truth, feature_names, threshold, absolute)
    metrics["_gm"] = _adjacency(predicted, feature_names, threshold, absolute)
    if _is_weighted(predicted):
        metrics["_preds"] = _weighted_adjacency(
            predicted, feature_names, threshold, absolute
        )
    else:
        metrics["_preds"] = metrics["_gm"].copy()
    metrics["_double_for_anticausal"] = double_for_anticausal

    metrics["Tp"], metrics["Tn"], metrics["Fp"], metrics["Fn"] = _conf_mat(
        truth, predicted, feature_names
    )

    metrics["precision"] = _precision(metrics)
    metrics["recall"] = _recall(metrics)
    metrics["f1"] = _f1(metrics)
    metrics["aupr"] = _aupr(metrics)
    metrics["SHD"] = _SHD(metrics)
    # Cross check if the number of nodes in the predicted graph is the same as the
    # number of nodes in the ground truth graph
    if predicted_graph.number_of_nodes() < ground_truth.number_of_nodes():
        # Determine what are the nodes missing in the predicted graph
        missing_nodes = list(set(ground_truth.nodes()) -
                             set(predicted_graph.nodes()))
        # Create a copy of the predicted graph
        predicted_graph_copy = predicted_graph.copy()
        # Add the missing nodes to the predicted graph
        predicted_graph_copy.add_nodes_from(missing_nodes)
    #     predicted_graph_array = nx.to_numpy_array(predicted_graph_copy)
    # else:
    #     predicted_graph_array = nx.to_numpy_array(predicted_graph)

    true_adj = utils.graph_to_adjacency(ground_truth, feature_names)
    true_adj = true_adj.astype(np.int8)
    est_adj = utils.graph_to_adjacency(predicted_graph, feature_names)
    est_adj = est_adj.astype(np.int8)
    metrics["SID"] = SID(trueGraph=true_adj, estGraph=est_adj)

    return Metrics(metrics["Tp"], metrics["Tn"], metrics["Fn"], metrics["Fp"],
                   metrics["precision"], metrics["recall"], metrics["aupr"],
                   metrics["f1"], metrics["SHD"], metrics["SID"])


def _is_weighted(G: AnyGraph) -> bool:
    """
    Returns if a graph is weighted by checking if list of edges has data

    Arguments:
        - G (Graph or DiGraph): a graph or digraph

    Returns:
        True if edges have no data.
    """
    ddicts = map(lambda t: bool(t[2]), list(G.edges().data()))
    return any(ddicts)


def _binary_adj_matrix(
        G: AnyGraph,
        order: Optional[List] = None,
        threshold: float = 0.0,
        absolute: bool = False
) -> np.ndarray:
    """
    Returns a binary adjacency matrix from a weighted adjacency matrix. If the
    values in the adjacency matrix are greater than the threshold (default 0.0)
    then that value is transformed into 1.0.

    Arguments:
        - G (Graph or DiGraph): Graph or Digraph
        - threshold (float): Min value of weight to be valued as 1 in the binary
            matrix.
        - absolute (bool): Whether performing the comparison of weights against
            the threshold using absolute value. Default is false.

    Returns:
        np.ndarray with the binary version of the weights.
    """
    if order is None:
        order = sorted(list(G.nodes()))
    m = np.zeros((len(order), len(order)))
    for u, v in G.edges():
        w = G.get_edge_data(u, v)['weight']
        if w is not None:
            if absolute:
                w = np.abs(w)
            if w >= threshold:
                m[order.index(u), order.index(v)] = 1

    return m.astype(np.int16)


def _adjacency(
        G: AnyGraph,
        order: Optional[List] = None,
        threshold=0.0,
        absolute=False) -> np.ndarray:
    """
    Retrieves the adjacency matrix from the graph using NetworkX method if the
    graph is not weighted. This method produces a matrix with 1/0 values only
    since it is used to compute metrics.

    Arguments:
        - G (Graph or DiGraph): the graph to extract the adj. matrix from.
        - order (list): The order of nodes to be used in the adjacency matrix. If
            none specified the node list will be sorted in ascending order.
        - threshold (float): The minimum weight for an edge to be considered as
            present in the adjacency matrix. Default is 0.0.
        - absolute (bool): Default is false. Whether to consider absoute values
            when comparing edge weights agains the threshold.

    Returns:
        numpy.matrix: An adjacency matrix formed by 0s and 1s where the order of
            the nodes is given as argument to the class, and the minimum weight
            must be above the threshold also specified.
    """
    if order is None:
        order = sorted(list(G.nodes()))
    if nx.is_weighted(G) and all(
            [x[-1]['weight'] is not None for x in G.edges(data=True)]):
        result = _binary_adj_matrix(G, order, threshold, absolute)
    else:
        # Scipy adjacency sometimes throws an exception based on type.
        # result = adjacency_matrix(G, nodelist=order).todense()
        nodeset = set(order)
        if nodeset - set(G):
            # delete from order any nodes not in G
            order = [n for n in order if n in G.nodes()]
        adj_matrix = nx.to_numpy_array(G, nodelist=order)
        adj_matrix[np.isnan(adj_matrix)] = 0
        result = adj_matrix

    return result


def _weighted_adjacency(
        G: AnyGraph,
        order: Optional[List] = None,
        threshold=0.0,
        absolute=False) -> np.ndarray:
    """
    Retrieves the adjacency matrix from the graph using NetworkX method if the
    graph is not weighted. This method produces a matrix with 1/0 values only

    Arguments:
        - G (Graph or DiGraph): the graph to extract the adj. matrix from.
        - order (list): The order of nodes to be used in the adjacency matrix. If
            None, the order of the nodes in the graph will be used.
        - threshold (float): The minimum weight to be considered in the matrix.
            Default is 0.0.
        - absolute (bool): Whether performing the comparison of weights against
            the threshold using absolute value. Default is false.

    Returns:
        numpy.matrix: An adjacency matrix formed by 0s and 1s where the order of
            the nodes is given as argument to the class, and the minimum weight
            must be above the threshold also specified.
    """
    if order is None:
        order = sorted(list(G.nodes()))
    F = G.copy()

    def value(x): return abs(x) if absolute else x

    adj_matrix = np.zeros((len(order), len(order)))
    for u, v in F.edges():
        # check if F.get_edge_data(u, v) is not None and contains a key 'weight'
        if 'weight' not in F.get_edge_data(u, v):
            w = 1
        else:
            w = value(F.get_edge_data(u, v)['weight'])
        if w is not None:
            if w >= threshold:
                adj_matrix[order.index(u), order.index(v)] = w
                if not G.is_directed():
                    adj_matrix[order.index(v), order.index(u)] = w
        else:
            adj_matrix[order.index(u), order.index(v)] = 0
            if not G.is_directed():
                adj_matrix[order.index(v), order.index(u)] = 0


    return adj_matrix


def _intersect_matrices(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """
    Computes the intersection of two matrices.

    Parameters:
        mat1: (np.matrix) the first matrix
        mat2: (np.matrix) the second matrix

    Returns:
        np.ndarray: A new matrix with the values in common between the two matrices.
    """
    if not (mat1.shape == mat2.shape):
        raise ValueError("Both graphs must have the same number of nodes")
    mat_intersect = np.where((mat1 == mat2), mat1, 0)

    return mat_intersect


def _positive(matrix: np.ndarray) -> np.ndarray:
    """
    Returns a matrix where negative values are converted to zero,
    and positive values remain the same
    """
    return (matrix > 0).astype(int) * matrix


def _negative(matrix: np.ndarray) -> np.ndarray:
    """
    Returns a matrix where positive values are converted to zero,
    and negative values remain the same.
    """
    return (matrix < 0).astype(int) * matrix



def _conf_mat(truth, est, feature_names):
    if truth.is_directed() and est.is_directed():
        return _conf_mat_directed(truth, est, feature_names)
    else:
        return _conf_mat_undirected(truth, est, feature_names)


def _conf_mat_directed(truth, est, feature_names):
    """
    Computes the confusion matrix for two directed graphs. This method is
    currently only used for directed graphs.

    Arguments:
        truth: (nx.DiGraph) the ground truth graph
        est: (nx.DiGraph) the estimated graph

    Returns:
        Tp: (int) number of true positives
        Tn: (int) number of true negatives
        Fp: (int) number of false positives
        Fn: (int) number of false negatives
    """
    assert truth.is_directed(), "true graph must be directed"
    assert est.is_directed(), "estimated graph must be directed"

    def is_arrow(G, u, v):
        if u not in G.nodes() or v not in G.nodes():
            return False
        return G.has_edge(u, v)  # and not G.has_edge(v, u)

    num_elems = len(feature_names)

    truePositives = np.zeros(
        (num_elems, num_elems)).astype(int)
    estPositives = np.zeros(
        (num_elems, num_elems)).astype(int)
    for i, u in enumerate(feature_names):
        for j, v in enumerate(feature_names):
            if is_arrow(truth, u, v):
                truePositives[i, j] = 1
            if is_arrow(est, u, v):
                estPositives[i, j] = 1

    zeros = np.zeros((num_elems, num_elems))

    Tp = (np.minimum(truePositives == estPositives, truePositives)).sum()
    Tn = (truePositives == estPositives).sum() - Tp - num_elems
    Tn = 0 if Tn < 0 else Tn

    Fp = int((np.maximum(estPositives - truePositives, zeros)).sum())
    Fn = int((np.maximum(truePositives - estPositives, zeros)).sum())

    return Tp, Tn, Fp, Fn


def _conf_mat_undirected(truth, est, feature_names):
    """
    Computes the confusion matrix for two undirected graphs. This method is
    currently only used for undirected graphs.

    Arguments:
        truth: (nx.Graph) the ground truth graph
        est: (nx.Graph) the estimated graph

    Returns:
        Tp: (int) number of true positives
        Tn: (int) number of true negatives
        Fp: (int) number of false positives
        Fn: (int) number of false negatives
    """
    assert not truth.is_directed(), "true graph must be undirected"
    assert not est.is_directed(), "estimated graph must be undirected"

    def is_edge(G, u, v):
        if u not in G.nodes() or v not in G.nodes():
            return False
        return G.has_edge(u, v)

    truePositives = np.zeros(
        (len(feature_names), len(feature_names))).astype(int)
    estPositives = np.zeros(
        (len(feature_names), len(feature_names))).astype(int)
    for i, u in enumerate(feature_names):
        for j, v in enumerate(feature_names):
            if is_edge(truth, u, v):
                truePositives[i, j] = 1
                truePositives[j, i] = 1
            if is_edge(est, u, v):
                estPositives[i, j] = 1
                estPositives[j, i] = 1

    zeros = np.zeros((len(feature_names), len(feature_names)))

    # Excluir las diagonales del conteo
    eq = (truePositives == estPositives)
    np.fill_diagonal(eq, False)

    Tp = int((np.minimum(eq, truePositives).sum() / 2))
    Tn = int(eq.sum() / 2) - Tp
    Fp = int((np.maximum(estPositives - truePositives, zeros)).sum() / 2)
    Fn = int((np.maximum(truePositives - estPositives, zeros)).sum() / 2)

    return Tp, Tn, Fp, Fn


def _confusion_matrix(G, metrics):
    def comp(M):
        return 1 - M

    # These are the intersection between the adjacency matrices. The second one
    # is between the complementary matrices
    G_n_g = _intersect_matrices(metrics["_Gm"], metrics["_gm"])
    cG_n_cg = _intersect_matrices(comp(metrics["_Gm"]), comp(metrics["_gm"]))
    N = metrics["_Gm"].shape[0]
    if isinstance(G, nx.DiGraph):
        _Tp = int(G_n_g.sum())
        _Tn = (
            cG_n_cg[np.triu_indices(n=N, k=+1)].sum()
            + cG_n_cg[np.tril_indices(n=N, k=-1)].sum()
        )
        _Fp = int(_negative(metrics["_Gm"] - metrics["_gm"]).sum())
        _Fn = int(_negative(metrics["_gm"] - metrics["_Gm"]).sum())
    elif isinstance(G, nx.Graph):
        _Tp = int(G_n_g.sum() / 2)
        _Tn = cG_n_cg[np.triu_indices(n=N, k=1)].sum()
        _Fp = int(_negative(metrics["_Gm"] - metrics["_gm"]).sum() / 2)
        _Fn = int(_negative(metrics["_gm"] - metrics["_Gm"]).sum() / 2)
    else:
        raise TypeError("Only networkx Graph and DiGraph graphs supported")
    return _Tp, _Tn, _Fn, _Fp


def _precision(metrics):
    """
    Precision = True Positives / (True Positives + False Positives)

    Arguments:
        metrics (dict): A dictionary containing the following keys:
            - Tp (int): The number of true positives.
            - Fp (int): The number of false positives.

    Returns:
        float: The precision score.
    """
    if (metrics["Tp"] + metrics["Fp"]) == 0:
        return 0
    else:
        return metrics["Tp"] / (metrics["Tp"] + metrics["Fp"])


def _recall(metrics):
    """
    Recall = True Positives / (True Positives + False Negatives)

    Arguments:
        metrics (dict): A dictionary containing the following keys:
            - Tp (int): The number of true positives.
            - Fn (int): The number of false negatives.

    Returns:
        float: The recall score.
    """
    if (metrics["Tp"] + metrics["Fn"]) == 0.:
        return 0
    else:
        return metrics["Tp"] / (metrics["Tp"] + metrics["Fn"])


def _aupr(metrics):
    """
    Area Under the Precision-Recall Curve (AUPR)

    Arguments:
        metrics (dict): A dictionary containing the following keys:
            - _Gm (numpy.ndarray): The ground truth adjacency matrix.
            - _preds (numpy.ndarray): The predicted adjacency matrix.

    Returns:
        float: The AUPR score.
    """
    true_labels = np.squeeze(np.asarray(metrics["_Gm"].ravel()))
    predicted = np.squeeze(np.asarray(metrics["_preds"].ravel()))
    try:
        _sk_precision, _sk_recall, _ = precision_recall_curve(
            true_labels, predicted)
    except ValueError:
        return 0.0
    return auc(_sk_recall, _sk_precision)


def _f1(metrics):
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Arguments:
        metrics (dict): A dictionary containing the following keys:
            - precision (float): The precision score.
            - recall (float): The recall score.

    Returns:
        float: The F1 score.
    """
    if metrics["recall"] + metrics["precision"] == 0.0:
        return 0.0
    else:
        return (
            2
            * (metrics["recall"] * metrics["precision"])
            / (metrics["recall"] + metrics["precision"])
        )


def _SHD(metrics):
    r"""
    Compute the Structural Hamming Distance.
    The Structural Hamming Distance (SHD) is a standard distance to compare
    graphs by their adjacency matrix. It consists in computing the difference
    between the two (binary) adjacency matrixes: every edge that is either
    missing or not in the target graph is counted as a mistake. Note that
    for directed graph, two mistakes can be counted as the edge in the wrong
    direction is false and the edge in the good direction is missing ; the
    `double_for_anticausal` argument accounts for this remark. Setting it to
    `False` will count this as a single mistake.

    Returns:
        int: Structural Hamming Distance (int).
            The value tends to zero as the graphs tend to be identical.
    """
    diff = np.abs(metrics["_Gm"] - metrics["_gm"])
    if metrics["_double_for_anticausal"]:
        _shd = np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1
        _shd = np.sum(diff) / 2

    return int(_shd)
