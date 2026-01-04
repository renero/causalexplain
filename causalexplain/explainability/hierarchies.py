"""
Hierarchy of links

Can I use the information above to decide wether to connect groups
of variables linked together?

"""
# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name, E1121:too-many-function-args
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

from collections import defaultdict
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from causalexplain.independence.mic import pairwise_mic


class Hierarchies:
    """
    Class representing the hierarchy of links between variables.

    Parameters
    ----------
    method : str or Callable, optional
        Method to use to compute the correlation. Default is 'spearman', but can
        also be 'pearson', 'kendall' or 'mic'.
    alpha : float, optional
        Threshold for the correlation. Default is 0.6.
    c : int, optional
        Number of clusters to be formed. Default is 15. Only valid with MIC.
    linkage_method : str, optional
        Method to use to compute the linkage. Default is 'complete'.
    correlation_th : float, optional
        Deprecated; retained for backward compatibility. Not used for pruning.
    prog_bar : bool, optional
        Whether to show a progress bar during computation. Default is False.
    verbose : bool, optional
        Whether to print additional information. Default is False.
    silent : bool, optional
        Whether to suppress all output. Default is False.
    """

    correlations = None
    linkage_mat = None

    def __init__(
            self,
            method: str = 'spearman',
            mic_alpha: float = 0.6,
            mic_c: int = 15,
            linkage_method: str = 'complete',
            correlation_th: float = None,
            prog_bar: bool = False,
            verbose: bool = False,
            silent: bool = False):
        """
        Initialize the Hierarchies object.

        Parameters
        ----------
        method : str or Callable, optional
            Method to use to compute the correlation. Default is 'spearman',
            but can also be 'pearson', 'kendall' or 'mic'.
        mic_alpha : float, optional
            Threshold for the correlation. Default is 0.6.
        mic_c : int, optional
            Number of clusters to be formed. Default is 15. Only valid with MIC.
        linkage_method : str, optional
            Method to use to compute the linkage. Default is 'complete'.
        correlation_th : float, optional
            Deprecated; retained for backward compatibility.
        prog_bar : bool, optional
            Whether to show a progress bar during computation. Default is False.
        verbose : bool, optional
            Whether to print additional information. Default is False.
        silent : bool, optional
            Whether to suppress all output. Default is False.
        """
        self.method = method
        self.alpha = mic_alpha
        self.c = mic_c
        self.linkage_method = linkage_method
        self.correlation_th = correlation_th
        self.prog_bar = prog_bar
        self.verbose_ = verbose
        self.silent_ = silent
        self.linkage_mat: np.ndarray = None

    def fit(self, X: pd.DataFrame):
        """
        Compute the hierarchy of links between variables using the correlation
        method specified in `corr_method`.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : None
            Ignored.

        Returns
        -------
        self : Hierarchies
            The fitted Hierarchies object.
        """
        # copy X into self.data
        self.data = X.copy()
        self.feature_names = list(self.data.columns)

        # Keep an empty correlated_features map for backward compatibility.
        self.correlations = self.compute_correlation_matrix(
            self.data, method=self.method, mic_alpha=self.alpha, mic_c=self.c,
            prog_bar=self.prog_bar)
        self.correlated_features = self.compute_correlated_features(
            self.correlations, self.correlation_th, self.feature_names)

        # Compute the dissimilarity matrix
        self.dissimilarity = 1 - np.abs(self.correlations)
        close_to_zero = self.dissimilarity < 1.0e-6
        self.dissimilarity[close_to_zero] = 0.0
        self.linkage_mat = linkage(squareform(
            self.dissimilarity), self.linkage_method)

        return self

    @staticmethod
    def compute_correlation_matrix(
            data: pd.DataFrame,
            method='spearman',
            mic_alpha: float = 0.6,
            mic_c: int = 15,
            prog_bar=False):
        """
        Compute the correlation matrix.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.
        method : str or Callable, optional
            Method to use to compute the correlation. Default is 'spearman',
            but can also be 'pearson', 'kendall' or 'mic'.
        prog_bar : bool, optional
            Whether to show a progress bar during computation. Default is False.

        Returns
        -------
        correlations : pd.DataFrame
            The correlation matrix.
        """
        if method in ['spearman', 'pearson', 'kendall']:
            correlations = data.corr(method=method)
        elif method == 'mic':
            correlations = pairwise_mic(
                data, alpha=mic_alpha, c=mic_c, prog_bar=prog_bar)
        else:
            raise ValueError(
                f"Unknown correlation method: {method}. Use 'spearman', "
                f"'pearson', 'kendall' or 'mic'.")

        return correlations

    @staticmethod
    def compute_correlated_features(correlations, correlation_th, feature_names, verbose=False):
        """
        Return an empty mapping (correlation pruning is deprecated).

        Parameters
        ----------
        correlations : pd.DataFrame
            The correlation matrix.
        correlation_th : float
            Deprecated; retained for backward compatibility.
        feature_names : List[str]
            The list of feature names.
        verbose : bool, optional
            Whether to print additional information. Default is False.

        Returns
        -------
        correlated_features : defaultdict(list)
            Empty mapping kept for backward compatibility.
        """
        return defaultdict(list)

    def expand_clusters_perm_importance(self, pi, ground_truth=None):
        """
        Expand the clusters of the linkage matrix to include the features that are
        in the same cluster in the permutation importance matrix.
        It expands, for each cluster, with the metrics related to correlation,
        deltas, backward PI, etc. Used to determine if some criteria can be extracted.

        Parameters
        ----------
        pi : pd.DataFrame
            Permutation importance matrix.
        ground_truth : pd.DataFrame, optional
            Ground truth matrix.

        Returns
        -------
        None
        """
        clusters = self._clusters_from_linkage(
            self.linkage_mat, self.data.columns)
        correlations = np.abs(self.data.corr("spearman"))
        for i, feature in enumerate(self.data.columns):
            print(f"{feature}")
            conn = "└─"
            for j, (name, fwd_PI) in enumerate(pi.feature_importances_[feature].items()):
                # check if this the last item in the list
                conn = "└─> " if j == len(pi.feature_importances_[
                    feature]) - 1 else "├─> "
                print(f" {conn}{name:>4s}: {fwd_PI:.02f}", end="")
                if feature in pi.feature_importances_[name].keys():
                    bwd_PI = pi.feature_importances_[name][feature]
                    print(
                        f" r({bwd_PI:.02f},∂={np.abs(fwd_PI-bwd_PI):.02f})", end="")
                    reverse = True
                else:
                    print(" r(0.00,∂=0.00)", end="")
                    reverse = False
                print(f" | c({correlations.loc[feature, name]:.02f})", end="")
                print(f" | R2({pi.regression_importances_[i]:.2f})", end="")
                degree = self._are_connected(clusters, feature, name)
                connected = True if degree is not None else False
                if connected:
                    print(f" | d{degree:.02f}", end="")
                else:
                    print(" | d0.00", end="")
                if ground_truth is None:
                    print("")
                if reverse:
                    if fwd_PI > bwd_PI:
                        check = "✅" if ground_truth.has_edge(
                            feature, name) else "❌"
                        print(f" | {feature:4s} -> {name:4s} {check}")
                    else:
                        check = "✅" if ground_truth.has_edge(
                            name, feature) else "❌"
                        print(f" | {feature:4s} <- {name:4s} {check}")
                else:
                    print("")

    def _cluster_features(self, method, threshold):
        """
        Cluster the features of the data based on the linkage matrix obtained
        from the hierarchical clustering.

        Parameters
        ----------
        method : str
            Method to compute the correlation.
        threshold : float
            Threshold for the correlation.

        Returns
        -------
        clustered : pd.DataFrame
            Correlation matrix.
        sorted_colnames : List[str]
            List of sorted column names.
        """
        # Keep the indices to sort labels
        labels = fcluster(self.linkage_mat, threshold, criterion='distance')
        labels_order = np.argsort(labels)
        sorted_colnames = self.data.columns[labels_order]

        # Build a new dataframe with the sorted columns
        for idx, i in enumerate(sorted_colnames):
            if idx == 0:
                clustered = pd.DataFrame(self.data[i])
            else:
                df_to_append = pd.DataFrame(self.data[i])
                clustered = pd.concat([clustered, df_to_append], axis=1)
        return clustered.corr(method), sorted_colnames

    def hierarchical_dissimilarities(self):
        """
        Compute the dissimilarities between features in a hierarchical clustering.

        Returns
        -------
        hierarchical_dissimilarity : pd.DataFrame
            Dissimilarities between features.
        """
        feature_names = list(self.data.columns)
        hierarchical_dissimilarity = pd.DataFrame(columns=feature_names)
        clusters = self._clusters_from_linkage(self.linkage_mat, feature_names)
        for feature in feature_names:
            remaining_features = [f for f in feature_names if f != feature]
            for name in remaining_features:
                hierarchical_dissimilarity.loc[feature, name] = self._are_connected(
                    clusters, feature, name)

        hierarchical_dissimilarity[hierarchical_dissimilarity.isna()] = 0.0

        return hierarchical_dissimilarity

    def _clusters_from_linkage(self, linkage_mat: np.ndarray, features: List[str]):
        """
        Get the clusters from the linkage matrix.

        Parameters
        ----------
        linkage_mat : np.array
            The linkage matrix.
        features : List[str]
            The list of feature names.

        Returns
        -------
        clusters : dict
            The clusters from the linkage matrix.
        """
        clusters = {}
        Z = linkage_mat
        new_cluster_id = len(features)
        for n in range(Z.shape[0]):
            new_cluster = new_cluster_id
            new_cluster_id += 1
            feature_a = int(Z[n][0])
            feature_b = int(Z[n][1])
            if feature_a < len(features):
                feature_a = features[feature_a]
            else:
                feature_a = f'K{feature_a}'
            if feature_b < len(features):
                feature_b = features[feature_b]
            else:
                feature_b = f'K{feature_b}'
            clusters[f"K{new_cluster}"] = [(feature_a, feature_b), Z[n][2]]

        return clusters

    @staticmethod
    def _get_cluster(clusters: List[str], node: str) -> str:
        """
        Get the cluster that contains the given node.

        Parameters
        ----------
        clusters : List[str]
            The list of clusters.
        node : str
            The node to search for.

        Returns
        -------
        cluster : str
            The cluster that contains the given node.
        """
        for k in clusters.keys():
            if node in clusters[k][0]:
                return k
        return None

    @staticmethod
    def _is_cluster(node: str) -> bool:
        """
        Check if the given node is a cluster.

        Parameters
        ----------
        node : str
            The node to check.

        Returns
        -------
        is_cluster : bool
            True if the node is a cluster, False otherwise.
        """
        return node.startswith('K')

    @staticmethod
    def _contains_a_cluster(clusters: List[str], node: str) -> bool:
        """
        Check if the given node contains a cluster.

        Parameters
        ----------
        clusters : List[str]
            The list of clusters.
        node : str
            The node to check.

        Returns
        -------
        contains_cluster : bool
            True if the node contains a cluster, False otherwise.
        """
        return any([Hierarchies._is_cluster(n) for n in clusters[node][0]])

    @staticmethod
    def _get_cluster_element(clusters: List[str], node: str) -> str:
        """
        Get the cluster element from the given node.

        Parameters
        ----------
        clusters : List[str]
            The list of clusters.
        node : str
            The node to get the cluster element from.

        Returns
        -------
        cluster_element : str
            The cluster element from the given node.
        """
        for n in clusters[node][0]:
            if Hierarchies._is_cluster(n):
                return n
        return None

    @staticmethod
    def _in_cluster(cluster, node):
        """
        Check if the given node is in the given cluster.

        Parameters
        ----------
        cluster : str
            The cluster to check.
        node : str
            The node to check.

        Returns
        -------
        in_cluster : bool
            True if the node is in the cluster, False otherwise.
        """
        return node in cluster[0]

    @staticmethod
    def _in_same_cluster(clusters: List[str], node1: str, node2: str) -> str:
        """
        Check if the given nodes are in the same cluster.

        Parameters
        ----------
        clusters : List[str]
            The list of clusters.
        node1 : str
            The first node to check.
        node2 : str
            The second node to check.

        Returns
        -------
        cluster : str
            The cluster that contains both nodes, or None if they are not in the same cluster.
        """
        for k in clusters.keys():
            if node1 in clusters[k][0] and node2 in clusters[k][0]:
                return k
        return None

    def _are_connected(
            self,
            clusters: List[str],
            node1: str,
            node2: str) -> Tuple[bool, float]:
        """
        Determine if two nodes are connected in the hierarchical clustering represented
        in the clusters dictionary, obtained from the `linkage` function.

        Parameters
        ----------
            - clusters (List[str])
                List of clusters.
            - node1 (str)
                First node.
            - node2 (str)
                Second node.

        Returns
        -------
            - float
                Degree of disimilarity between the nodes. The higher the value, the more
                dissimilar the nodes are. If None is returned, the nodes are not connected.
        """
        k = Hierarchies._in_same_cluster(clusters, node1, node2)
        if k is not None:
            # Obtain the second valud in the dictionary of clusters for the pair of nodes
            return clusters[k][1]
        cluster1 = Hierarchies._get_cluster(clusters, node1)
        cluster2 = Hierarchies._get_cluster(clusters, node2)
        if Hierarchies._contains_a_cluster(clusters, cluster1):
            ref_cluster = Hierarchies._get_cluster_element(clusters, cluster1)
            if Hierarchies._in_cluster(clusters[ref_cluster], node2):
                disimilarity = clusters[Hierarchies._get_cluster(
                    clusters, node1)][1]
                return disimilarity
        if Hierarchies._contains_a_cluster(clusters, cluster2):
            ref_cluster = Hierarchies._get_cluster_element(clusters, cluster2)
            if Hierarchies._in_cluster(clusters[ref_cluster], node1):
                disimilarity = clusters[Hierarchies._get_cluster(
                    clusters, node2)][1]
                return disimilarity
        return None

def _get_directed_pair(g, u, v):
    if g.has_edge(u, v) and not g.has_edge(v, u):
        return (u, v)
    elif g.has_edge(v, u) and not g.has_edge(u, v):
        return (v, u)
    else:
        return None


def _get_direction(g: nx.DiGraph, u: str, v: str) -> Tuple[str, Union[int, None]]:
    """
    Returns the direction of the edge between u and v.

    Arguments:
    ----------
        - g (networkx.DiGraph): The graph to search in.
        - u (str): The source node.
        - v (str): The target node.

    Returns:
    --------
        - (str, int): A string respresenting the direction of the edge
            between u and v, and an integer representing it. The string
            can be either '-->', '<--', '<->' or ' · '. And the integer
            can be either 1, -1, 0 or None for each possibility.

    """

    if not g.has_node(u) or not g.has_node(v):
        raise ValueError(f"Node(s) {u} and/or {v} not in graph.")

    if g.has_edge(u, v) and not g.has_edge(v, u):
        return "-->", 1
    elif g.has_edge(v, u) and not g.has_edge(u, v):
        return "<--", -1
    elif g.has_edge(u, v) and g.has_edge(v, u):
        return "<->", 0
    else:
        return " · ", None


def connect_isolated_nodes(G, linkage_mat, feature_names, verbose=False):
    """
    Connect isolated nodes in the graph, based on their relationship in the
    hierarchical clustering provided through the linkage_mat.

    Arguments:
    ----------
        - G (networkx.DiGraph): The graph to search in.
        - linkage_mat (np.ndarray): The linkage matrix.
        - feature_names (List[str]): The list of feature names.
        - verbose (bool): Whether to print information about the process.

    Returns:
    --------
        - networkx.DiGraph: The graph with connected isolated nodes.

    Notes:
    ------
        The linkage matrix is a matrix of the following form:

            [i, j, distance, n_items, direction]

        where:
            - i, j: The indices of the two nodes in the linkage.
            - distance: The distance between the two nodes.
            - n_items: The number of items in the two nodes.
            - direction: The direction of the edge between the two nodes.

        and it can be obtained using the function: `compute_hierarchies`
    """
    node_names = [f for f in feature_names]
    G_h = nx.DiGraph()
    G_h.add_edges_from(G.edges(data=False))

    for i in range(linkage_mat.shape[0]):
        _, num = linkage_mat[i][2], int(linkage_mat[i][3])
        if num > 2:
            continue
        u, v = node_names[int(linkage_mat[i][0])
                          ], node_names[int(linkage_mat[i][1])]
        arrow, direction = _get_direction(G_h, u, v)

        # Consider only clusters formed by two nodes, not other clusters.
        features_cluster = True if num == 2 else False
        if direction is None and features_cluster:
            G_h.add_edge(u, v, weight=None)
            if verbose:
                print(f"Adding edge {u} {arrow} {v}")

    return G_h


def connect_hierarchies(G, linkage_mat, feature_names, verbose=False):
    cluster_id = len(feature_names)
    node_names = [f for f in feature_names]
    clusters = {}
    G_h = nx.DiGraph()
    G_h.add_edges_from(G.edges(data=True))
    if verbose:
        print(
            f"{'from':>4s}  :  {'to':<4s}  {'weight':6s}  {'n.items':7s} {'names':5s}")
    for i in range(linkage_mat.shape[0]):
        u, v = node_names[int(linkage_mat[i][0])
                          ], node_names[int(linkage_mat[i][1])]
        weight, num = linkage_mat[i][2], int(linkage_mat[i][3])
        arrow, _ = _get_direction(G_h, u, v)
        kname = f"K#{cluster_id}"
        node_names.append(kname)
        clusters[kname] = (u, v)
        if verbose:
            print(f"{u:>4s} {arrow:^s} {v:<4s}  {weight:6.4f}  {str(num):^7s} {kname:^5s}",
                  end="")
        cluster_id += 1

        # If I'm forming the first cluster, nothing to do.
        if cluster_id <= len(feature_names)+1:
            if verbose:
                print()
            continue

        # Determine if source is a node or a cluster
        if u in clusters:
            pair = _get_directed_pair(G_h, *clusters[u])
            if pair is not None:
                source = pair[1]
        else:
            source = u

        # Determine if target is a node or cluster. If node, continue
        if v in clusters:
            pair = _get_directed_pair(G_h, *clusters[v])
            if pair is not None:
                target = pair[0]
            else:
                if verbose:
                    print()
                continue
        else:
            if verbose:
                print()
            continue

        if _get_directed_pair(G_h, source, target) is None:
            if verbose:
                print(f" Add {source} --> {target}")
            G_h.add_edge(source, target)
        else:
            if verbose:
                print()
            continue

    return G_h


def plot_dendogram_correlations(correlations, feature_names: List[str], **kwargs):
    """
    Plot the dendrogram of the correlation matrix.

    Parameters
    ----------
        - correlations (pd.DataFrame)
            Correlation matrix.
        - feature_names (List[str])
            List of feature names.
        - kwargs
            Keyword arguments to be passed to the plot_dendogram function.
    """
    figsize = kwargs.get("figsize", (5, 3))
    dissimilarity = 1 - abs(correlations)
    Z = linkage(squareform(dissimilarity), 'single')
    plt.figure(figsize=figsize)
    ax = plt.gca()
    dendrogram(Z, labels=feature_names, orientation='left', ax=ax)
    plt.show()
    return Z


if __name__ == "__main__":
    m_alpha = 0.8
    m_c = 15

    test_data = pd.read_csv("/Users/renero/phd/data/generated_linear_10.csv")
    h = Hierarchies(method='mic', mic_alpha=m_alpha, mic_c=m_c)
    h.fit(test_data)
    h.plot()
    plt.show()
