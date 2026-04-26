"""
Application of early stage FCI rules to setup the skeleton of a causal graph to
an existing graph in order to prune potentially spurious edges.

(c) J. Renero, 2022
"""

import itertools
import logging
from collections import defaultdict

import networkx as nx
import numpy as np
import pygam
from hyppo.independence import Hsic
from pandas import DataFrame
from mlforge.progbar import ProgBar


# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid_name, C0116:missing-function-docstring, R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements, R1702:too-many-branches
# pylint: disable=W0106:expression-not-assigned


log = logging.getLogger(__name__)


class GraphIndependence(object):
    """
    Class for removing independent edges from a causal graph.

    Parameters:
    -----------
    - base_graph (nx.DiGraph): The graph to be cleaned
    - condlen (int): The number of conditioning sets to use in the
        independence test. Defaults to 1, meaning that the cond. indep.
        test are of length=1.
    - condsize (int): The size of the conditioning sets to use in the
        independence test. Default is 0, which means that conditioning
        sets will be based on direct connections
    - prog_bar (bool): Whether to show a progress bar
    - verbose (bool): Whether to print debug messages
    - silent (bool): Whether to suppress all output

    Methods:
    --------
    - fit(X: DataFrame, y=None) -> Tuple[nx.DiGraph, Dict[str, List[Tuple[str, Tuple[str]]]]]:
        Remove edges from the graph that are conditionally independent.
    - predict() -> nx.DiGraph:
        Predicts the causal graph using the current independence tests and returns
        the resulting graph.
    - fit_predict(X: DataFrame, y=None) -> nx.DiGraph:
        Fits the model to the data and returns predictions.
    """

    def __init__(
            self,
            base_graph,
            condlen: int = 1,
            condsize: int = 0,
            max_rows: int = 500,
            prog_bar: bool = True,
            verbose: bool = False,
            silent: bool = False):

        self.base_graph = base_graph
        self.condlen = condlen
        self.condsize = condsize
        self.max_rows = max_rows
        self.prog_bar = prog_bar
        self.verbose = verbose
        self.silent_ = silent

        self._fit_desc = "Removing independent edges"

    def fit(self, X: DataFrame, y=None):
        """
        Remove edges from the graph that are conditionally independent.

        Arguments:
        ----------
        - base_graph (nx.DiGraph): The graph to be cleaned
        - data (DataFrame): The data used to discover the causal graph
        - condlen (int): The number of conditioning sets to use in the
            independence test. Defaults to 1, meaning that the cond. indep.
            test are of length=1.
        - condsize (int): The size of the conditioning sets to use in the
            independence test. Default is 0, which means that conditioning
            sets will be based on direct connections
        - prog_bar (bool): Whether to show a progress bar
        - verbose (bool): Whether to print debug messages

        Returns:
        --------
            nx.DiGraph: The cleaned graph
            actions: A dictionary of actions that were taken
        """
        # TODO: This was ON in the original code, but it seems to be a bug
        # condlen = 1
        # condsize = 0

        self.data = X.copy()
        self.y = y.copy() if y is not None else None

        # Check if base_graph contains data columns
        self.feature_names = list(self.data.columns)
        self.actions = defaultdict(list)
        self.sepset = {}
        for p in itertools.permutations(self.feature_names, 2):
            self.sepset[p] = ()
        # edge = list(self.base_graph.edges)

        self.G_skl = nx.DiGraph()
        self.G_skl.add_nodes_from(self.base_graph.nodes(data=True))
        self.G_skl.add_edges_from(self.base_graph.edges(data=True))

        # check if all column names in experiment.data correspond to nodes in G_skl.
        # If not add them
        for col in self.data.columns:
            if col not in self.G_skl.nodes():
                self.G_skl.add_node(col)

        pbar = ProgBar().start_subtask("Graph.Indep_fit", len(self.feature_names))
        for feature_idx, feature_name in enumerate(self.feature_names):
            self._remove_independent_edges(
                feature_name, self.condlen, self.condsize)
            pbar.update_subtask("Graph.Indep_fit", feature_idx+1)

        pbar.remove("Graph.Indep_fit")
        return self

    def predict(self):
        """
        Predicts the causal graph using the current independence tests and returns the resulting graph.

        Returns:
            The predicted causal graph.
        """
        return self.G_skl

    def fit_predict(self, X: DataFrame, y=None):
        """
        Fits the model to the data and returns predictions.

        Parameters:
        X (DataFrame): The input data to fit the model on.
        y (optional): The target variable to fit the model on.

        Returns:
        The predictions made by the model.
        """
        self.fit(X, y)
        return self.predict()

    def _gen_cond_sets(self, x, y, size):
        """ A function to build the set of conditioning sets to be for
        variables x and y on graph g of a certain size when generating a
        skeleton

        Arguments:
        ----------
        - x (str): One variable being tested for independence,
        - y (str): The other variable being tested for independence,
        - size (int): the size of each conditioning set to be returned

        Returns:
        --------
        A list of conditioning sets to be tested
        """
        # Handle size 0 case
        if size == 0:
            return [()]
        # Get all neighbors of x in g
        adjy = self.G_skl.neighbors(x)
        # Remove y from neighbour list
        adj = [node for node in adjy if node != y]
        # Generate all unique combinations of len size
        combos = list(itertools.combinations(adj, size))
        return combos

    @staticmethod
    def _residuals(x, y, Z):
        """
        Train a GAM with cond_set to predict "i", and then "j". For both cases
        get the residuals.

        Arguments:
        - x (np.array): x-values of the data
        - y (np.array): y-values of the data
        - Z (np.array): Z-values of the data

        Returns:
        - r_x (np.array): residuals of GAM fitted to predict x from Z
        - r_y (np.array): residuals of GAM fitted to predict y from Z
        """
        g = pygam.GAM()
        g.fit(Z, x)
        residuals_x = g.deviance_residuals(Z, x)
        g.fit(Z, y)
        residuals_y = g.deviance_residuals(Z, y)
        return residuals_x, residuals_y

    def _cond_indep_test(self, x: np.array, y: np.array, Z: np.array = None):
        """
        Perform the conditional independence test configured in the
        main class attribute

        Arguments:
        ----------
        - x (np.array): The first variable being tested for independence
        - y (np.array): The second variable being tested for independence
        - Z (np.array): optional (default=None) The conditioning set, if any

        Returns:
        --------
        - bool: True if x is independent of y, False otherwise
        - float: the t-statistic of the HSIC independence test
        - float: the p-value of the independence test
        """
        if x.shape[0] > self.max_rows:
            idx = np.random.choice(x.shape[0], self.max_rows, replace=False)
            x = x[idx]
            y = y[idx]
            if Z is not None:
                Z = Z[idx]

        if Z is None:
            stat, pval = Hsic().test(x, y)
        else:
            r_x, r_y = self._residuals(x, y, Z)
            stat, pval = Hsic().test(r_x, r_y)
        indep = pval > 0.05
        return indep, stat, pval

    def _update_graph(self, x, y, cs):
        log.debug("    ! %s⫫%s|(%s) Independent!", x, y, ','.join(cs))
        self.G_skl.remove_edge(x, y)
        self.sepset[(x, y)] = cs
        self.sepset[(y, x)] = cs

    def _update_actions(self, tup):
        if tup is not None:
            if tup[2]:
                self.actions[tup[0]].append((tup[1], tup[2]))
            else:
                self.actions[tup[0]].append((tup[1], ()))

    @staticmethod
    def _empty(x: list):
        if len(x) == 0:
            return True
        else:
            return len(x[0]) == 0

    def _test_cond_independence(self, x_name, y_name, cond_sets):
        """
        Test if X and Y are independent, using all possible conditioning sets
        in the graph from nodes adjacent to X. If it is found that X and Y are
        independent, exploration finish, and the edge between X and Y is
        removed from the graph.
        """
        L = len(cond_sets)
        x = self.data.loc[:, x_name].values.reshape(-1, 1)
        y = self.data.loc[:, y_name].values.reshape(-1, 1)

        for idx, cond_set in enumerate(cond_sets):
            Z = self.data.loc[:, cond_set].values if not self._empty(
                cond_set) else None
            independent, _, _ = self._cond_indep_test(x, y, Z)
            if independent:
                self._update_graph(x_name, y_name, cond_set)
                return x_name, y_name, cond_set

            log.debug("    - [%02d/%02d] %s⫫%s|(%s) - Not independent",
                      idx + 1, L, x_name, y_name, ','.join(cond_set))

        return None

    def _remove_independent_edges(self, x, condlen, condsize):
        if self.actions is None:
            self.actions = defaultdict(list)

        neighbors = list(self.G_skl.neighbors(x))
        neighbors_str = ",".join(list(neighbors))
        log.debug("> Iterating over neighbors: %s", neighbors_str)
        for ny, y in enumerate(neighbors):
            log.debug("+ y = %s; %d/%d", y, ny + 1, len(neighbors))
            # Generate the conditioning sets needed for independence tests
            condSets = self._gen_cond_sets(x, y, condsize)
            if len(condSets) == 0:
                continue
            condlen += len(condSets)
            # Test for independence with each conditioning set
            cs = ",".join([f"({','.join(cs)})" for cs in condSets])
            msg = f"  > Check {x}⫫{y} over {len(condSets)} CS: [{cs}]"
            log.debug("%s", msg)
            tup = self._test_cond_independence(x, y, condSets)
            if tup is not None:
                self._update_actions(tup)

        return condlen

    def compute_cond_indep_pvals(self):
        """
        Perform the `_test_cond_independence` on all pairs of nodes in the graph, and
        store the p_value (3rd element in the return tuple) on the class
        `cond_indep_pvals`
        """
        self.cond_indep_pvals = {}
        if self.data.shape[0] > self.max_rows:
            log.debug("Data has more than %d rows. Sampling for cond. indep. test.",
                      self.max_rows)

        for x in self.feature_names:
            for y in self.feature_names:
                if x == y:
                    continue
                X = self.data.loc[:, x].values.reshape(-1, 1)
                Y = self.data.loc[:, y].values.reshape(-1, 1)
                _, _, pval = self._cond_indep_test(X, Y, None)
                self.cond_indep_pvals[(x, y)] = pval

        return self.cond_indep_pvals
