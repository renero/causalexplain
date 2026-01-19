# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

import logging
from itertools import chain, combinations, permutations
import os

import networkx as nx
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from mlforge.progbar import ProgBar  #type: ignore

from ...common import utils
from .ci_tests import chi_square, pearsonr
from .estimators import StructureEstimator
from .pdag import PDAG
from ...metrics.compare_graphs import evaluate_graph

SHOW_PROGRESS = True


class PC(StructureEstimator):
    """
    Class for constraint-based estimation of DAGs using the PC algorithm
    from a given data set.  Identifies (conditional) dependencies in data
    set using chi_square dependency test and uses the PC algorithm to
    estimate a DAG pattern that satisfies the identified dependencies. The
    DAG pattern can then be completed to a faithful DAG, if possible.

    References
    ----------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and
        Techniques, 2009, Section 18.2
    [2] Neapolitan, Learning Bayesian Networks, Section 10.1.2 for the PC algorithm
        (page 550),
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf # noqa
    """

    dag = None
    pdag = None
    is_fitted_ = False
    metrics = None

    def __init__(self, name: str, independencies=None, **kwargs):
        """
        Class intialization.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.  (If some
            values in the data are missing the data cells should be set to
            `numpy.NaN`.  Note that pandas converts each column containing
            `numpy.NaN`s to dtype `float`.)
        independencies: Independencies object
            Independencies object containing a set of conditional independence
            assertions that will be used to estimate the DAG skeleton.  If
            independencies is None, all conditional independence assertions
            will be tested.
        kwargs: key-value arguments
            Additional arguments passed to the `StructureEstimator` base class.
            - variant: str (one of "orig", "stable", "parallel")ss
        """
        super().__init__(independencies=independencies)
        self.name = name
        self.variant = kwargs.get("variant", "stable")
        self.ci_test = kwargs.get("ci_test", "pearsonr")
        self.max_cond_vars = kwargs.get("max_cond_vars", 5)
        self.return_type = kwargs.get("return_type", "dag")
        self.significance_level = kwargs.get("significance_level", 0.01)
        self.n_jobs = kwargs.get("n_jobs", -1)
        self.prog_bar = kwargs.get("prog_bar", True)

    def fit(self, X, **kwargs):
        """Fit the PC algorithm to the input data."""

        assert isinstance(
            X, pd.DataFrame), "X must be a pandas DataFrame object"

        self.data = X
        self.feature_names = list(X.columns)
        state_names = kwargs.get("state_names", None)
        complete_samples_only = kwargs.get("complete_samples_only", True)
        self._init_data(data=self.data, state_names=state_names,
                        complete_samples_only=complete_samples_only)

        # Step 0: Do checks that the specified parameters are correct, else
        # throw meaningful error.
        if self.variant not in ("orig", "stable", "parallel"):
            raise ValueError(
                f"variant must be one of: orig, stable, or parallel. "
                f"Got: {self.variant}"
            )
        if (not callable(self.ci_test)) and (self.ci_test not in (
                "chi_square", "independence_match", "pearsonr")):
            raise ValueError(
                "ci_test must be a callable or one of: chi_square, pearsonr, "
                "independence_match"
            )

        if (self.ci_test in ("chi_square", "pearsonr")) and (self.data is None):
            raise ValueError(
                "For using Chi Square or Pearsonr, data arguement must be specified"
            )

        # Step 1: Run the PC algorithm to build the skeleton and get the separating
        # sets.
        skel, separating_sets = self.build_skeleton(**kwargs)

        if self.return_type.lower() == "skeleton":
            return skel, separating_sets

        # Step 2: Orient the edges based on build the PDAG/CPDAG.
        self.pdag = self.skeleton_to_pdag(skel, separating_sets)

        self.dag = self.pdag.to_dag()

        # Step 3: Either return the CPDAG or fully orient the edges to build a DAG.
        if self.return_type.lower() in ("pdag", "cpdag"):
            self.is_fitted_ = True
            return self.pdag
        if self.return_type.lower() == "dag":
            self.is_fitted_ = True
            return nx.DiGraph(self.pdag.to_dag())
        raise ValueError(
            f"return_type must be one of: dag, pdag, cpdag, or skeleton. "
            f"Got: {self.return_type}"
        )

    def fit_predict(self, train, test, ref_graph: nx.DiGraph = None, **kwargs):
        self.fit(train, **kwargs)
        if ref_graph:
            self.metrics = evaluate_graph(
                ref_graph, self.dag, self.feature_names)
        return self.dag

    def build_skeleton(self, **kwargs):
        """
        Estimates a graph skeleton (UndirectedGraph) from a set of independencies
        using (the first part of) the PC algorithm. The independencies can either be
        provided as an instance of the `Independencies`-class or by passing a
        decision function that decides any conditional independency assertion.
        Returns a tuple `(skeleton, separating_sets)`.

        If an Independencies-instance is passed, the contained IndependenceAssertions
        have to admit a faithful BN representation. This is the case if
        they are obtained as a set of d-seperations of some Bayesian network or
        if the independence assertions are closed under the semi-graphoid axioms.
        Otherwise the procedure may fail to identify the correct structure.

        Parameters
        ----------

        Returns
        -------
        skeleton: UndirectedGraph
            An estimate for the undirected graph skeleton of the BN underlying the data.

        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes then
            conditionally independent. (needed for edge orientation procedures)

        References
        ----------
        [1] Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm 10.2 (page 550)
            http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf # noqa
        [2] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
            Section 3.4.2.1 (page 85), Algorithm 3.3

        """

        # Initialize initial values and structures.
        lim_neighbors = 0
        separating_sets = dict()
        if self.ci_test == "chi_square":
            ci_test_fn = chi_square
        elif self.ci_test == "pearsonr":
            ci_test_fn = pearsonr
        elif callable(self.ci_test):
            ci_test_fn = self.ci_test
        else:
            raise ValueError(
                f"ci_test must either be chi_square, pearsonr, independence_match, "
                f"or a function. Got: {ci_test_fn}"
            )

        pbar = ProgBar("PC", self.max_cond_vars) if self.prog_bar else None

        # Step 1: Initialize a fully connected undirected graph
        graph = nx.complete_graph(n=self.variables, create_using=nx.Graph)

        # Exit condition: 1. If all the nodes in graph has less than
        # `lim_neighbors` neighbors. or  2. `lim_neighbors` is greater than
        # `max_conditional_variables`.
        while not all(
            [len(list(graph.neighbors(var))) <
             lim_neighbors for var in self.variables]
        ):

            # Step 2: Iterate over the edges and find a conditioning set of
            # size `lim_neighbors` which makes u and v independent.
            if self.variant == "orig":
                for (u, v) in graph.edges():
                    for separating_set in chain(
                        combinations(set(graph.neighbors(u)) -
                                     set([v]), lim_neighbors),
                        combinations(set(graph.neighbors(v)) -
                                     set([u]), lim_neighbors),
                    ):
                        # If a conditioning set exists remove the edge, store
                        # the separating set and move on to finding conditioning set for
                        # next edge.
                        if ci_test_fn(
                            u,
                            v,
                            separating_set,
                            data=self.data,
                            independencies=self.independencies,
                            significance_level=self.significance_level,
                            **kwargs,
                        ):
                            separating_sets[frozenset((u, v))] = separating_set
                            graph.remove_edge(u, v)
                            break

            elif self.variant == "stable":
                # In case of stable, precompute neighbors as this is the stable
                # algorithm.
                neighbors = {node: set(graph[node]) for node in graph.nodes()}
                for (u, v) in graph.edges():
                    for separating_set in chain(
                        combinations(set(graph.neighbors(u)) -
                                     set([v]), lim_neighbors),
                        combinations(set(graph.neighbors(v)) -
                                     set([u]), lim_neighbors),
                    ):
                        # If a conditioning set exists remove the edge, store the
                        # separating set and move on to finding conditioning set for
                        # next edge.
                        if ci_test_fn(
                            u,
                            v,
                            separating_set,
                            data=self.data,
                            independencies=self.independencies,
                            significance_level=self.significance_level,
                            **kwargs,
                        ):
                            separating_sets[frozenset((u, v))] = separating_set
                            graph.remove_edge(u, v)
                            break

            elif self.variant == "parallel":
                neighbors = {node: set(graph[node]) for node in graph.nodes()}

                def _parallel_fun(u, v):
                    for separating_set in chain(
                        combinations(set(graph.neighbors(u)) -
                                     set([v]), lim_neighbors),
                        combinations(set(graph.neighbors(v)) -
                                     set([u]), lim_neighbors),
                    ):
                        if ci_test_fn(
                            u,
                            v,
                            separating_set,
                            data=self.data,
                            independencies=self.independencies,
                            significance_level=self.significance_level,
                            **kwargs,
                        ):
                            return (u, v), separating_set

                results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                    delayed(_parallel_fun)(u, v) for (u, v) in graph.edges()
                )
                for result in results:
                    if result is not None:
                        (u, v), sep_set = result
                        graph.remove_edge(u, v)
                        separating_sets[frozenset((u, v))] = sep_set

            else:
                raise ValueError(
                    f"variant must be one of (orig, stable, parallel). "
                    f"Got: {self.variant}"
                )

            # Step 3: After iterating over all the edges, expand the search space by
            # increasing the size of conditioning set by 1.
            if lim_neighbors >= self.max_cond_vars:
                logging.info(
                    "Reached maximum number of allowed conditional variables. Exiting"
                )
                break
            lim_neighbors += 1

            if self.prog_bar and SHOW_PROGRESS:
                pbar.update_subtask("PC", 1)

        pbar.remove("PC") if self.prog_bar else None

        return graph, separating_sets

    @staticmethod
    def skeleton_to_pdag(skeleton, separating_sets):
        """Orients the edges of a graph skeleton based on information from
        `separating_sets` to form a DAG pattern (DAG).

        Parameters
        ----------
        skeleton: UndirectedGraph
            An undirected graph skeleton as e.g. produced by the
            estimate_skeleton method.

        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes then
            conditionally independent. (needed for edge orientation)

        Returns
        -------
        pdag: DAG
            An estimate for the DAG pattern of the BN underlying the data. The
            graph might contain some nodes with both-way edges (X->Y and Y->X).
            Any completion by (removing one of the both-way edges for each such
            pair) results in a I-equivalent Bayesian network DAG.

        References
        ----------
        Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm 10.2 (page 550)
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf # noqa

        """

        pdag = skeleton.to_directed()
        node_pairs = list(permutations(pdag.nodes(), 2))

        # 1) for each X-Z-Y, if Z not in the separating set of X,Y, then orient
        # edges as X->Z<-Y (Algorithm 3.4 in Koller & Friedman PGM, page 86)
        for pair in node_pairs:
            X, Y = pair
            if not skeleton.has_edge(X, Y):
                for Z in set(skeleton.neighbors(X)) & set(skeleton.neighbors(Y)):
                    if Z not in separating_sets[frozenset((X, Y))]:
                        pdag.remove_edges_from([(Z, X), (Z, Y)])

        progress = True
        while progress:  # as long as edges can be oriented (removed)
            num_edges = pdag.number_of_edges()

            # 2) for each X->Z-Y, orient edges to Z->Y
            # (Explanation in Koller & Friedman PGM, page 88)
            for pair in node_pairs:
                X, Y = pair
                if not pdag.has_edge(X, Y):
                    for Z in (set(pdag.successors(X)) - set(pdag.predecessors(X))) & (
                        set(pdag.successors(Y)) & set(pdag.predecessors(Y))
                    ):
                        pdag.remove_edge(Y, Z)

            # 3) for each X-Y with a directed path from X to Y, orient edges to X->Y
            for pair in node_pairs:
                X, Y = pair
                if pdag.has_edge(Y, X) and pdag.has_edge(X, Y):
                    for path in nx.all_simple_paths(pdag, X, Y):
                        is_directed = True
                        for src, dst in list(zip(path, path[1:])):
                            if pdag.has_edge(dst, src):
                                is_directed = False
                        if is_directed:
                            pdag.remove_edge(Y, X)
                            break

            # 4) for each X-Z-Y with X->W, Y->W, and Z-W, orient edges to Z->W
            for pair in node_pairs:
                X, Y = pair
                for Z in (
                    set(pdag.successors(X))
                    & set(pdag.predecessors(X))
                    & set(pdag.successors(Y))
                    & set(pdag.predecessors(Y))
                ):
                    for W in (
                        (set(pdag.successors(X)) - set(pdag.predecessors(X)))
                        & (set(pdag.successors(Y)) - set(pdag.predecessors(Y)))
                        & (set(pdag.successors(Z)) & set(pdag.predecessors(Z)))
                    ):
                        pdag.remove_edge(W, Z)

            progress = num_edges > pdag.number_of_edges()

        # TODO: This is temp fix to get a PDAG object.
        edges = set(pdag.edges())
        undirected_edges = []
        directed_edges = []
        for u, v in edges:
            if (v, u) in edges:
                undirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))
        return PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)


def main(dataset_name, input_path=None, output_path=None, save=False):
    if input_path is None:
        input_path=os.path.expanduser("~/phd/data/sachs/")
    if output_path is None:
        output_path=os.path.expanduser("~/phd/output/RC4/sachs/compared/")
    ref_graph = utils.graph_from_dot_file(f"{input_path}{dataset_name}.dot")
    data = pd.read_csv(f"{input_path}{dataset_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    train = data.sample(frac=0.8, random_state=42)
    test = data.drop(train.index)

    pc = PC(name=dataset_name, variant="stable",
            ci_test="pearsonr", max_cond_vars=5)
    pc.fit_predict(train=data, test=None, ref_graph=ref_graph)

    for edge in pc.dag.edges():
        print(edge)
    print(pc.metrics)

    utils.graph_to_adjacency_file(
        pc.dag, f"{output_path}{dataset_name}_PC.adj", list(data.columns))
    print(f"DAG saved to {output_path}{dataset_name}_PC.adj")

    # if save:
    #     where_to = utils.save_experiment(rex.name, output_path, rex)
    #     print(f"Saved '{rex.name}' to '{where_to}'")


if __name__ == "__main__":
    main("sachs_long")
