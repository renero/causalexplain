"""
FCI algorithm.

A graph learner which wraps the implementation of the FCI algorithm
in the (c) causallearn library (https://github.com/py-why/causal-learn).

"""

# pylint: disable=E1101:no-member
# pylint: disable=W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=W0106:expression-not-assigned
# pylint: disable=C0103:invalid-name, C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, R0902:too-many-instance-attributes
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=R1702:too-many-branches

import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci

from causalexplain.common import utils
from causalexplain.metrics.compare_graphs import evaluate_graph


class FCI:
    """
    A graph learner which wraps the implementation of the FCI algorithm
    in (c) causallearn library (https://github.com/py-why/causal-learn).
    """

    def __init__(
            self,
            name: str,
            independence_test_method: str = "fisherz",
            alpha: float = 0.05,
            depth: int = -1,
            max_path_length: int = -1,
            verbose: bool = False,
            background_knowledge=None,
            show_progress: bool = True,
            node_names=None,
            causes_only=True):
        """
        Initialize the FCI algorithm creating an FCI learner.

        Parameters
        ----------
        name: str, name of the experiment
        independence_test_method: str, name of the function of the independence 
            test method being used: [fisherz, chisq, gsq, kci]
            - fisherz: Fisher's Z conditional independence test
            - chisq: Chi-squared conditional independence test
            - gsq: G-squared conditional independence test
            - kci: Kernel-based conditional independence test
        alpha: float, desired significance level of independence tests 
            (p_value) in (0,1)
        depth: The depth for the fast adjacency search, or -1 if unlimited
        max_path_length: the maximum length of any discriminating path, 
            or -1 if unlimited.
        verbose: True is verbose output should be printed or logged
        background_knowledge: background knowledge
        causes_only: bool, if True, only causes are returned
            (default: True), by filtering the CPDAG and considering only
            the edges indicating a causal relationship.
        """
        super().__init__()
        self.name = name
        self.independence_test_method = independence_test_method
        self.alpha = alpha
        self.depth = depth
        self.max_path_length = max_path_length
        self.verbose = verbose
        self.background_knowledge = background_knowledge
        self.prog_bar = show_progress
        self.node_names = node_names
        self.causes_only = causes_only

    def fit_predict(self,
                    X: pd.DataFrame,
                    X_test: pd.DataFrame = None,
                    ref_graph: nx.DiGraph = None):
        """Fit the model and return the estimated graph."""
        graph, edges = fci(
            X.values,
            indep_test_method=self.independence_test_method,
            alpha=self.alpha,
            depth=self.depth,
            max_path_length=self.max_path_length,
            verbose=False,
            show_progress=False,
            background_knowledge=self.background_knowledge,
            node_names=self.node_names
        )

        if self.causes_only:
            adjacency_matrix = self.filter_causes_only(
                adj_matrix=graph.graph)
            self.dag = utils.graph_from_adjacency(
                adjacency_matrix, node_labels=list(X.columns))
        else:
            self.dag = utils.graph_from_adjacency(
                graph.graph, node_labels=list(X.columns))

        self.metrics = evaluate_graph(ref_graph, self.dag)

        return self

    def filter_causes_only(self, adj_matrix):
        """
        Filter the adjacency matrix to include only causal relationships.

        This method processes the input adjacency matrix and creates a new 
        matrix that only includes direct causal relationships 
        (i.e., where i causes j).

        Parameters:
        -----------
        adj_matrix : numpy.ndarray
            The input adjacency matrix to be filtered.

        Returns:
        --------
        numpy.ndarray
            A new adjacency matrix with only causal relationships preserved.
        """
        result_matrix = np.zeros_like(adj_matrix)

        # Iterate over each pair (i, j) in the matrix
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                # If G[i, j] = -1 and G[j, i] = 1, set result_matrix[i, j] to 1
                if (adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1) \
                        or (adj_matrix[i, j] == 2 and adj_matrix[j, i] == 1):
                    result_matrix[i, j] = 1

        return result_matrix


def main(dataset_name,
         input_path="/Users/renero/phd/data/sachs/",
         output_path="/Users/renero/phd/output/RC4/sachs/compared/",
         save=False,
         **kwargs):
    """
    Create a call to FCI with a sample dataset.
    """
    data = pd.read_csv(f"{input_path}{dataset_name}.csv")
    ref_graph = utils.graph_from_dot_file(f"{input_path}{dataset_name}.dot")

    fci = FCI(name=dataset_name, **kwargs)
    fci.fit_predict(X=data, ref_graph=ref_graph)

    if fci.dag:
        for edge in fci.dag.edges():
            print(edge)
    else:
        for edge in fci.pag.edges():
            print(edge)
    if fci.metrics:
        print(fci.metrics)
    else:
        print("No metrics available")

    # if save:
    #     where_to = utils.save_experiment(rex.name, output_path, rex)
    #     print(f"Saved '{rex.name}' to '{where_to}'")


# Create a call to FCI with a sample dataset.
if __name__ == "__main__":
    main("sachs", alpha=0.75, depth=9)
    # main("rex_generated_linear_1", njobs=1)
