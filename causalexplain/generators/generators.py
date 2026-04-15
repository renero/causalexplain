"""Acyclic Graph Generator.

Generates a dataset out of an acyclic FCM.
Author : Olivier Goudet and Diviyan Kalainathan

Minor update: J. Renero (due to pandas changes to as_matrix()), and bug fix in
polynomial mechanism.
"""

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import scale

from causalexplain.generators.mechanisms import *
from causalexplain.common.plot import dag2dot


class AcyclicGraphGenerator(object):
    """Generates a cross-sectional dataset out of a cyclic FCM."""

    def __init__(self, causal_mechanism,
                 initial_variable_generator=gmm_cause,
                 points=500, nodes=20, timesteps=0, parents_max=5, verbose=False):
        """Initialize a synthetic acyclic graph generator."""
        super(AcyclicGraphGenerator, self).__init__()
        self.mechanism = {'linear': LinearMechanism,
                          'polynomial': Polynomial_Mechanism,
                          'sigmoid_add': SigmoidAM_Mechanism,
                          'sigmoid_mix': SigmoidMix_Mechanism,
                          'gp_add': GaussianProcessAdd_Mechanism,
                          'gp_mix': GaussianProcessMix_Mechanism}[causal_mechanism]
        self.nodes = int(nodes)
        self.data = pd.DataFrame(
            None, columns=[f"V{i}" for i in range(self.nodes)])
        if timesteps == 0:
            self.timesteps = np.inf
        else:
            self.timesteps = timesteps
        self.points = int(points)
        self.adjacency_matrix = np.zeros((self.nodes, self.nodes), dtype=int)
        # "max parents" is an upper bound, so clamp impossible requests.
        self.parents_max = max(0, min(int(parents_max), max(0, self.nodes - 1)))
        self.initial_generator = initial_variable_generator
        self.cfunctions = None
        self.g = None
        self.verbose = verbose

    def init_variables(self):
        """Redefine the causes of the graph."""
        self.adjacency_matrix = np.zeros((self.nodes, self.nodes), dtype=int)
        self.data = pd.DataFrame(None, columns=[f"V{i}" for i in range(self.nodes)])

        for i in range(self.nodes - 1):
            successors = list(range(i + 1, self.nodes))
            max_children = min(self.parents_max, len(successors))
            if max_children <= 0:
                continue
            num_children = np.random.randint(0, max_children + 1)
            if num_children == 0:
                continue
            for j in np.random.choice(successors, num_children, replace=False):
                self.adjacency_matrix[i, j] = 1

        self.g = nx.DiGraph(self.adjacency_matrix)

        # Mechanisms
        if self.verbose:
            print("Generating mechanisms...")
        self.cfunctions = [
            self.mechanism(int(sum(self.adjacency_matrix[:, i])), self.points,
                           verbose=self.verbose)
            if sum(self.adjacency_matrix[:, i]) else self.initial_generator
            for i in range(self.nodes)
        ]

    def generate(self, rescale=True) -> (nx.DiGraph, pd.DataFrame):
        """Generate data from an FCM containing cycles."""
        if self.cfunctions is None:
            if self.verbose:
                print("CFunctions empty, initializing...")
            self.init_variables()

        for i in nx.topological_sort(self.g):
            if self.verbose:
                print(f"Generating V{i}")
            # Root cause
            if not sum(self.adjacency_matrix[:, i]):
                if self.verbose:
                    print(f"  V{i} is a root cause")
                self.data[f'V{i}'] = self.cfunctions[i](
                    self.points, verbose=self.verbose)
            # Generating causes
            else:
                if self.verbose:
                    print(
                        f"  V{i} parents: {self.adjacency_matrix[:, i].nonzero()[0]}")
                column = self.adjacency_matrix[:, i].nonzero()[0]
                self.data[f'V{i}'] = self.cfunctions[i](
                    np.stack(self.data.iloc[:, column].values),
                    verbose=self.verbose)
            if rescale:
                self.data[f'V{i}'] = scale(np.stack(self.data[f'V{i}'].values))

        return self.g, self.data

    def to_csv(self, fname_radical, **kwargs):
        """
        Save data to the csv format by default, in two separate files.

        Optional keyword arguments can be passed to pandas.
        """
        if self.data is None:
            raise ValueError("Graph has not yet been generated. \
                              Use self.generate() to do so.")

        self.data.to_csv(fname_radical+'_data.csv', **kwargs)
        pd.DataFrame(self.adjacency_matrix).to_csv(
            fname_radical+'_target.csv', **kwargs)
        # Save also the DOT format
        graph_dot_format = dag2dot(self.g).to_string()
        graph_dot_format = f"strict {graph_dot_format[:-9]}\n}}"
        with open(fname_radical+'_target.dot', "w") as f:
            f.write(graph_dot_format)


if __name__ == '__main__':
    save = True

    # Seed
    np.random.seed(1342)

    # Generate data
    g = AcyclicGraphGenerator("polynomial", points=20,
                              nodes=10, timesteps=0, parents_max=3, verbose=False)
    graph, data = g.generate()

    # Cross check experiment
    # import matplotlib.pyplot as plt
    # print("V1")
    # print(data.V1.values)
    # plt.scatter(data.V1, data.V3)
    # plt.xlabel("V1"); plt.ylabel("V3")
    # plt.show()

    if save:
        # Save to file
        data.to_csv(
            "/Users/renero/phd/data/RC3/rex_generated_polynew_1.csv", index=False)
        # write to file
        graph_dot_format = dag2dot(graph).to_string()
        graph_dot_format = f"strict {graph_dot_format[:-9]}\n}}"
        with open("/Users/renero/phd/data/RC3/rex_generated_polynew_1.dot", "w") as f:
            f.write(graph_dot_format)
