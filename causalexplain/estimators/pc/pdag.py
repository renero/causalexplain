import networkx as nx

from .dag import DAG
from warnings import warn


class PDAG(nx.DiGraph):
    """Partially directed acyclic graph (PDAG) representation."""

    def __init__(self, directed_ebunch=[], undirected_ebunch=[], latents=[]):
        """Initialize a PDAG from directed and undirected edge lists."""
        super(PDAG, self).__init__(
            directed_ebunch
            + undirected_ebunch
            + [(Y, X) for (X, Y) in undirected_ebunch]
        )
        self.latents = set(latents)
        self.directed_edges = set(directed_ebunch)
        self.undirected_edges = set(undirected_ebunch)
        # TODO: Fix the cycle issue
        # import pdb; pdb.set_trace()
        # try:
        #     # Filter out undirected edges as they also form a cycle in
        #     # themself when represented using directed edges.
        #     cycles = filter(lambda t: len(t) > 2, nx.simple_cycles(self))
        #     if cycles:
        #         out_str = "Cycles are not allowed in a PDAG. "
        #         out_str += "The following path forms a loop: "
        #         out_str += "".join(["({u},{v}) ".format(u=u, v=v) for (u, v) in cycles])
        #         raise ValueError(out_str)
        # except nx.NetworkXNoCycle:
        #     pass

    def copy(self):
        """
        Returns a copy of the object instance.

        Returns
        -------
        PDAG instance: Returns a copy of self.
        """
        return PDAG(
            directed_ebunch=list(self.directed_edges.copy()),
            undirected_ebunch=list(self.undirected_edges.copy()),
            latents=self.latents,
        )

    def to_dag(self, required_edges=[]):
        """
        Returns one possible DAG which is represented using the PDAG.

        Parameters
        ----------
        required_edges: list, array-like of 2-tuples
            The list of edges that should be included in the DAG.

        Returns
        -------
        Returns an instance of DAG.

        Examples
        --------

        """
        # Add required edges if it doesn't form a new v-structure or an opposite edge
        # is already present in the network.
        dag = DAG()
        # Add all the nodes and the directed edges
        dag.add_nodes_from(self.nodes())
        dag.add_edges_from(self.directed_edges)
        dag.latents = self.latents

        pdag = self.copy()
        while pdag.number_of_nodes() > 0:
            # find node with (1) no directed outgoing edges and
            #                (2) the set of undirected neighbors is either empty or
            #                    undirected neighbors + parents of X are a clique
            found = False
            for X in pdag.nodes():
                directed_outgoing_edges = set(pdag.successors(X)) - set(
                    pdag.predecessors(X)
                )
                undirected_neighbors = set(pdag.successors(X)) & set(
                    pdag.predecessors(X)
                )
                neighbors_are_clique = all(
                    (
                        pdag.has_edge(Y, Z)
                        for Z in pdag.predecessors(X)
                        for Y in undirected_neighbors
                        if not Y == Z
                    )
                )

                if not directed_outgoing_edges and (
                    not undirected_neighbors or neighbors_are_clique
                ):
                    found = True
                    # add all edges of X as outgoing edges to dag
                    for Y in pdag.predecessors(X):
                        dag.add_edge(Y, X)
                    pdag.remove_node(X)
                    break

            if not found:
                warn(
                    "PDAG has no faithful extension (= no oriented DAG with the "
                    + "same v-structures as PDAG). Remaining undirected PDAG edges "
                    + "oriented arbitrarily."
                )
                for X, Y in pdag.edges():
                    if not dag.has_edge(Y, X):
                        try:
                            dag.add_edge(X, Y)
                        except ValueError:
                            pass
                break
        return dag
