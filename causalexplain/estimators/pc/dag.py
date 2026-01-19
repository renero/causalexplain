import itertools
import networkx as nx
import numpy as np


from .independencies import Independencies

class DAG(nx.DiGraph):
    """
    Base class for all Directed Graphical Models.

    Each node in the graph can represent either a random variable, `Factor`,
    or a cluster of random variables. Edges in the graph represent the
    dependencies between these.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is
        created. The data can be an edge list or any Networkx graph object.
    """

    def __init__(self, ebunch=None, latents=set()):
        super(DAG, self).__init__(ebunch)
        self.latents = set(latents)
        cycles = []
        try:
            cycles = list(nx.find_cycle(self))
        except nx.NetworkXNoCycle:
            pass
        else:
            out_str = "Cycles are not allowed in a DAG."
            out_str += "\nEdges indicating the path taken for a loop: "
            out_str += "".join([f"({u},{v}) " for (u, v) in cycles])
            raise ValueError(out_str)

    def add_node(self, node, weight=None, latent=False):
        """
        Adds a single node to the Graph.

        Parameters
        ----------
        node: str, int, or any hashable python object.
            The node to add to the graph.

        weight: int, float
            The weight of the node.

        latent: boolean (default: False)
            Specifies whether the variable is latent or not.
        """

        # Check for networkx 2.0 syntax
        if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], dict):
            node, attrs = node
            if attrs.get("weight", None) is not None:
                attrs["weight"] = weight
        else:
            attrs = {"weight": weight}

        if latent:
            self.latents.add(node)

        super(DAG, self).add_node(node, weight=weight)

    def add_nodes_from(self, nodes, weights=None, latent=False):
        """Add multiple nodes to the graph."""
        nodes = list(nodes)

        if isinstance(latent, bool):
            latent = [latent] * len(nodes)

        if weights:
            if len(nodes) != len(weights):
                raise ValueError(
                    "The number of elements in nodes and weights" "should be equal."
                )
            for index in range(len(nodes)):
                self.add_node(
                    node=nodes[index], weight=weights[index], latent=latent[index]
                )
        else:
            for index in range(len(nodes)):
                self.add_node(node=nodes[index], latent=latent[index])

    def add_edge(self, u, v, weight=None):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u, v : nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge

        """
        super(DAG, self).add_edge(u, v, weight=weight)

    def add_edges_from(self, ebunch, weights=None):
        """Add multiple edges to the graph."""
        ebunch = list(ebunch)

        if weights:
            if len(ebunch) != len(weights):
                raise ValueError(
                    "The number of elements in ebunch and weights" "should be equal"
                )
            for index in range(len(ebunch)):
                self.add_edge(ebunch[index][0], ebunch[index]
                              [1], weight=weights[index])
        else:
            for edge in ebunch:
                self.add_edge(edge[0], edge[1])

    def get_parents(self, node):
        """
        Returns a list of parents of node.

        Throws an error if the node is not present in the graph.

        Parameters
        ----------
        node: string, int or any hashable python object.
            The node whose parents would be returned.

        """
        return list(self.predecessors(node))

    def moralize(self):
        """
        Removes all the immoralities in the DAG and creates a moral
        graph (UndirectedGraph).

        A v-structure X->Z<-Y is an immorality if there is no directed edge
        between X and Y.

        """
        moral_graph = UndirectedGraph()
        moral_graph.add_nodes_from(self.nodes())
        moral_graph.add_edges_from(self.to_undirected().edges())

        for node in self.nodes():
            moral_graph.add_edges_from(
                itertools.combinations(self.get_parents(node), 2)
            )

        return moral_graph

    def get_leaves(self):
        """
        Returns a list of leaves of the graph.

        """
        return [node for node, out_degree in self.out_degree_iter() if out_degree == 0]

    def out_degree_iter(self, nbunch=None, weight=None):
        if nx.__version__.startswith("1"):
            return super(DAG, self).out_degree_iter(nbunch, weight)
        else:
            return iter(self.out_degree(nbunch, weight))

    def in_degree_iter(self, nbunch=None, weight=None):
        if nx.__version__.startswith("1"):
            return super(DAG, self).in_degree_iter(nbunch, weight)
        else:
            return iter(self.in_degree(nbunch, weight))

    def get_roots(self):
        """
        Returns a list of roots of the graph.
        """
        return [
            node for node, in_degree in dict(self.in_degree()).items() if in_degree == 0
        ]

    def get_children(self, node):
        """
        Returns a list of children of node.
        Throws an error if the node is not present in the graph.

        Parameters
        ----------
        node: string, int or any hashable python object.
            The node whose children would be returned.
        """
        return list(self.successors(node))

    def get_independencies(self, latex=False, include_latents=False):
        """
        Computes independencies in the DAG, by checking d-seperation.

        Parameters
        ----------
        latex: boolean
            If latex=True then latex string of the independence assertion
            would be created.

        include_latents: boolean
            If True, includes latent variables in the independencies. Otherwise,
            only generates independencies on observed variables.
        """
        nodes = set(self.nodes())
        if not include_latents:
            nodes = set(self.nodes()) - self.latents

        independencies = Independencies()
        for start in nodes:
            if not include_latents:
                rest = set(self.nodes()) - {start} - self.latents
            else:
                rest = set(self.nodes()) - {start}

            for r in range(len(rest)):
                for observed in itertools.combinations(rest, r):
                    d_seperated_variables = (
                        rest
                        - set(observed)
                        - set(
                            self.active_trail_nodes(
                                start,
                                observed=observed,
                                include_latents=include_latents,
                            )[start]
                        )
                    )
                    if d_seperated_variables:
                        independencies.add_assertions(
                            [start, d_seperated_variables, observed]
                        )
        independencies.reduce()

        if not latex:
            return independencies
        else:
            return independencies.latex_string()

    def local_independencies(self, variables):
        """
        Returns an instance of Independencies containing the local independencies
        of each of the variables.

        Parameters
        ----------
        variables: str or array like
            variables whose local independencies are to be found.

        """

        independencies = Independencies()
        for variable in (
            variables if isinstance(variables, (list, tuple)) else [variables]
        ):
            non_descendents = (
                set(self.nodes())
                - {variable}
                - set(nx.dfs_preorder_nodes(self, variable))
            )
            parents = set(self.get_parents(variable))
            if non_descendents - parents:
                independencies.add_assertions(
                    [variable, non_descendents - parents, parents]
                )
        return independencies

    def is_iequivalent(self, model):
        """
        Checks whether the given model is I-equivalent

        Two graphs G1 and G2 are said to be I-equivalent if they have same skeleton
        and have same set of immoralities.

        Parameters
        ----------
        model : A DAG object, for which you want to check I-equivalence

        Returns
        --------
        boolean : True if both are I-equivalent, False otherwise
        """
        if not isinstance(model, DAG):
            raise TypeError(
                f"Model must be an instance of DAG. Got type: {type(model)}"
            )

        if (self.to_undirected().edges() == model.to_undirected().edges()) and (
            self.get_immoralities() == model.get_immoralities()
        ):
            return True
        return False

    def get_immoralities(self):
        """
        Finds all the immoralities in the model
        A v-structure X -> Z <- Y is an immorality if there is no direct edge between X and Y .

        Returns
        -------
        set: A set of all the immoralities in the model

        """
        immoralities = set()
        for node in self.nodes():
            for parents in itertools.combinations(self.predecessors(node), 2):
                if not self.has_edge(parents[0], parents[1]) and not self.has_edge(
                    parents[1], parents[0]
                ):
                    immoralities.add(tuple(sorted(parents)))
        return immoralities

    def is_dconnected(self, start, end, observed=None):
        """
        Returns True if there is an active trail (i.e. d-connection) between
        `start` and `end` node given that `observed` is observed.

        Parameters
        ----------
        start, end : int, str, any hashable python object.
            The nodes in the DAG between which to check the d-connection/active trail.

        observed : list, array-like (optional)
            If given the active trail would be computed assuming these nodes to
            be observed.

        """
        if end in self.active_trail_nodes(start, observed)[start]:
            return True
        else:
            return False

    def minimal_dseparator(self, start, end):
        """
        Finds the minimal d-separating set for `start` and `end`.

        Parameters
        ----------
        start: node
            The first node.

        end: node
            The second node.

        References
        ----------
        [1] Algorithm 4, Page 10: Tian, Jin, Azaria Paz, and Judea Pearl. Finding
        minimal d-separators. Computer Science Department, University of California,
        1998.
        """
        if (end in self.neighbors(start)) or (start in self.neighbors(end)):
            raise ValueError(
                "No possible separators because start and end are adjacent"
            )
        an_graph = self.get_ancestral_graph([start, end])
        separator = set(
            itertools.chain(self.predecessors(start), self.predecessors(end))
        )
        # If any of the parents were latents, take the latent's parent
        while len(separator.intersection(self.latents)) != 0:
            separator_copy = separator.copy()
            for u in separator:
                if u in self.latents:
                    separator_copy.remove(u)
                    separator_copy.update(set(self.predecessors(u)))
            separator = separator_copy
        # Remove the start and end nodes in case it reaches there while removing latents.
        separator.difference_update({start, end})

        # If the initial set is not able to d-separate, no d-separator is possible.
        if an_graph.is_dconnected(start, end, observed=separator):
            return None

        # Go through the separator set, remove one element and check if it remains
        # a dseparating set.
        minimal_separator = separator.copy()

        for u in separator:
            if not an_graph.is_dconnected(start, end, observed=minimal_separator - {u}):
                minimal_separator.remove(u)

        return minimal_separator

    def get_markov_blanket(self, node):
        """
        Returns a markov blanket for a random variable. In the case
        of Bayesian Networks, the markov blanket is the set of
        node's parents, its children and its children's other parents.

        Returns
        -------
        list(blanket_nodes): List of nodes contained in Markov Blanket

        Parameters
        ----------
        node: string, int or any hashable python object.
              The node whose markov blanket would be returned.

        """
        children = self.get_children(node)
        parents = self.get_parents(node)
        blanket_nodes = children + parents
        for child_node in children:
            blanket_nodes.extend(self.get_parents(child_node))
        blanket_nodes = set(blanket_nodes)
        blanket_nodes.discard(node)
        return list(blanket_nodes)

    def active_trail_nodes(self, variables, observed=None, include_latents=False):
        """
        Returns a dictionary with the given variables as keys and all the nodes reachable
        from that respective variable as values.

        Parameters
        ----------
        variables: str or array like
            variables whose active trails are to be found.

        observed : List of nodes (optional)
            If given the active trails would be computed assuming these nodes to be
            observed.

        include_latents: boolean (default: False)
            Whether to include the latent variables in the returned active trail nodes.

        References
        ----------
        Details of the algorithm can be found in 'Probabilistic Graphical Model
        Principles and Techniques' - Koller and Friedman
        Page 75 Algorithm 3.1
        """
        if observed:
            if isinstance(observed, set):
                observed = list(observed)

            observed_list = (
                observed if isinstance(observed, (list, tuple)) else [observed]
            )
        else:
            observed_list = []
        ancestors_list = self._get_ancestors_of(observed_list)

        # Direction of flow of information
        # up ->  from parent to child
        # down -> from child to parent

        active_trails = {}
        for start in variables if isinstance(variables, (list, tuple)) else [variables]:
            visit_list = set()
            visit_list.add((start, "up"))
            traversed_list = set()
            active_nodes = set()
            while visit_list:
                node, direction = visit_list.pop()
                if (node, direction) not in traversed_list:
                    if node not in observed_list:
                        active_nodes.add(node)
                    traversed_list.add((node, direction))
                    if direction == "up" and node not in observed_list:
                        for parent in self.predecessors(node):
                            visit_list.add((parent, "up"))
                        for child in self.successors(node):
                            visit_list.add((child, "down"))
                    elif direction == "down":
                        if node not in observed_list:
                            for child in self.successors(node):
                                visit_list.add((child, "down"))
                        if node in ancestors_list:
                            for parent in self.predecessors(node):
                                visit_list.add((parent, "up"))
            if include_latents:
                active_trails[start] = active_nodes
            else:
                active_trails[start] = active_nodes - self.latents

        return active_trails

    def _get_ancestors_of(self, nodes):
        """
        Returns a dictionary of all ancestors of all the observed nodes including the
        node itself.

        Parameters
        ----------
        nodes: string, list-type
            name of all the observed nodes

        """
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]

        for node in nodes:
            if node not in self.nodes():
                raise ValueError(f"Node {node} not in not in graph")

        ancestors_list = set()
        nodes_list = set(nodes)
        while nodes_list:
            node = nodes_list.pop()
            if node not in ancestors_list:
                nodes_list.update(self.predecessors(node))
            ancestors_list.add(node)
        return ancestors_list

    def to_pdag(self):
        """
        Returns the PDAG (the equivalence class of DAG; also known as CPDAG) of the DAG.

        Returns
        -------
        PDAG: An instance of pgmpy.base.PDAG.

        Examples
        --------

        """
        pass

    def do(self, nodes, inplace=False):
        """
        Applies the do operator to the graph and returns a new DAG with the
        transformed graph.

        The do-operator, do(X = x) has the effect of removing all edges from
        the parents of X and setting X to the given value x.

        Parameters
        ----------
        nodes : list, array-like
            The names of the nodes to apply the do-operator for.

        inplace: boolean (default: False)
            If inplace=True, makes the changes to the current object,
            otherwise returns a new instance.

        Returns
        -------
        pgmpy.base.DAG: A new instance of DAG modified by the do-operator

        References
        ----------
        Causality: Models, Reasoning, and Inference, Judea Pearl (2000). p.70.
        """
        dag = self if inplace else self.copy()

        if isinstance(nodes, (str, int)):
            nodes = [nodes]
        else:
            nodes = list(nodes)

        if not set(nodes).issubset(set(self.nodes())):
            raise ValueError(
                f"Nodes not found in the model: {set(nodes) - set(self.nodes)}"
            )

        for node in nodes:
            parents = list(dag.predecessors(node))
            for parent in parents:
                dag.remove_edge(parent, node)
        return dag

    def get_ancestral_graph(self, nodes):
        """
        Returns the ancestral graph of the given `nodes`. The ancestral graph only
        contains the nodes which are ancestors of atleast one of the variables in
        node.

        Parameters
        ----------
        node: iterable
            List of nodes whose ancestral graph needs to be computed.

        Returns
        -------
        pgmpy.base.DAG instance: The ancestral graph.

        """
        return self.subgraph(nodes=self._get_ancestors_of(nodes=nodes))

    def to_daft(
        self,
        node_pos="circular",
        latex=True,
        pgm_params={},
        edge_params={},
        node_params={},
    ):
        """
        Returns a daft (https://docs.daft-pgm.org/en/latest/) object which can be rendered for
        publication quality plots. The returned object's render method can be called to see the plots.

        Parameters
        ----------
        node_pos: str or dict (default: circular)
            If str: Must be one of the following: circular, kamada_kawai, planar, random, shell, sprint,
                spectral, spiral. Please refer: https://networkx.org/documentation/stable//reference/drawing.html#module-networkx.drawing.layout for details on these layouts.

            If dict should be of the form {node: (x coordinate, y coordinate)} describing the x and y coordinate of each
            node.

            If no argument is provided uses circular layout.

        latex: boolean
            Whether to use latex for rendering the node names.

        pgm_params: dict (optional)
            Any additional parameters that need to be passed to `daft.PGM` initializer.
            Should be of the form: {param_name: param_value}

        edge_params: dict (optional)
            Any additional edge parameters that need to be passed to `daft.add_edge` method.
            Should be of the form: {(u1, v1): {param_name: param_value}, (u2, v2): {...} }

        node_params: dict (optional)
            Any additional node parameters that need to be passed to `daft.add_node` method.
            Should be of the form: {node1: {param_name: param_value}, node2: {...} }

        Returns
        -------
        daft.PGM object: A plot of the DAG.

        """
        try:
            from daft import PGM
        except ImportError as e:
            raise ImportError(
                "Package daft required. Please visit: https://docs.daft-pgm.org/en/latest/ for installation instructions."
            )

        if isinstance(node_pos, str):
            supported_layouts = {
                "circular": nx.circular_layout,
                "kamada_kawai": nx.kamada_kawai_layout,
                "planar": nx.planar_layout,
                "random": nx.random_layout,
                "shell": nx.shell_layout,
                "spring": nx.spring_layout,
                "spectral": nx.spectral_layout,
                "spiral": nx.spiral_layout,
            }
            if node_pos not in supported_layouts.keys():
                raise ValueError(
                    "Unknown node_pos argument. Please refer docstring for accepted values"
                )
            else:
                node_pos = supported_layouts[node_pos](self)
        elif isinstance(node_pos, dict):
            for node in self.nodes():
                if node not in node_pos.keys():
                    raise ValueError(f"No position specified for {node}.")
        else:
            raise ValueError(
                "Argument node_pos not valid. Please refer to the docstring."
            )

        daft_pgm = PGM(**pgm_params)
        for node in self.nodes():
            try:
                extra_params = node_params[node]
            except KeyError:
                extra_params = dict()

            if latex:
                daft_pgm.add_node(
                    node,
                    fr"${node}$",
                    node_pos[node][0],
                    node_pos[node][1],
                    observed=True,
                    **extra_params,
                )
            else:
                daft_pgm.add_node(
                    node,
                    f"{node}",
                    node_pos[node][0],
                    node_pos[node][1],
                    observed=True,
                    **extra_params,
                )

        for u, v in self.edges():
            try:
                extra_params = edge_params[(u, v)]
            except KeyError:
                extra_params = dict()
            daft_pgm.add_edge(u, v, **extra_params)

        return daft_pgm

    @staticmethod
    def get_random(n_nodes=5, edge_prob=0.5, latents=False):
        """
        Returns a randomly generated DAG with `n_nodes` number of nodes with
        edge probability being `edge_prob`.

        Parameters
        ----------
        n_nodes: int
            The number of nodes in the randomly generated DAG.

        edge_prob: float
            The probability of edge between any two nodes in the topologically
            sorted DAG.

        latents: bool (default: False)
            If True, includes latent variables in the generated DAG.

        Returns
        -------
        pgmpy.base.DAG instance: The randomly generated DAG.
        """
        # Step 1: Generate a matrix of 0 and 1. Prob of choosing 1 = edge_prob
        adj_mat = np.random.choice(
            [0, 1], size=(n_nodes, n_nodes), p=[1 - edge_prob, edge_prob]
        )

        # Step 2: Use the upper triangular part of the matrix as adjacency.
        nodes = list(range(n_nodes))
        edges = nx.convert_matrix.from_numpy_array(
            np.triu(adj_mat, k=1), create_using=nx.DiGraph
        ).edges()

        dag = DAG(edges)
        dag.add_nodes_from(nodes)
        if latents:
            dag.latents = set(
                np.random.choice(
                    dag.nodes(), np.random.randint(low=0, high=len(dag.nodes()))
                )
            )
        return dag
