#
# Module to compute conditional independencencies in a DAG, and
# sufficient tests.
#
# Author: Jesús Renero
#
import itertools
from typing import List, Optional, Union
import networkx as nx
try:
    from networkx.algorithms.d_separation import d_separated as nx_d_separated
except Exception:  # pragma: no cover - fallback for older NX layouts
    nx_d_separated = None
try:
    from networkx.algorithms.d_separation import is_d_separator as nx_is_d_separator
except Exception:  # pragma: no cover - fallback for older NX layouts
    nx_is_d_separator = None


def _d_separated(graph, x, y, z):
    """Compat wrapper for NetworkX d-separation across versions."""
    if hasattr(nx, "d_separated"):
        return nx.d_separated(graph, x, y, z)
    if nx_d_separated is not None:
        return nx_d_separated(graph, x, y, z)
    if nx_is_d_separator is not None:
        return nx_is_d_separator(graph, x, y, z)
    raise AttributeError("networkx d-separation helpers are not available")


class ConditionalIndependencies:
    """
    A class to store conditional independencies in a graph.

    Attributes
    ----------
    _cache : dict
        A dictionary representing the conditional independencies.

    Methods
    -------
    add(x, y, z)
        Adds a new conditional independence to the cache.
    __str__()
        Returns a string representation of the conditional independencies.
    __repr__()
        Returns a string representation of the conditional independencies.
    """

    def __init__(self):
        self._cache = {}

    def __str__(self):
        s = "Conditional Independencies:\n"
        for (x, y, z) in self._cache:
            if z == None:
                s += f"  {x} ⊥ {y}\n"
            else:
                s += f"  {x} ⊥ {y} | {z}\n"
        return s

    def __repr__(self) -> str:
        """Returns a string representation of the ConditionalIndependencies object."""
        items = []
        for x, y, z in self._cache.items():
            if z is None:
                items.append(f"({x!r}, {y!r}):{[]!r}")
            else:
                items.append(f"({x!r}, {y!r}):{list(z)!r}")
        return "{" + ", ".join(items) + "}"

    def add(self, var1: str, var2: str,
            conditioning_set: Optional[List[str]] = None) -> None:
        """
        Adds a new conditional independence to the cache.

        Parameters
        ----------
        var1 : str
            A node in the graph.
        var2 : str
            A node in the graph.
        conditioning_set : list of str or None
            A set of nodes in the graph.
        """
        conditioning_set = tuple(conditioning_set) if conditioning_set is not None else None
        if not self._cache:
            self._cache[(var1, var2, conditioning_set)] = True
            return
        if (var1, var2, conditioning_set) in self._cache:
            return
        self._cache[(var1, var2, conditioning_set)] = True


class SufficientSets:
    """
    A class to represent the sufficient sets of a conditional independence test.

    Attributes
    ----------
    _cache : list
        A list of tuples representing the sufficient sets.

    Methods
    -------
    add(suff_set)
        Adds a new sufficient set to the cache.

        Parameters
        ----------
        suff_set : list
            A list of tuples representing the new sufficient set to be added.

    __str__()
        Returns a string representation of the sufficient sets.
    """

    def __init__(self):
        self._cache = []

    def add(self, suff_set):
        """
        Adds a new sufficient set to the cache.

        Parameters
        ----------
        suff_set : list
            A list of tuples representing the new sufficient set to be added.
        """

        for element in suff_set:
            # and ((y, x) not in self._cache):
            if (element not in self._cache):
                self._cache.append(element)

    def __str__(self):
        """
        Returns a string representation of the sufficient sets.

        Returns
        -------
        str
            A string representation of the sufficient sets.
        """
        s = "Sufficient sets:\n"
        if self._cache:
            for sufficient_set in self._cache:
                s += f"  {sufficient_set}\n"
        else:
            s = "No sufficient sets found"

        return s

    def __repr__(self):
        s = "["
        last = False
        if self._cache:
            for sufficient_set in self._cache:
                s += f"{sufficient_set}"
                if sufficient_set == self._cache[-1]:
                    last = True
                if not last:
                    s += ", "
        s += "]"
        return s


def get_backdoor_paths(dag: nx.DiGraph, x: str, y: str):
    """
    Returns all backdoor paths between two nodes in a graph. A backdoor path
    is a path that starts with an edge towards 'x' and ends with an edge
    towards 'y'.

    Parameters:
    -----------
    dag: nx.DiGraph
        A directed graph
    x: str
        A node in the graph
    y: str
        A node in the graph

    Returns:
    --------
    paths: list
        A list of paths between x and y
    """
    # Check if x or y are not in the graph
    if x not in dag.nodes() or y not in dag.nodes():
        return []
    
    # If x and y are the same node, return empty list
    if x == y:
        return []
        
    undirected_graph = dag.to_undirected()
    # list all paths between 'x' and 'y'
    paths = (p for p in nx.all_simple_paths(
        undirected_graph, source=x, target=y) if len(p) > 1 and dag.has_edge(p[1], x))
    return list(paths)


def get_paths(graph: nx.DiGraph, x: str, y: str):
    """
    Returns all simple paths between two nodes in a directed graph.

    Parameters
    ----------
        - graph (nx.DiGraph): A directed graph.
        - x (str): The starting node.
        - y (str): The ending node.

    Returns
    -------
        - list: A list of all simple paths between x and y.
    """
    # Check if x or y are not in the graph
    if x not in graph.nodes() or y not in graph.nodes():
        return []
    
    # If x and y are the same node, return empty list
    if x == y:
        return []
        
    return list(nx.all_simple_paths(graph, source=x, target=y))


def find_colliders_in_path(dag: nx.DiGraph, path: List[str]):
    """
    Returns all colliders in a path.

    Parameters:
    -----------
    G: nx.DiGraph
        A directed graph
    path: list
        A path formed by nodes in the graph

    Returns:
    --------
    colliders: set
        A set of colliders in the path
    """
    colliders = []
    for i in range(1, len(path)-1):
        if dag.has_edge(path[i-1], path[i]) and dag.has_edge(path[i+1], path[i]):
            colliders.append(path[i])

    return set(colliders)


def get_sufficient_sets_for_pair(dag, x, y, verbose=False):
    """
    Compute the sufficient sets for a pair of nodes in a graph. A sufficient set
    is a set of nodes that blocks all backdoor paths between x and y.

    Parameters:
    -----------
    G: nx.DiGraph
        A directed graph
    x: str
        A node in the graph
    y: str
        A node in the graph
    verbose: bool
        If True, print additional information

    Returns:
    --------
    sufficient_sets: list
        A list of sufficient sets for the pair of nodes (x, y)
    """
    backdoor_paths = get_backdoor_paths(dag, x, y)
    if verbose:
        if backdoor_paths:
            print(f" Found {len(backdoor_paths)} backdoor paths")
        else:
            print(" No backdoor paths found")
    sufficient_sets = []
    for path in backdoor_paths:
        print(f"  Checking backdoor path: {path}") if verbose else None
        # get all nodes in the path except the first and last
        sufficient_set = path[1:-1]
        # check that no node in sufficient_set is descendant of x
        descendants = nx.descendants(dag, x)
        if any([d in descendants for d in sufficient_set]):
            if verbose:
                print(
                    f"Path {path} discarded because it contains a descendant of x")
            continue

        sufficient_sets.append(sufficient_set)

    # Now I must check that the nodes in the sufficient set block every backdoor path
    # between x and y
    final_suff_set = []
    for sufficient_set in sufficient_sets:
        if verbose:
            print(
                f"     ",
                f"Checking that {sufficient_set} blocks all backdoor paths "
                f"between {x} and {y}")
        # Check if any of the nodes in the sufficient set is in a collider in the path
        colliders = find_colliders_in_path(dag, [x] + sufficient_set + [y])
        # If any of the nodes in the sufficient set is a collider, then continue
        if len(colliders) > 0:
            if verbose:
                print(
                    f"        {sufficient_set} contains a collider: {colliders}")
            continue
        all_conditions = True
        for path in backdoor_paths:
            if verbose:
                print(f"      Checking path {path}")
            # Check that this path can be blocked by any node in the sufficient set
            # The path is blocked if any of the nodes in the sufficient set
            # is in the path
            colliders = find_colliders_in_path(dag, path)
            if verbose:
                if colliders:
                    print(f"        ! Colliders in path: {colliders}")
                else:
                    print("        - No colliders in path")
            # Find what nodes from the sufficient set are in the path
            nodes_in_path = set(sufficient_set).intersection(set(path))
            if verbose:
                if nodes_in_path:
                    print(
                        f"        + Nodes from sufficient set in path: {nodes_in_path}")
                else:
                    print(f"        - No nodes from sufficient set in path")
            if len(nodes_in_path) > 0:
                # Check that at least one of the nodes in the path is NOT a collider
                if nodes_in_path.intersection(colliders) == set():
                    if verbose:
                        print(
                            f"          Path {path} blocked by nodes in "
                            f"{sufficient_set} ")
                elif len(nodes_in_path) == len(colliders):
                    if verbose:
                        print(f"      ALL nodes in {sufficient_set} are colliders "
                              f"in {path} \n"
                              f"      => {nodes_in_path.intersection(colliders)} == "
                              f"{colliders}")
                    all_conditions = False
                else:
                    if verbose:
                        print(f"      Some nodes in {sufficient_set} are NOT colliders "
                              f"in {path} ")
            else:
                if verbose:
                    print(f"        No nodes in {sufficient_set} are in {path}, "
                          f"so they do not block this path ")
                all_conditions = False
        if all_conditions:
            if verbose:
                print(
                    f"          {sufficient_set} blocks all backdoor paths "
                    f"between x and y")
            final_suff_set.append(sufficient_set)

    return final_suff_set


def get_sufficient_sets(dag, verbose=False):
    """
    Get the sufficient sets (admissible sets) for all pairs of nodes in a graph.

    Parameters:
    -----------
    G: nx.DiGraph
        A directed graph
    verbose: bool
        If True, print additional information

    Returns:
    --------
    suff_sets: Suff_Sets
        A list of sufficient sets for all pairs of nodes in the graph
    """
    suff_sets = SufficientSets()
    for x, y in itertools.combinations(dag.nodes(), 2):
        if verbose:
            print(f"Checking pair ({x}, {y})...", end="", sep="")
        sufficient_set = get_sufficient_sets_for_pair(dag, x, y, verbose)
        if sufficient_set:
            print(f"  Adding sufficient set: {sufficient_set}") if verbose else None
            suff_sets.add(sufficient_set)

    return suff_sets

#
# XXX: This is not used, and should be removed or replaced by the NX 
# implementation of d-separation or DoWhy implementation.
#
def get_conditional_independencies(dag, verbose=False):
    """
    Computes the set of conditional independencies implied by the graph G.

    Parameters:
    -----------
    dag : networkx.DiGraph
        The directed acyclic graph representing the causal relationships
        between the variables.
    verbose : bool, optional
        If True, prints additional information about the computation.

    Returns:
    --------
    cond_indeps : Cond_Indep
        The object containing the set of conditional independencies implied
        by the graph G.
    """
    cond_indeps = ConditionalIndependencies()
    # Enumerate all pairs of nodes in G that are not d_separated
    for x, y in itertools.combinations(dag.nodes(), 2):
        # Check if x and y are connected by an edge
        if dag.has_edge(x, y) or dag.has_edge(y, x):
            continue
        if not _d_separated(dag, {x}, {y}, set()):
            if verbose:
                print(f"Pair ({x}, {y})")
            paths = get_paths(dag, x, y)
            # Check if any of the paths contains a collider
            for path in paths:
                if verbose:
                    print("  Path:", path)
                colliders = find_colliders_in_path(dag, path)
                if len(colliders) == 0:
                    blockers = set(path[1:-1])
                    if len(blockers) == 1:
                        cond_indeps.add(x, y, blockers.pop())
                    else:
                        # Check if the set of blockers is exactly the entire graph
                        # without "x" and "y"
                        if blockers != set(dag.nodes()) - {x, y}:
                            cond_indeps.add(x, y, tuple(blockers))
                            if verbose:
                                print(f"    (no colliders on path {path})\n"
                                      f"     {x} ⊥ {y} | {blockers}")
                        else:
                            if verbose:
                                print(f"    The set of blockers is the entire graph. ")
                else:
                    if verbose:
                        print(f"    Colliders in path: {colliders}")
                    for blocker in path[1:-1]:
                        if verbose:
                            print(f"    Blocking on {blocker}")
                        if blocker not in colliders:
                            cond_indeps.add(x, y, blocker)
                            if verbose:
                                print(f"       {x} ⊥ {y} | {blocker}")
                        else:
                            if verbose:
                                print(
                                    f"    Blocking on {blocker} is Collider: "
                                    f"{colliders}")
        else:
            cond_indeps.add(x, y)
            if verbose:
                print(f"Pair ({x}, {y})\n"
                      f"   {x} ⊥ {y} | ∅")

    return cond_indeps


def custom_main():
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ('x1', 'x2'),
            ('x2', 'x3'),
            ('x1', 'x4'),
            ('x2', 'x4')
        ]
    )
    ss = get_sufficient_sets(G, verbose=True)
    print(ss)
    cond_independencies = get_conditional_independencies(G, verbose=True)
    print(cond_independencies)


def main():
    G = nx.DiGraph()
    G.add_edges_from([('z1', 'x'), ('z1', 'z3'), ('z3', 'x'),
                     ('z3', 'y'), ('x', 'y'), ('z2', 'z3'), ('z2', 'y')])

    ss = get_sufficient_sets(G, verbose=True)
    print(ss)

    cond_independencies = get_conditional_independencies(G, verbose=False)
    print(cond_independencies)


def dag_main():
    from causalexplain.estimators.pc.dag import DAG

    G = DAG([('x1', 'x2'), ('x2', 'x3'), ('x1', 'x4'), ('x2', 'x4')])
    ci = G.get_independencies()
    print(ci)


if __name__ == "__main__":
    custom_main()
    # main()
    print("\n-----\n")
    dag_main()
