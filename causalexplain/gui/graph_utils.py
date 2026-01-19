"""Graph helper utilities for the GUI."""

from __future__ import annotations

from typing import Any, Optional

import networkx as nx


def clean_node_name(name: Any) -> str:
    """Normalize a node name by stripping whitespace and quotes."""
    text = str(name).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text


def normalize_graph(graph: nx.Graph) -> nx.DiGraph:
    """Return a cleaned directed graph with normalized node names."""
    cleaned = nx.DiGraph()
    for node in graph.nodes:
        cleaned.add_node(clean_node_name(node))
    for edge in graph.edges:
        if not isinstance(edge, (tuple, list)) or len(edge) < 2:
            continue
        src, dst = edge[0], edge[1]
        cleaned.add_edge(clean_node_name(src), clean_node_name(dst))
    if cleaned.has_node("\\n"):
        cleaned.remove_node("\\n")
    return cleaned


def dag_is_valid(
    graph: Optional[nx.DiGraph], min_edges: int, max_edges: int
) -> bool:
    """Check that the graph is a DAG within the edge constraints."""
    if graph is None:
        return False
    if not nx.is_directed_acyclic_graph(graph):
        return False
    edge_count = graph.number_of_edges()
    if edge_count < min_edges or edge_count > max_edges:
        return False
    return all(degree > 0 for _, degree in graph.degree())


def graph_from_dot(path: str) -> Optional[nx.DiGraph]:
    """Load a DOT file into a normalized directed graph."""
    if not path:
        return None
    graph = nx.drawing.nx_pydot.read_dot(path)
    return normalize_graph(graph)
