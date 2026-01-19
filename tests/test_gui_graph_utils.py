import networkx as nx

from causalexplain.gui.graph_utils import (
    clean_node_name,
    dag_is_valid,
    graph_from_dot,
    normalize_graph,
)


def test_clean_node_name_strips_quotes() -> None:
    assert clean_node_name(' "A" ') == "A"


def test_normalize_graph_cleans_nodes(tmp_path) -> None:
    graph = nx.DiGraph()
    graph.add_node('"A"')
    graph.add_node("B")
    graph.add_node("\\n")
    graph.add_edge('"A"', "B")
    normalized = normalize_graph(graph)

    assert "A" in normalized.nodes
    assert "B" in normalized.nodes
    assert "\\n" not in normalized.nodes
    assert ("A", "B") in normalized.edges


def test_dag_is_valid_handles_constraints() -> None:
    graph = nx.DiGraph()
    graph.add_edge("A", "B")

    assert dag_is_valid(graph, 0, 2)
    assert not dag_is_valid(None, 0, 2)

    graph.add_edge("B", "A")
    assert not dag_is_valid(graph, 0, 2)


def test_graph_from_dot_normalizes(tmp_path) -> None:
    dot_file = tmp_path / "graph.dot"
    dot_file.write_text('digraph { "A" -> "B"; }')

    graph = graph_from_dot(str(dot_file))

    assert graph is not None
    assert "A" in graph.nodes
    assert "B" in graph.nodes
    assert ("A", "B") in graph.edges
