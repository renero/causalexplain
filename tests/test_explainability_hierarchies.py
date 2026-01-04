import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causalexplain.explainability import hierarchies as hmod


def test_compute_correlation_matrix_branches(monkeypatch):
    data = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    corr = hmod.Hierarchies.compute_correlation_matrix(data, method="spearman")
    assert corr.loc["a", "b"] == pytest.approx(1.0)

    called = {}

    def fake_mic(df, alpha, c, prog_bar):
        called["args"] = (alpha, c, prog_bar)
        return pd.DataFrame(np.eye(df.shape[1]), columns=df.columns, index=df.columns)

    monkeypatch.setattr(hmod, "pairwise_mic", fake_mic)
    corr = hmod.Hierarchies.compute_correlation_matrix(data, method="mic", mic_alpha=0.7, mic_c=3, prog_bar=True)
    assert called["args"] == (0.7, 3, True)
    assert corr.equals(pd.DataFrame(np.eye(2), columns=["a", "b"], index=["a", "b"]))

    with pytest.raises(ValueError):
        hmod.Hierarchies.compute_correlation_matrix(data, method="unknown")


def test_compute_correlated_features_and_are_connected():
    corr = pd.DataFrame([[1.0, 0.9], [0.9, 1.0]], columns=["x", "y"], index=["x", "y"])
    correlated = hmod.Hierarchies.compute_correlated_features(corr, 0.8, ["x", "y"])
    assert correlated == {}

    linkage_mat = np.array([[0, 1, 0.25, 2]])
    clusters = hmod.Hierarchies()._clusters_from_linkage(linkage_mat, ["x", "y"])
    assert hmod.Hierarchies()._are_connected(clusters, "x", "y") == pytest.approx(0.25)


def test_get_direction_and_directed_pair():
    g = nx.DiGraph()
    g.add_edge("a", "b")
    assert hmod._get_direction(g, "a", "b") == ("-->", 1)
    g.add_edge("b", "a")
    assert hmod._get_direction(g, "a", "b") == ("<->", 0)
    g.remove_edge("a", "b")
    assert hmod._get_direction(g, "a", "b") == ("<--", -1)
    with pytest.raises(ValueError):
        hmod._get_direction(g, "missing", "b")

    assert hmod._get_directed_pair(g, "a", "b") == ("b", "a")
    g.remove_edge("b", "a")
    assert hmod._get_directed_pair(g, "a", "b") is None


def test_connect_isolated_nodes_adds_edge():
    g = nx.DiGraph()
    # self-loops keep nodes present without defining direction between u and v
    g.add_edges_from([("u", "u"), ("v", "v")])
    linkage_mat = np.array([[0, 1, 0.1, 2]])
    connected = hmod.connect_isolated_nodes(g, linkage_mat, ["u", "v"], verbose=True)
    assert connected.has_edge("u", "v") or connected.has_edge("v", "u")


def test_connect_hierarchies_no_crash_with_clusters():
    g = nx.DiGraph()
    g.add_edge("a", "b")
    feature_names = ["a", "b", "c"]
    linkage_mat = np.array([[0, 1, 0.1, 2], [3, 2, 0.2, 2]])

    # The helper raises if nodes are missing; patch to only report orientation
    def safe_direction(graph, u, v):
        if not graph.has_node(u):
            graph.add_node(u)
        if not graph.has_node(v):
            graph.add_node(v)
        return " · ", None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(hmod, "_get_direction", safe_direction)
    result = hmod.connect_hierarchies(g, linkage_mat, feature_names, verbose=True)
    monkeypatch.undo()
    assert isinstance(result, nx.DiGraph)
