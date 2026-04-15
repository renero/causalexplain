import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causalexplain.common import utils


def test_valid_output_name_handles_existing_files(tmp_path):
    existing = tmp_path / "exp.pickle"
    existing.touch()
    candidate = utils.valid_output_name("exp", str(tmp_path), extension="pickle")
    assert candidate.endswith("_1.pickle")
    # Create the suggested name so the next call increments again
    Path(candidate).touch()
    candidate = utils.valid_output_name("exp", str(tmp_path), extension=".pickle")
    assert candidate.endswith("_2.pickle")


def test_save_and_load_experiment_roundtrip(tmp_path):
    obj = {"a": 1}
    path = utils.save_experiment("run", str(tmp_path), obj)
    assert path.endswith(".pickle")
    loaded = utils.load_experiment("run", str(tmp_path))
    assert loaded == obj


def test_graph_from_dictionary_with_weights():
    g = utils.graph_from_dictionary({"u": [("v", 0.3)], "x": ["y"]})
    assert g.has_edge("v", "u") and g["v"]["u"]["weight"] == 0.3
    assert g.has_edge("y", "x")


def test_graph_from_adjacency_threshold_and_inverse():
    adj = np.array([[0, 0.5], [0, 0]])
    g = utils.graph_from_adjacency(adj, node_labels=["a", "b"], th=0.1)
    assert g.has_edge("a", "b")
    g_inv = utils.graph_from_adjacency(adj, inverse=True, absolute_values=True, th=0.1)
    assert g_inv.has_edge(1, 0)


def test_graph_from_adjacency_file_and_to_file(tmp_path):
    mat = np.array([[0, 1], [0, 0]])
    df = pd.DataFrame(mat, columns=["a", "b"])
    adj_file = tmp_path / "adj.csv"
    df.to_csv(adj_file, index=False)
    g, loaded_df = utils.graph_from_adjacency_file(adj_file, th=0.0)
    assert np.array_equal(loaded_df.values, mat)
    assert g.has_edge("a", "b")
    out_file = tmp_path / "out.csv"
    utils.graph_to_adjacency_file(g, out_file, labels=["a", "b"])
    assert out_file.exists()


def test_graph_to_adjacency_symbol_and_weight():
    g = nx.DiGraph()
    g.add_edge("a", "b", **{"b": ">"})
    g.add_edge("b", "a", weight=2)
    mat = utils.graph_to_adjacency(g, labels=["a", "b"])
    assert mat[0, 1] == 2  # symbol ">" maps to 2
    assert mat[1, 0] == 2


def test_select_device_invalid_force():
    with pytest.raises(ValueError):
        utils.select_device(force="baddevice")
    assert utils.select_device(force="cpu") == "cpu"


def test_graph_operations_union_and_intersection():
    g1 = nx.DiGraph()
    g2 = nx.DiGraph()
    g1.add_edge("a", "b")
    g2.add_edge("a", "b")
    nx.set_node_attributes(g1, {"a": {"score": 1}, "b": {"score": 2}})
    nx.set_node_attributes(g2, {"a": {"score": 3}, "b": {"score": 1}})
    inter = utils.graph_intersection(g1, g2)
    assert inter.has_edge("a", "b") and inter.nodes["a"]["score"] == 2
    union = utils.graph_union(g1, g2)
    assert union.has_edge("a", "b")


def test_prior_helpers_and_cycles():
    feature_names = ["x", "y", "z"]
    prior = [["x"], ["y"], ["z"]]
    assert utils.valid_candidates_from_prior(feature_names, "y", prior) == ["x"]
    assert utils.valid_candidates_from_prior(
        feature_names, "missing", prior
    ) == feature_names

    dag = nx.DiGraph([("z", "y"), ("y", "x"), ("x", "z")])
    broken = utils.break_cycles_using_prior(dag, prior, verbose=False)
    assert not list(nx.simple_cycles(broken))


def test_potential_misoriented_edges_and_break_cycles_if_present():
    discrepancies = {
        "a": {"b": SimpleNamespace(shap_gof=0.9)},
        "b": {"a": SimpleNamespace(shap_gof=0.1)},
    }
    loop = ["a", "b"]
    candidates = utils.potential_misoriented_edges(loop, discrepancies)
    assert candidates == [("a", "b", pytest.approx(0.8))]

    dag = nx.DiGraph([("a", "b"), ("b", "a")])
    fixed = utils.break_cycles_if_present(dag, discrepancies)
    assert not list(nx.simple_cycles(fixed))


def test_feature_helpers_and_casting():
    arr = np.array([[0, 1], [1, 2]])
    names = utils.get_feature_names(arr)
    assert names == ["X0", "X1"]
    types = utils.get_feature_types(arr)
    assert types["X0"] == "binary"

    df = pd.DataFrame({"cat": ["a", "b", "a"], "num": [1.0, 2.0, 3.0]})
    casted = utils.cast_categoricals_to_int(df)
    assert casted["cat"].dtype.kind in "iu"


@pytest.mark.parametrize(
    "f1,f2,x_values,expected",
    [
        ([0.1, 0.9], [0.9, 0.1], [0.0, 1.0], 0.5),
        ([0.1, 0.2], [0.3, 0.4], [0.0, 1.0], None),
    ],
)
def test_find_crossing_point(f1, f2, x_values, expected):
    assert utils.find_crossing_point(f1, f2, x_values) == expected


@pytest.mark.parametrize(
    "seconds,expected_unit",
    [(1.0, "s"), (120.0, "m"), (4000.0, "h"), (90000.0, "d"), (700000.0, "w"), (3_000_000.0, "m"), (40_000_000.0, "y"), (400_000_000.0, "a")],
)
def test_format_time_branches(seconds, expected_unit):
    t, unit = utils.format_time(seconds)
    assert unit == expected_unit and t > 0


def test_format_time_negative_raises():
    with pytest.raises(ValueError):
        utils.format_time(-1.0)


def test_stringfy_object_lists_relevant_attributes():
    obj = SimpleNamespace(alpha=1, beta=[1, 2], gamma=np.zeros((1, 1)))
    summary = utils.stringfy_object(obj)
    assert "alpha" in summary and "beta" in summary


def test_combine_dags_and_correct_edges_from_prior():
    dag1 = nx.DiGraph([("a", "b")])
    dag2 = nx.DiGraph([("b", "c")])
    for g in (dag1, dag2):
        g.add_node("a", score=1)
        g.add_node("b", score=1)
        g.add_node("c", score=1)
    discrepancies = {
        "a": {"b": SimpleNamespace(shap_gof=0.2)},
        "b": {"a": SimpleNamespace(shap_gof=0.2), "c": SimpleNamespace(shap_gof=0.3)},
        "c": {"b": SimpleNamespace(shap_gof=0.3)},
    }
    prior = [["a"], ["b"], ["c"]]
    union, inter, u_clean, _ = utils.combine_dags(dag1, dag2, discrepancies, prior=prior)
    assert union.has_edge("a", "b") and union.has_edge("b", "c")
    assert not list(nx.simple_cycles(u_clean))
    assert inter.number_of_edges() == 0


def test_list_files_and_json_reading(tmp_path, monkeypatch):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "b.txt").write_text("{}", encoding="utf-8")
    files = utils.list_files("*.json", str(tmp_path))
    assert files == ["a"]
    prior = utils.read_json_file(str(tmp_path / "a.json"))
    assert prior == []
    # Invalid JSON returns empty list
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert utils.read_json_file(str(tmp_path / "bad.json")) == []
    # Missing file
    assert utils.read_json_file(str(tmp_path / "missing.json")) == []
