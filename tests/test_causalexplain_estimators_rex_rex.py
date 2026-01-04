from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causalexplain.estimators.rex import rex as rex_module
from causalexplain.estimators.rex.rex import Rex


@pytest.mark.parametrize(
    "model_type, explainer, expected",
    [("nn", "explainer", "gradient"), ("gbt", "gradient", "tree")],
)
def test_check_model_and_explainer_adjusts(model_type, explainer, expected):
    rex = Rex(name="demo", model_type=model_type, explainer=explainer)
    assert rex.explainer == expected


def test_gbt_explainer_forces_tree():
    rex = Rex(name="demo", model_type="gbt", explainer="explainer")
    assert rex.explainer == "tree"


def test_get_prior_from_ref_graph(monkeypatch):
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    graph = nx.DiGraph([("root", "child")])
    monkeypatch.setattr(rex_module.utils, "graph_from_dot_file", lambda path: graph)

    prior = rex._get_prior_from_ref_graph("/tmp")

    assert prior == [["root"], ["child"]]


def test_get_prior_from_ref_graph_missing(monkeypatch):
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    monkeypatch.setattr(rex_module.utils, "graph_from_dot_file", lambda path: None)

    assert rex._get_prior_from_ref_graph("/tmp") is None


@pytest.mark.internal
def test_filter_adjacency_matrix_filters_values():
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    adjacency = np.array([[0.2, 0.05], [-0.04, 0.11]])

    filtered = rex._filter_adjacency_matrix(adjacency, tolerance=0.1)

    assert filtered.tolist() == [[0.2, 0.0], [0.0, 0.11]]


@pytest.mark.internal
def test_dag_from_bootstrap_adj_matrix_filters_and_breaks(monkeypatch):
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    rex.feature_names = ["A", "B"]
    rex.shaps = SimpleNamespace(shap_discrepancies={})

    captured = {}

    def fake_graph_from_adjacency(adj, names):
        captured["adjacency"] = adj.copy()
        dag = nx.DiGraph()
        for i, src in enumerate(names):
            for j, dst in enumerate(names):
                if adj[i, j] != 0:
                    dag.add_edge(src, dst)
        return dag

    def fake_break_cycles(dag, *_args, **_kwargs):
        captured["broken_graph_edges"] = set(dag.edges())
        return dag

    monkeypatch.setattr(rex_module.utils, "graph_from_adjacency", fake_graph_from_adjacency)
    monkeypatch.setattr(rex_module.utils, "break_cycles_if_present", fake_break_cycles)

    adjacency = np.array([[0.0, 0.2], [0.05, 0.0]])

    dag = rex._dag_from_bootstrap_adj_matrix(adjacency, tolerance=0.1)

    assert captured["adjacency"].tolist() == [[0.0, 0.2], [0.0, 0.0]]
    assert ("A", "B") in dag.edges()
    assert ("B", "A") not in captured["broken_graph_edges"]


@pytest.mark.internal
def test_dag_from_bootstrap_adj_matrix_validations(monkeypatch):
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    rex.feature_names = ["A", "B"]
    rex.shaps = SimpleNamespace(shap_discrepancies={})
    adjacency = [[0.0, 0.2], [0.0, 0.0]]

    with pytest.raises(AssertionError):
        rex._dag_from_bootstrap_adj_matrix(adjacency, tolerance=0.1)
    with pytest.raises(AssertionError):
        rex._dag_from_bootstrap_adj_matrix(np.array(adjacency), tolerance=-0.1)


@pytest.mark.internal
def test_build_bootstrapped_adjacency_matrix_averages(monkeypatch):
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    rex.is_fitted_ = True
    rex.n_features_in_ = 2
    rex.feature_names = ["A", "B"]
    rex.prog_bar = False
    rex.verbose = False
    rex.models = object()

    matrices = [
        np.array([[0, 1], [0, 0]]),
        np.array([[0, 0], [1, 0]]),
    ]

    def fake_iteration(iter_idx, *_args, **_kwargs):
        return matrices[iter_idx]

    monkeypatch.setattr(Rex, "_bootstrap_iteration", staticmethod(fake_iteration))

    X = pd.DataFrame([[1, 2], [3, 4]], columns=rex.feature_names)
    result = rex._build_bootstrapped_adjacency_matrix(X, num_iterations=2, sampling_split=1.0)

    assert result.tolist() == [[0.0, 0.5], [0.5, 0.0]]


@pytest.mark.internal
def test_build_bootstrapped_adjacency_matrix_requires_fit():
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    rex.is_fitted_ = False
    X = pd.DataFrame([[1, 2]], columns=["A", "B"])

    with pytest.raises(ValueError):
        rex._build_bootstrapped_adjacency_matrix(X)


def test_find_best_tolerance(monkeypatch):
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    metric_values = [0.3, 0.6] + [0.4] * 17
    call_count = {"i": -1}

    def fake_dag_from_bootstrap_adj_matrix(_adjacency, tolerance):
        # Map tolerance progression to metric index.
        call_count["i"] += 1
        return nx.DiGraph([("A", "B")])

    def fake_evaluate_graph(_ref, _dag):
        return SimpleNamespace(f1=metric_values[call_count["i"]])

    monkeypatch.setattr(rex, "_dag_from_bootstrap_adj_matrix", fake_dag_from_bootstrap_adj_matrix)
    monkeypatch.setattr(rex_module, "evaluate_graph", fake_evaluate_graph)

    tolerance = rex._find_best_tolerance(
        ref_graph=nx.DiGraph(),
        key_metric="f1",
        direction="maximize",
        iter_adjacency_matrix=np.zeros((1, 1)),
    )

    assert tolerance == pytest.approx(0.15)
    assert metric_values[1] == 0.6


def test_score_handles_cases(monkeypatch):
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    rex.feature_names = ["A", "B"]
    metrics_obj = SimpleNamespace(score=True)

    def fake_evaluate_graph(ref_graph, pred_graph, feature_names=None):
        assert feature_names == rex.feature_names
        return metrics_obj

    monkeypatch.setattr(rex_module, "evaluate_graph", fake_evaluate_graph)

    rex.G_final = nx.DiGraph([("A", "B")])
    rex.G_shap = nx.DiGraph([("A", "B")])
    rex.G_indep = nx.DiGraph([("A", "B")])

    assert rex.score(ref_graph=None) is None
    assert rex.score(ref_graph=nx.DiGraph()) == metrics_obj
    assert rex.score(ref_graph=nx.DiGraph(), predicted_graph="shap") == metrics_obj
    assert rex.score(ref_graph=nx.DiGraph(), predicted_graph="indep") == metrics_obj
    with pytest.raises(ValueError):
        rex.score(ref_graph=nx.DiGraph(), predicted_graph="invalid")


def test_summarize_knowledge(monkeypatch):
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    df = object()

    class DummyKnowledge:
        def __init__(self, rex_instance, graph):
            self.rex = rex_instance
            self.graph = graph

        def info(self):
            return df

    monkeypatch.setattr(rex_module, "Knowledge", DummyKnowledge)

    assert rex.summarize_knowledge(ref_graph=None) is None

    result = rex.summarize_knowledge(ref_graph=nx.DiGraph())
    assert result is df
    assert isinstance(rex.knowledge, DummyKnowledge)


@pytest.mark.internal
def test_set_sampling_split():
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    # Formula produces a deterministic proportion based on bootstrap_trials.
    val = rex._set_sampling_split()
    assert 0 < val < 1


@pytest.mark.internal
def test_steps_from_hpo(monkeypatch):
    rex = Rex(name="demo", model_type="nn", explainer="gradient", hpo_n_trials=5)

    class DummyPipeline:
        def __init__(self):
            self._args = {"hpo_n_trials": [3, 2]}

        def contains_method(self, name, exact_match=True):
            return 1 if "tune_fit" in name else 0

        def contains_argument(self, name):
            return name in self._args

        def get_argument_value(self, name):
            return self._args[name][0]

        def all_argument_values(self, name):
            return self._args.get(name, [])

    fit_steps = DummyPipeline()
    assert rex._steps_from_hpo(fit_steps) == 3
