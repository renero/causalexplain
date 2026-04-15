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


def test_bootstrapped_adjacency_matrix_defaults_to_none():
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    assert rex.bootstrapped_adjacency_matrix is None


def test_bootstrap_preserves_raw_adjacency_matrix(monkeypatch):
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    rex.feature_names = ["A", "B"]
    rex.explainer = "gradient"
    rex.shaps = None

    raw_matrix = np.array([[0.0, 0.6], [0.2, 0.0]])

    def fake_build(*_args, **_kwargs):
        return raw_matrix.copy()

    def fake_dag_from_bootstrap_adj_matrix(adjacency, tolerance):
        assert tolerance == 0.3
        assert np.array_equal(adjacency, raw_matrix)
        return nx.DiGraph([("A", "B")])

    monkeypatch.setattr(rex, "_build_bootstrapped_adjacency_matrix", fake_build)
    monkeypatch.setattr(rex, "_dag_from_bootstrap_adj_matrix", fake_dag_from_bootstrap_adj_matrix)

    dag = rex.bootstrap(
        pd.DataFrame([[1, 2], [3, 4]], columns=rex.feature_names),
        tolerance=0.3,
    )

    assert list(dag.edges()) == [("A", "B")]
    assert np.array_equal(rex.bootstrapped_adjacency_matrix, raw_matrix)


def test_bootstrapped_adjacency_matrix_survives_save_load(tmp_path):
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    rex.feature_names = ["A", "B"]
    rex.bootstrapped_adjacency_matrix = np.array([[0.0, 0.5], [0.5, 0.0]])

    rex_module.utils.save_experiment(
        "rex_bootstrap",
        str(tmp_path),
        rex,
        overwrite=True,
    )
    loaded = rex_module.utils.load_experiment("rex_bootstrap", str(tmp_path))

    assert np.array_equal(
        loaded.bootstrapped_adjacency_matrix,
        rex.bootstrapped_adjacency_matrix,
    )
    assert loaded.feature_names == rex.feature_names


def test_get_diagnostics_bundle_requires_predict_artifacts():
    rex = Rex(name="demo", model_type="nn", explainer="gradient")

    with pytest.raises(RuntimeError, match="Regressor errors are unavailable"):
        rex.get_diagnostics_bundle()


def test_get_diagnostics_bundle_returns_normalized_frames():
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    rex.feature_names = ["A", "B", "C"]
    rex.models = SimpleNamespace(
        scoring=np.array([0.1, np.float32(0.2), 0.3], dtype=object))
    rex.bootstrapped_adjacency_matrix = np.array([
        [0.0, 0.8, 0.0],
        [0.0, 0.0, 0.4],
        [0.2, 0.0, 0.0],
    ], dtype=np.float32)
    rex.shaps = SimpleNamespace(
        shap_mean_values={
            "A": np.array([0.5, 0.6], dtype=np.float32),
            "B": np.array([0.7, 0.8], dtype=np.float32),
            "C": np.array([0.9, 1.0], dtype=np.float32),
        }
    )
    rex.tolerance = 0.3

    bundle = rex.get_diagnostics_bundle()

    assert set(bundle) == {
        "metadata",
        "regressor_errors",
        "bootstrap_matrix",
        "bootstrap_edges",
        "shap_mean_matrix",
        "shap_mean_long",
    }
    assert bundle["regressor_errors"]["target"].tolist() == ["A", "B", "C"]
    assert bundle["regressor_errors"]["error"].tolist() == pytest.approx(
        [0.1, 0.2, 0.3])
    assert bundle["bootstrap_matrix"].loc["A", "B"] == pytest.approx(0.8)
    assert bundle["bootstrap_edges"]["source"].tolist() == ["A", "B", "C"]
    assert bundle["bootstrap_edges"]["target"].tolist() == ["B", "C", "A"]
    assert bundle["bootstrap_edges"]["weight"].tolist() == pytest.approx(
        [0.8, 0.4, 0.2])
    assert np.isnan(bundle["shap_mean_matrix"].loc["A", "A"])
    assert bundle["shap_mean_matrix"].loc["A", "B"] == pytest.approx(0.5)
    assert bundle["shap_mean_matrix"].loc["A", "C"] == pytest.approx(0.6)
    assert len(bundle["shap_mean_long"]) == 6


def test_get_diagnostics_bundle_optionally_includes_knowledge(monkeypatch):
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    rex.feature_names = ["A", "B"]
    rex.models = SimpleNamespace(scoring=np.array([0.1, 0.2], dtype=object))
    rex.bootstrapped_adjacency_matrix = np.array([[0.0, 0.3], [0.0, 0.0]])
    rex.shaps = SimpleNamespace(
        shap_mean_values={
            "A": np.array([0.4], dtype=np.float32),
            "B": np.array([0.5], dtype=np.float32),
        }
    )

    knowledge = pd.DataFrame({"origin": ["A"], "target": ["B"]})
    monkeypatch.setattr(rex, "summarize_knowledge", lambda graph: knowledge)

    bundle = rex.get_diagnostics_bundle(
        include_knowledge=True, ref_graph=nx.DiGraph([("A", "B")]))

    assert bundle["knowledge"].equals(knowledge)


def test_export_diagnostics_writes_workbook(monkeypatch, tmp_path):
    rex = Rex(name="demo", model_type="nn", explainer="gradient")
    bundle = {
        "metadata": pd.DataFrame({"field": ["name"], "value": ["demo"]}),
        "bootstrap_matrix": pd.DataFrame(
            [[0.0, 0.2], [0.0, 0.0]],
            index=["A", "B"],
            columns=["A", "B"],
        ),
    }
    monkeypatch.setattr(rex, "get_diagnostics_bundle", lambda **_kwargs: bundle)

    writes = []

    class DummyWriter:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            writes.append(("writer", self.path))
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_to_excel(self, writer, sheet_name, index=False, **_kwargs):
        writes.append((sheet_name, index))

    monkeypatch.setattr(rex_module.pd, "ExcelWriter", DummyWriter)
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)

    exported = rex.export_diagnostics(str(tmp_path / "diag_report"))

    assert exported.endswith(".xlsx")
    assert ("writer", str(tmp_path / "diag_report.xlsx")) in writes
    assert ("metadata", False) in writes
    assert ("bootstrap_matrix", True) in writes


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
