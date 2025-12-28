import inspect
from types import SimpleNamespace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LinearRegression

from causalexplain.explainability import hierarchies as hmod
from causalexplain.explainability import perm_importance as pimod
from causalexplain.explainability import regression_quality as rqmod
from causalexplain.explainability import shapley as smod


class DummyProgBar:
    def start_subtask(self, name, total):
        return self

    def update_subtask(self, name, value):
        return self

    def remove(self, name):
        return self


@pytest.fixture(autouse=True)
def patch_progbar(monkeypatch):
    monkeypatch.setattr(hmod, "ProgBar", DummyProgBar, raising=False)
    monkeypatch.setattr(pimod, "ProgBar", DummyProgBar, raising=False)
    monkeypatch.setattr(smod, "ProgBar", DummyProgBar, raising=False)


def test_hierarchies_fit_clusters_and_dissimilarity():
    data = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [1.1, 2.1, 2.9, 4.1],
            "c": [4.0, 3.0, 2.0, 1.0],
        }
    )
    h = hmod.Hierarchies(method="spearman", correlation_th=0.5)
    h.fit(data)

    # Exercise clustering branches
    clustered_corr, sorted_cols = h._cluster_features("spearman", threshold=0.5)
    assert set(sorted_cols) == set(data.columns)
    assert clustered_corr.shape == (3, 3)

    dis = h.hierarchical_dissimilarities()
    assert set(dis.columns) == set(data.columns)
    assert np.all(dis.values >= 0)


def test_hierarchies_helpers_and_plot(monkeypatch):
    data = pd.DataFrame({"a": [1, 2, 1.5], "b": [2, 1, 3]})
    corr = hmod.Hierarchies.compute_correlation_matrix(data, method="spearman")
    correlated = hmod.Hierarchies.compute_correlated_features(
        corr, 0.1, ["a", "b"], verbose=True
    )
    assert isinstance(correlated, dict)

    linkage_mat = np.array([[0, 1, 0.5, 2]])
    clusters = hmod.Hierarchies()._clusters_from_linkage(linkage_mat, ["a", "b"])
    assert hmod.Hierarchies._get_cluster(clusters, "a") is not None
    assert not hmod.Hierarchies._is_cluster("a")
    assert hmod.Hierarchies._contains_a_cluster(clusters, "K2") is False
    assert hmod.Hierarchies._get_cluster_element(clusters, "K2") is None
    assert hmod.Hierarchies._in_cluster(clusters["K2"], "a")
    assert hmod.Hierarchies._in_same_cluster(clusters, "a", "b") == "K2"
    assert hmod.Hierarchies()._are_connected(clusters, "a", "b") == pytest.approx(0.5)

    monkeypatch.setattr(hmod.plt, "show", lambda *args, **kwargs: None)
    Z = hmod.plot_dendogram_correlations(corr, ["a", "b"], figsize=(2, 2))
    assert Z is not None


def test_hierarchies_expand_clusters_and_connect_verbose(monkeypatch):
    h = hmod.Hierarchies()
    h.data = pd.DataFrame({"a": [1, 2], "b": [2, 1]})
    h.linkage_mat = np.array([[0, 1, 0.2, 2]])
    pi = SimpleNamespace(
        feature_importances_={
            "a": {"b": 0.1},
            "b": {"a": 0.2},
        },
        regression_importances_=[0.5, 0.6],
    )
    ground_truth = nx.DiGraph([("a", "b")])
    h.expand_clusters_perm_importance(pi, ground_truth)

    G = nx.DiGraph()
    G.add_edge("a", "b")
    linkage_mat = np.array([[0, 1, 0.1, 2]])
    conn = hmod.connect_hierarchies(G, linkage_mat, ["a", "b"], verbose=True)
    assert isinstance(conn, nx.DiGraph)


def test_hierarchies_correlated_features_verbose():
    corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], columns=["a", "b"], index=["a", "b"])
    correlated = hmod.Hierarchies.compute_correlated_features(corr, 0.4, ["a", "b"], verbose=True)
    assert correlated["a"] == ["b"]


def test_hierarchies_correlated_features_no_threshold():
    corr = pd.DataFrame(np.eye(2), columns=["a", "b"], index=["a", "b"])
    correlated = hmod.Hierarchies.compute_correlated_features(corr, None, ["a", "b"])
    assert correlated == {}


def test_hierarchies_nested_cluster_connection():
    clusters = {
        "K2": [("a", "K3"), 0.4],
        "K3": [("b", "c"), 0.7],
    }
    h = hmod.Hierarchies()
    assert h._are_connected(clusters, "a", "b") == 0.4


def test_connect_isolated_nodes_skips_large_clusters():
    G = nx.DiGraph()
    linkage_mat = np.array([[0, 1, 0.1, 3]])
    result = hmod.connect_isolated_nodes(G, linkage_mat, ["a", "b"], verbose=True)
    assert isinstance(result, nx.DiGraph)


def test_regression_quality_predict_and_verbose(capsys):
    assert isinstance(rqmod.RegQuality(), rqmod.RegQuality)
    scores = [0.1, 0.2, 5.0, 6.0]
    outliers = rqmod.RegQuality.predict(scores, gamma_shape=1, gamma_scale=1, threshold=0.9)
    assert outliers == set()

    _ = rqmod.RegQuality._mad_criteria(scores, verbose=True)
    verbose_out = capsys.readouterr().out
    assert "Median" in verbose_out

    gamma_indices = rqmod.RegQuality._gamma_criteria(scores, threshold=0.2, verbose=True)
    assert gamma_indices == {2, 3}


def _build_simple_models(df):
    # Simple container with regressors and predict
    class SimpleModels:
        def __init__(self, data):
            self.regressor = {}
            for col in data.columns:
                X = data.drop(columns=[col]).values
                y = data[col].values
                # Lightweight sklearn-style regressor
                lr = LinearRegression().fit(X, y)
                self.regressor[col] = lr

        def predict(self, X):
            return X.values

    return SimpleModels(df)


def test_permutation_importance_sklearn_flow(monkeypatch):
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.1, 2.1, 2.9, 4.1],
            "z": [0.5, 0.4, 0.6, 0.7],
        }
    )
    models = _build_simple_models(df)
    original_add_zeroes = pimod.PermutationImportance._add_zeroes

    def fake_add_zeroes(self, target, correlated_features):
        # track that we hit correlation branch
        self._zeroed = (target, tuple(correlated_features))
        return original_add_zeroes(self, target, correlated_features)

    # Avoid building DAGs via utils; only return edges passed in.
    def fake_digraph(X, feature_names, models_arg, connections, root_causes, reciprocity=True, anm_iterations=10, verbose=False):
        g = nx.DiGraph()
        for src, targets in connections.items():
            for tgt in targets:
                g.add_edge(tgt, src)
        return g

    monkeypatch.setattr(pimod.utils, "digraph_from_connected_features", fake_digraph)
    monkeypatch.setattr(pimod.PermutationImportance, "_add_zeroes", fake_add_zeroes, raising=False)

    pi = pimod.PermutationImportance(
        models=models,
        correlation_th=0.8,
        n_repeats=1,
        mean_pi_percentile=0.5,
        prog_bar=False,
        verbose=False,
    )
    pi.fit(df)
    graph = pi.predict(df)
    assert isinstance(graph, nx.DiGraph)
    assert pi.mean_pi_threshold >= 0.0
    assert hasattr(pi, "_zeroed")

    # Plot uses computed metrics and should not crash
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    returned = pi._plot_perm_imp("x", ax=ax)
    assert returned is not None


def test_shuffle_tensor_column_and_loss_computation():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pi = pimod.PermutationImportance(models=SimpleNamespace(regressor={"a": None}))
    shuffled = pi._shuffle_2Dtensor_column(tensor, 1)
    assert shuffled.shape == tensor.shape
    # Dummy model/dataloader to exercise loss computation
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_fn = torch.nn.MSELoss()

        def forward(self, x):
            return x.sum(dim=1, keepdim=True)

    dataset = [(torch.tensor([1.0, 2.0]), torch.tensor([3.0]))]
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = DummyModel()
    avg, std, losses = pi._compute_loss_shuffling_column(model, loader, shuffle_col=0)
    assert losses.size >= 1
    assert avg == pytest.approx(losses.mean())


def test_compute_perm_imp_repeats(monkeypatch):
    class DummyReg:
        def __init__(self):
            self.columns = ["f1", "f2", "f3"]
            X = torch.tensor([[1.0, 2.0, 3.0]])
            y = torch.tensor([[6.0]])
            self.train_loader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=1)
            self.model = DummyModel()

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_fn = torch.nn.MSELoss()

        def forward(self, x):
            return x.sum(dim=1, keepdim=True)

    reg = DummyReg()
    pi = pimod.PermutationImportance(models=SimpleNamespace(regressor={"t": reg}), n_repeats=2, prog_bar=False, verbose=False)
    pi.base_loss = {"t": 0.0}
    mean, std = pi._compute_perm_imp("t", reg, reg.model, num_vars=3)
    assert mean.shape[0] == 2


def test_permutation_importance_corr_branch_plot(monkeypatch):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.1, 2.1, 3.1], "z": [0.5, 0.6, 0.7]})
    models = _build_simple_models(df)
    monkeypatch.setitem(pimod.plt.rcParams, "text.usetex", False)
    import matplotlib.texmanager as texmanager
    monkeypatch.setattr(texmanager.TexManager, "_run_checked_subprocess", lambda *a, **k: None)
    monkeypatch.setattr(pimod, "subplots", lambda fn, *args, **kwargs: "stubbed")
    pi = pimod.PermutationImportance(
        models=models, correlation_th=0.5, n_repeats=1, prog_bar=False, verbose=True
    )
    pi._fit_sklearn(df)
    # Ensure correlation branches set correlated_features
    assert any(len(v) >= 0 for v in pi.correlated_features.values())
    pi.connections = {name: ["dummy"] for name in pi.feature_names}
    assert pi.plot(figsize=(2, 2)) == "stubbed"


def test_permutation_importance_obtain_corr_info():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.1, 2.1, 3.1]})
    models = _build_simple_models(df)
    pi = pimod.PermutationImportance(models=models, correlation_th=0.5, prog_bar=False, verbose=False)
    pi._obtain_correlation_info(df)
    assert pi.corr_matrix is not None


def test_permutation_importance_no_corr_branch():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 3.0]})
    models = _build_simple_models(df)
    pi = pimod.PermutationImportance(models=models, correlation_th=None, prog_bar=False, verbose=False)
    pi.fit(df)


def test_permutation_importance_add_zeroes_branch(monkeypatch):
    pi = pimod.PermutationImportance(models=SimpleNamespace(regressor={"x": None, "y": None}), prog_bar=False, verbose=False)
    pi.feature_names = ["x", "y"]
    pi.pi = {"x": {"importances_mean": np.array([0.1]), "importances_std": np.array([0.01])}}
    pi._add_zeroes("x", ["y"])
    assert pi.pi["x"]["importances_mean"].shape[0] == 2


def test_permutation_importance_pytorch_path(monkeypatch):
    # Dummy torch-style regressor/model
    class TorchReg:
        def __init__(self):
            self.model = DummyModel()
            self.columns = ["f1", "f2"]
            X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            y = torch.tensor([[3.0], [7.0]])
            self.train_loader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=1)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_fn = torch.nn.MSELoss()

        def forward(self, x):
            return x.sum(dim=1, keepdim=True)

    reg = TorchReg()
    models = SimpleNamespace(regressor={"a": reg, "b": reg})

    monkeypatch.setattr(
        pimod.utils,
        "valid_candidates_from_prior",
        lambda feats, tgt, prior: [f for f in feats if f != tgt],
    )
    monkeypatch.setattr(pimod.utils, "digraph_from_connected_features", lambda X, *args, **kwargs: nx.DiGraph())
    monkeypatch.setattr(pimod.utils, "break_cycles_if_present", lambda g, *args, **kwargs: g)

    pi = pimod.PermutationImportance(
        models=models,
        n_repeats=1,
        prog_bar=False,
        verbose=True,
    )
    pi._fit_pytorch()
    pi.mean_pi_threshold = 0.0
    pi.prior = []
    g = pi._predict_pytorch(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), root_causes=None)
    assert isinstance(g, nx.DiGraph)


def test_permutation_importance_fit_dispatch(monkeypatch):
    class TorchReg:
        def __init__(self):
            self.model = DummyModel()
            self.columns = ["x", "y"]
            X = torch.tensor([[1.0, 2.0]])
            y = torch.tensor([[3.0]])
            self.train_loader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=1)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_fn = torch.nn.MSELoss()

        def forward(self, x):
            return x.sum(dim=1, keepdim=True)

    monkeypatch.setattr(pimod, "MLPModel", TorchReg)
    models = SimpleNamespace(regressor={"x": TorchReg()})
    pi = pimod.PermutationImportance(models=models, prog_bar=False, verbose=False, n_repeats=1)
    pi.fit(pd.DataFrame({"x": [1.0], "y": [2.0]}))


def test_shap_estimator_fit_predict_and_adjust(monkeypatch):
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [1.5, 2.5, 3.5, 4.5],
            "c": [0.5, 1.0, 1.5, 2.0],
        }
    )

    class FakeModels:
        def __init__(self, columns):
            self.regressor = {col: SimpleNamespace(predict=lambda X: np.ones(len(X))) for col in columns}

        def predict(self, X):
            return X.values

    models = FakeModels(df.columns)

    def fake_run_explainer(self, target_name, model, X_train, X_test):
        # Return a small, deterministic SHAP matrix
        num_features = X_train.shape[1]
        return np.tile(np.arange(1, num_features + 1), (len(X_test), 1))

    monkeypatch.setattr(smod.ShapEstimator, "_run_selected_shap_explainer", fake_run_explainer)
    monkeypatch.setattr(smod.utils, "valid_candidates_from_prior", lambda feats, t, p: [f for f in feats if f != t])
    monkeypatch.setattr(
        smod.utils,
        "digraph_from_connected_features",
        lambda X, feature_names, models_arg, connections, root_causes, prior=None, reciprocity=False, anm_iterations=10, verbose=False: nx.DiGraph(
            [(parent, target) for target, parents in connections.items() for parent in parents]
        ),
    )
    monkeypatch.setattr(smod.utils, "break_cycles_if_present", lambda g, *args, **kwargs: g)

    sh = smod.ShapEstimator(models=models, mean_shap_percentile=0.5, prog_bar=False, verbose=False)
    sh.fit(df)
    dag = sh.predict(df)
    assert isinstance(dag, nx.DiGraph)
    assert sh.mean_shap_threshold >= 0.0
    assert sh.discrepancies is not None

    # Adjust edges based on discrepancies to exercise reversal logic
    graph = nx.DiGraph()
    graph.add_edges_from([("a", "b"), ("b", "a")])
    adjusted = sh._adjust_edges_from_shap_discrepancies(graph, increase_tolerance=0.1, sd_upper=1.0)
    assert isinstance(adjusted, nx.DiGraph)

    # Plot summary uses computed fields
    fig = sh._plot_shap_summary("a", ax=None, figsize=(3, 2))
    assert fig is not None


def test_shap_helpers_vector_and_error_contribution():
    sh = smod.ShapEstimator(models=None, prog_bar=False, verbose=False)
    discrepancies = pd.DataFrame({"a": [0.1, 0.2], "b": [0.2, 0.1]}, index=["a", "b"])
    vec = sh._input_vector(discrepancies, "a", "b", target_mean=0.15)
    assert vec.shape == (7,)

    predictions = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    reshaped = sh._adjust_predictions_shape(predictions, (2, 2))
    assert reshaped.shape == (2, 2)

    shap_values = pd.DataFrame({"a": [0.1, 0.2], "b": [0.0, 0.1]})
    y_true = pd.Series([1.0, 2.0])
    y_pred = pd.Series([1.0, 2.5])
    contribution = sh._individual_error_contribution(shap_values, y_true, y_pred)
    assert "a" in contribution.index

    # Debug message should be silent when verbose=False
    sh._debugmsg("Ignored edge", "a", 0.2, "b", 0.0, [0.1, 0.2, 0.1, 0, 0, 0, 0], 0.5, [])


def test_shap_get_method_caller_name_handles_unknown(monkeypatch):
    sh = smod.ShapEstimator(models=None, prog_bar=False, verbose=False)
    # Force inspect.currentframe to raise
    monkeypatch.setattr(inspect, "currentframe", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert sh._get_method_caller_name() == "unknown"


def test_shap_discrepancy_value_error_paths(monkeypatch):
    sh = smod.ShapEstimator(models=None, prog_bar=False, verbose=False)
    monkeypatch.setattr(smod.sms, "het_breuschpagan", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("fail")))
    disc = sh._compute_discrepancy(np.array([1, 2]), np.array([1, 2]), np.array([[1], [2]]), "t", "p")
    assert isinstance(disc, smod.ShapDiscrepancy)


def test_shap_fit_target_uses_model_attr():
    class WithModel:
        def __init__(self):
            self.model = SimpleNamespace(cpu=lambda: self, predict=lambda X: np.ones(len(X)), cuda=lambda: self)
            self.predict = lambda X: np.ones(len(X))

    models = SimpleNamespace(regressor={"a": WithModel(), "b": SimpleNamespace(predict=lambda X: np.ones(len(X)))})
    res = smod.ShapEstimator._shap_fit_target_variable(
        "a",
        models,
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        pd.DataFrame({"a": [5, 6], "b": [7, 8]}),
        ["a", "b"],
        on_gpu=True,
        verbose=True,
        run_selected_shap_explainer_func=lambda *args, **kwargs: np.array([[1.0], [2.0]]),
    )
    assert res[0] == "a"


def test_shap_explainer_variants(monkeypatch):
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_test = np.array([[5.0, 6.0]])

    class DummyModel:
        def predict(self, X):
            return np.sum(X, axis=1)

        def to(self, device):
            return self

    # Kernel branch
    sh = smod.ShapEstimator(models=None, prog_bar=False, verbose=False)
    sh.explainer = "kernel"
    sh.shap_explainer = {}
    monkeypatch.setattr(
        smod.shap,
        "KernelExplainer",
        lambda predict, data: SimpleNamespace(shap_values=lambda x: [np.ones((len(x), data.shape[1]))]),
    )
    kernel_vals = sh._run_selected_shap_explainer("t", DummyModel(), X_train, X_test)
    assert kernel_vals.shape == (1, 2)

    # Gradient branch
    sh.explainer = "gradient"
    monkeypatch.setattr(
        smod.shap,
        "GradientExplainer",
        lambda model, data: lambda x: SimpleNamespace(
            values=np.full((x[0].shape[0], data[0].shape[1]), 2.0)
        ),
    )
    grad_vals = sh._run_selected_shap_explainer("t", DummyModel(), X_train, X_test)
    assert grad_vals.shape == (1, 2)

    # Explainer branch
    sh.explainer = "explainer"
    monkeypatch.setattr(
        smod.shap,
        "Explainer",
        lambda predict, data: lambda x, silent=True: SimpleNamespace(values=np.full((len(x), data.shape[1]), 3.0)),
    )
    exp_vals = sh._run_selected_shap_explainer("t", DummyModel(), X_train, X_test)
    assert exp_vals.shape == (1, 2)

    sh.explainer = "unknown"
    with pytest.raises(ValueError):
        sh._run_selected_shap_explainer("t", DummyModel(), X_train, X_test)


def test_shap_threshold_and_zero_injection():
    sh = smod.ShapEstimator(models=None, prog_bar=False, verbose=False)
    sh.mean_shap_percentile = None
    sh.all_mean_shap_values = np.array([1.0, 2.0])
    sh._compute_scaled_shap_threshold()
    assert sh.mean_shap_threshold == 0.0

    sh.feature_names = ["a", "b", "c"]
    sh.all_mean_shap_values = [np.array([0.2, 0.3])]
    sh._add_zeroes("a", ["b"])
    assert len(sh.all_mean_shap_values[-1]) == 3


def test_shap_verbose_debug_and_tolerance(monkeypatch):
    sh = smod.ShapEstimator(models=None, prog_bar=False, verbose=True)
    # Force verbose paths inside helpers
    discrepancies = pd.DataFrame([[0.1, 0.2], [0.05, 0.1]], columns=["a", "b"], index=["a", "b"])
    assert not sh._increase_upper_tolerance(discrepancies)
    big_disc = pd.DataFrame([[10.0, 0.0], [0.0, 0.0]], columns=["a", "b"], index=["a", "b"])
    assert sh._increase_upper_tolerance(big_disc)

    cycles = [["x", "y", "x"]]
    assert sh._nodes_in_cycles(cycles, "x", "y")

    sh.feature_names = ["a", "b"]
    sh.discrepancies = discrepancies
    graph = nx.DiGraph([("a", "b")])
    sh._debugmsg("Ignored edge", "a", 0.2, "b", 0.0, [0.1, 0.2, 0.1, 0, 0, 0, 0], 0.5, [])
    sh._debugmsg("(*) Reversed edge", "a", 0.2, "b", 0.0, [0.1, 0.2, 0.1, 0, 0, 0, 0], 0.5, cycles)
    sh._adjust_edges_from_shap_discrepancies(graph, increase_tolerance=0.0, sd_upper=0.5)


def test_shap_fit_parallel_branch(monkeypatch):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]})
    models = _build_simple_models(df)

    # Force use of multiprocessing branch while avoiding real pools
    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, func, iterable):
            for target in iterable:
                yield (target, np.ones((2, 1)), np.array([0]), np.array([0.2, 0.3]))

        def close(self):
            return None

        def join(self):
            return None

    class FakeCtx:
        def Pool(self, processes):
            return FakePool()

    monkeypatch.setattr(smod, "get_context", lambda *args, **kwargs: FakeCtx())
    monkeypatch.setattr(
        smod.ShapEstimator,
        "_shap_fit_target_variable",
        lambda *args, **kwargs: ("a", np.ones((2, 1)), np.array([0]), np.array([0.1, 0.2])),
    )

    sh = smod.ShapEstimator(models=models, mean_shap_percentile=0.5, prog_bar=False, verbose=False, parallel_jobs=1)
    sh.fit(df)
    assert sh.is_fitted_


def test_shap_fit_progress_bar(monkeypatch):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    models = _build_simple_models(df)
    sh = smod.ShapEstimator(models=models, mean_shap_percentile=0.5, prog_bar=True, verbose=False, parallel_jobs=0)
    sh.fit(df)
    assert sh.mean_shap_threshold >= 0


def test_shap_custom_main_and_sachs_main(monkeypatch):
    # Avoid filesystem access by mocking dependencies
    monkeypatch.setattr(smod.utils, "graph_from_dot_file", lambda *args, **kwargs: nx.DiGraph())
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pd.DataFrame({"a": [1, 2], "b": [2, 3]}))
    monkeypatch.setattr(
        smod, "StandardScaler", lambda *args, **kwargs: SimpleNamespace(fit_transform=lambda x: x)
    )

    class DummyModels(SimpleNamespace):
        def __init__(self):
            super().__init__()
            self.regressor = {"a": SimpleNamespace(predict=lambda X: np.ones(len(X)))}
            self.root_causes = []

    dummy_rex = SimpleNamespace(models=DummyModels(), root_causes=[], shaps=None, is_fitted_=True)
    monkeypatch.setattr(
        smod.utils, "load_experiment", lambda *args, **kwargs: dummy_rex
    )
    monkeypatch.setattr(smod, "ShapEstimator", lambda *args, **kwargs: SimpleNamespace(fit=lambda *a, **k: None, predict=lambda *a, **k: None))

    smod.custom_main("dummy", path="/tmp/", output_path="/tmp/", scale=True)
    smod.sachs_main()


def test_shap_predict_progress_bar(monkeypatch):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]})
    models = _build_simple_models(df)

    def fake_run_explainer(self, target_name, model, X_train, X_test):
        # Shape with trailing singleton dimension to trigger reshape
        return np.ones((len(X_test), X_train.shape[1], 1))

    monkeypatch.setattr(smod.ShapEstimator, "_run_selected_shap_explainer", fake_run_explainer)
    monkeypatch.setattr(smod.utils, "valid_candidates_from_prior", lambda feats, t, p: [f for f in feats if f != t])
    monkeypatch.setattr(
        smod.utils,
        "digraph_from_connected_features",
        lambda X, feature_names, models_arg, connections, root_causes, prior=None, reciprocity=False, anm_iterations=10, verbose=False: nx.DiGraph(),
    )
    monkeypatch.setattr(smod.utils, "break_cycles_if_present", lambda g, *args, **kwargs: g)

    sh = smod.ShapEstimator(models=models, mean_shap_percentile=0.5, prog_bar=True, verbose=False)
    sh.fit(df)
    sh.predict(df)
