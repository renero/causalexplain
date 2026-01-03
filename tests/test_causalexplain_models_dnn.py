import pandas as pd
import pytest
import torch

from causalexplain.models import dnn


class DummyTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.linear(x)


class DummyMLPModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.model = DummyTorchModel()
        self.trained = False

    def train(self):
        self.trained = True


def _small_frame():
    return pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 2.0, 3.0]})


def test_nnregressor_fit_predict_and_score(monkeypatch):
    monkeypatch.setattr(dnn, "MLPModel", DummyMLPModel)
    class FakeProgBar:
        def start_subtask(self, *_, **__):
            return self

        def update_subtask(self, *_, **__):
            return None

        def remove(self, *_, **__):
            return None

    monkeypatch.setattr(dnn, "ProgBar", FakeProgBar)

    df = _small_frame()
    rex = dnn.NNRegressor(
        hidden_dim=[2],
        activation="relu",
        num_epochs=1,
        batch_size=2,
        prog_bar=False,
        early_stop=False,
        min_delta=0.0,
    )
    rex.fit(df)

    preds = rex.predict(df)
    assert preds.shape[0] == len(df.columns)

    scores = rex.score(df)
    assert scores.shape[0] == len(df.columns)
    assert rex.is_fitted_ is True


@pytest.mark.skipif(
    not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
    reason="MPS not available",
)
def test_nnregressor_predict_on_mps(monkeypatch):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    monkeypatch.setattr(dnn, "MLPModel", DummyMLPModel)
    df = _small_frame()
    try:
        rex = dnn.NNRegressor(
            device="mps",
            prog_bar=False,
            early_stop=False,
            min_delta=0.0,
            num_epochs=1,
        )
        rex.fit(df)
        preds = rex.predict(df)
        assert preds.shape[0] == len(df.columns)
    finally:
        torch.set_default_dtype(old_dtype)


def test_nnregressor_predict_requires_fit(monkeypatch):
    monkeypatch.setattr(dnn, "MLPModel", DummyMLPModel)
    rex = dnn.NNRegressor()
    with pytest.raises(AttributeError):
        rex.predict(_small_frame())


def test_nnregressor_keeps_correlated_features(monkeypatch):
    monkeypatch.setattr(dnn, "MLPModel", DummyMLPModel)
    monkeypatch.setattr(dnn, "ProgBar", lambda *_, **__: None)

    df = _small_frame()
    df["c"] = df["a"]
    base = dnn.NNRegressor(correlation_th=None, prog_bar=False, early_stop=False, min_delta=0.0, num_epochs=1)
    with_corr = dnn.NNRegressor(correlation_th=0.5, prog_bar=False, early_stop=False, min_delta=0.0, num_epochs=1)
    base.fit(df)
    with_corr.fit(df)

    expected_features = len(df.columns)
    for model in with_corr.regressor.values():
        assert model.kwargs["input_size"] == expected_features
    assert {
        target: model.kwargs["input_size"]
        for target, model in base.regressor.items()
    } == {
        target: model.kwargs["input_size"]
        for target, model in with_corr.regressor.items()
    }


def test_nnregressor_fit_parallel_jobs(monkeypatch):
    class DummyExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers
            self.submitted = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            future = type("F", (), {"result": lambda self: fn(*args, **kwargs)})()
            self.submitted.append(future)
            return future

    monkeypatch.setattr(
        "concurrent.futures.ThreadPoolExecutor", DummyExecutor)
    monkeypatch.setattr(
        "concurrent.futures.as_completed", lambda futures: futures)
    monkeypatch.setattr(dnn, "MLPModel", DummyMLPModel)

    df = _small_frame()
    rex = dnn.NNRegressor(
        prog_bar=False,
        early_stop=False,
        min_delta=0.0,
        num_epochs=1,
        parallel_jobs=2,
        device="cpu",
    )
    rex.fit(df)

    assert set(rex.regressor.keys()) == set(df.columns)


def test_nnregressor_tune_runs_with_fake_optuna(monkeypatch):
    monkeypatch.setattr(dnn, "ColumnsDataset", lambda target, data: data)
    monkeypatch.setattr(
        dnn,
        "DataLoader",
        lambda dataset, batch_size, shuffle=False: [(torch.ones((1, 1)), torch.ones((1, 1)))],
    )

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_fn = torch.nn.MSELoss()

        def forward(self, x):
            return torch.zeros((x.shape[0], 1))

    class DummyWrapper:
        def __init__(self):
            self.model = DummyModel()

    def fake_fit(self, X):
        self.regressor = {name: DummyWrapper() for name in X.columns}
        return self

    monkeypatch.setattr(dnn.NNRegressor, "fit", fake_fit)

    class FakeTrial:
        def __init__(self):
            self.params = {}

        def suggest_int(self, name, low, high):
            self.params[name] = low
            return low

        def suggest_categorical(self, name, options):
            self.params[name] = options[0]
            return options[0]

        def suggest_loguniform(self, name, low, high):
            self.params[name] = low
            return low

        def suggest_uniform(self, name, low, high):
            self.params[name] = low
            return low

        def suggest_float(self, name, low, high):
            self.params[name] = low
            return low

    class FakeFrozen:
        def __init__(self, params, value):
            self.params = params
            self.values = [value]

    class FakeStudy:
        def __init__(self):
            self.best_trials = []
            self.best_value = float("inf")

        def optimize(self, objective, n_trials, show_progress_bar=None, callbacks=None, **kwargs):
            trial = FakeTrial()
            # Run compute_loss with a single repeat to keep things quick.
            original_compute = objective.compute_loss
            objective.compute_loss = lambda model, dataloader, n_repeats=10: original_compute(
                model, dataloader, n_repeats=1
            )
            value = objective(trial)
            self.best_trials = [FakeFrozen(trial.params, value)]
            self.best_value = value

    monkeypatch.setattr(dnn.optuna, "create_study", lambda **kwargs: FakeStudy())
    monkeypatch.setattr(dnn.optuna.logging, "set_verbosity", lambda *_, **__: None)

    df = _small_frame()
    rex = dnn.NNRegressor(num_epochs=1, batch_size=1, prog_bar=False, early_stop=False, min_delta=0.0)
    result = rex.tune(df, df, n_trials=1)
    assert "hidden_dim" in result


def test_nnregressor_tune_fit_and_repr(monkeypatch):
    called = {}

    def fake_tune(self, *_, **__):
        self.min_tunned_loss = 0.25
        return {
            "hidden_dim": [1],
            "activation": "relu",
            "learning_rate": 0.01,
            "dropout": 0.0,
            "batch_size": 1,
            "num_epochs": 1,
        }

    def fake_fit(self, X):
        called["fit"] = True
        self.is_fitted_ = True
        return self

    monkeypatch.setattr(dnn.NNRegressor, "tune", fake_tune)
    monkeypatch.setattr(dnn.NNRegressor, "fit", fake_fit)
    df = _small_frame()

    rex = dnn.NNRegressor()
    rex.tune_fit(df, hpo_n_trials=1)
    assert called.get("fit") is True
    assert "hidden_dim" in repr(rex)


def test_nnregressor_fit_with_progress_and_correlation(monkeypatch):
    monkeypatch.setattr(dnn.utils, "get_feature_names", lambda X: list(X.columns))
    monkeypatch.setattr(
        dnn.utils,
        "get_feature_types",
        lambda X: {list(X.columns)[0]: "categorical", list(X.columns)[1]: "binary"},
    )
    monkeypatch.setattr(dnn, "MLPModel", DummyMLPModel)

    monkeypatch.setattr(
        dnn,
        "ProgBar",
        lambda *_, **__: type(
            "PB", (), {"start_subtask": lambda self, *a, **k: self, "update_subtask": lambda *a, **k: None, "remove": lambda *a, **k: None}
        )(),
    )
    monkeypatch.setattr(dnn, "ColumnsDataset", lambda target, data: data)
    monkeypatch.setattr(
        dnn,
        "DataLoader",
        lambda dataset, batch_size, shuffle=False: [
            (torch.ones((len(dataset), 1)), torch.zeros((len(dataset), 1)))
        ],
    )

    df = _small_frame()
    rex = dnn.NNRegressor(
        prog_bar=True, correlation_th=0.5, num_epochs=1, batch_size=1, early_stop=False, min_delta=0.0
    )
    rex.fit(df)
    preds = rex.predict(df)
    scores = rex.score(df)
    assert preds.size > 0
    assert len(scores) == len(df.columns)


def test_nnregressor_fit_handles_stack_failure(monkeypatch):
    def raise_error(*_, **__):
        raise RuntimeError("boom")

    monkeypatch.setattr(dnn.inspect, "getouterframes", raise_error)
    monkeypatch.setattr(dnn, "MLPModel", DummyMLPModel)
    monkeypatch.setattr(dnn.utils, "get_feature_names", lambda X: list(X.columns))
    monkeypatch.setattr(dnn.utils, "get_feature_types", lambda X: {c: "binary" for c in X.columns})
    monkeypatch.setattr(dnn, "ColumnsDataset", lambda target, data: data)
    monkeypatch.setattr(
        dnn,
        "DataLoader",
        lambda dataset, batch_size, shuffle=False: [
            (torch.ones((len(dataset), 1)), torch.zeros((len(dataset), 1)))
        ],
    )
    monkeypatch.setattr(
        dnn,
        "ProgBar",
        lambda *_, **__: type(
            "PB", (), {"start_subtask": lambda self, *a, **k: self, "update_subtask": lambda *a, **k: None, "remove": lambda *a, **k: None}
        )(),
    )

    df = _small_frame()
    rex = dnn.NNRegressor(num_epochs=1, batch_size=1, early_stop=False, min_delta=0.0)
    rex.fit(df)


def test_dnn_custom_main_runs_with_mocks(monkeypatch):
    class DummyScaler:
        def fit_transform(self, data):
            return data.values

    monkeypatch.setattr(dnn.pd, "read_csv", lambda *_: _small_frame())
    monkeypatch.setattr(dnn, "StandardScaler", lambda *_, **__: DummyScaler())
    dnn.custom_main(tune=False, score=False)


def test_dnn_custom_main_tune_branch(monkeypatch):
    class DummyScaler:
        def fit_transform(self, data):
            return data.values

    class DummyNN:
        def __init__(self, *_, **__):
            self.called = False

        def tune_fit(self, *_args, **_kwargs):
            self.called = True

    monkeypatch.setattr(dnn.pd, "read_csv", lambda *_: _small_frame())
    monkeypatch.setattr(dnn, "StandardScaler", lambda *_, **__: DummyScaler())
    monkeypatch.setattr(dnn, "NNRegressor", DummyNN)
    dnn.custom_main(tune=True, score=False)
