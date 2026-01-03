import numpy as np
import pandas as pd
import pytest

from causalexplain.models import gbt


def _dataframe():
    return pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0, 1, 0, 1]})


def test_gbt_fit_predict_and_score(monkeypatch):
    monkeypatch.setattr(gbt, "ProgBar", lambda *_, **__: type("PB", (), {"start_subtask": lambda *a, **k: None, "update_subtask": lambda *a, **k: None, "remove": lambda *a, **k: None})())
    df = _dataframe()
    model = gbt.GBTRegressor(random_state=0, n_estimators=10, prog_bar=False)
    model.fit(df)

    assert "x" in model.regressor and "y" in model.regressor
    # Binary target should trigger the classifier path.
    assert model.regressor["y"].__class__.__name__ == "GradientBoostingClassifier"

    preds = model.predict(df)
    assert preds.shape[0] == len(df.columns)

    scores = model.score(df)
    assert scores.shape[0] == len(df.columns)
    assert np.all(scores <= 1.0)


def test_gbt_predict_requires_fit():
    model = gbt.GBTRegressor()
    with pytest.raises(AttributeError):
        model.predict(_dataframe())


def test_gbt_keeps_correlated_features(monkeypatch):
    monkeypatch.setattr(gbt, "ProgBar", lambda *_, **__: None)

    df = _dataframe()
    df["z"] = df["x"]
    base = gbt.GBTRegressor(correlation_th=None, prog_bar=False, n_estimators=5)
    with_corr = gbt.GBTRegressor(correlation_th=0.5, prog_bar=False, n_estimators=5)
    base.fit(df)
    with_corr.fit(df)

    expected_features = len(df.columns) - 1
    assert all(reg.n_features_in_ == expected_features for reg in with_corr.regressor.values())
    assert {
        name: reg.n_features_in_
        for name, reg in base.regressor.items()
    } == {
        name: reg.n_features_in_
        for name, reg in with_corr.regressor.items()
    }


def test_gbt_tune_and_tune_fit(monkeypatch):
    monkeypatch.setattr(gbt.utils, "cast_categoricals_to_int", lambda X: X)

    class FakeTrial:
        def __init__(self):
            self.params = {}

        def suggest_float(self, name, low, high):
            self.params[name] = low
            return low

        def suggest_int(self, name, low, high):
            self.params[name] = low
            return low

    class FakeFrozen:
        def __init__(self, params, value):
            self.params = params
            self.values = [value]

    FakeClassifier = type(
        "GradientBoostingClassifier",
        (),
        {
            "predict": lambda self, X: [1 for _ in range(len(X))],
            "score": lambda self, *_, **__: 0.9,
        },
    )

    FakeRegressor = type(
        "GradientBoostingRegressor",
        (),
        {
            "predict": lambda self, X: [0.5 for _ in range(len(X))],
            "score": lambda self, *_, **__: 0.8,
        },
    )

    def fake_fit(self, X):
        self.feature_names = list(X.columns)
        self.regressor = {
            self.feature_names[0]: FakeRegressor(),
            self.feature_names[1]: FakeClassifier(),
        }
        self.is_fitted_ = True
        return self

    monkeypatch.setattr(gbt.GBTRegressor, "fit", fake_fit)

    class FakeStudy:
        def __init__(self):
            self.best_trials = []
            self.best_value = float("inf")

        def optimize(self, objective, n_trials, show_progress_bar=None, callbacks=None, **kwargs):
            trial = FakeTrial()
            value = objective(trial)
            self.best_trials = [FakeFrozen(trial.params, value)]
            self.best_value = value

    monkeypatch.setattr(gbt.optuna, "create_study", lambda **kwargs: FakeStudy())
    monkeypatch.setattr(gbt.optuna.logging, "set_verbosity", lambda *_, **__: None)

    df = _dataframe()
    model = gbt.GBTRegressor(prog_bar=False, silent=True)
    args = model.tune(df, df, n_trials=1)
    assert "learning_rate" in args and model.best_params

    called = {}

    def fake_tune(self, *_, **__):
        called["tune"] = True
        self.min_tunned_loss = 0.1
        return args

    monkeypatch.setattr(gbt.GBTRegressor, "tune", fake_tune)
    monkeypatch.setattr(gbt.GBTRegressor, "fit", fake_fit)
    model.tune_fit(df, hpo_n_trials=1)
    assert called.get("tune") is True


def test_gbt_fit_with_progress_bar(monkeypatch):
    FakeRegressor = type(
        "GradientBoostingRegressor",
        (),
        {
            "__init__": lambda self, *args, **kwargs: None,
            "fit": lambda self, X, y: self._set_features(X),
            "_set_features": lambda self, X: setattr(self, "n_features_in_", X.shape[1]) or self,
            "predict": lambda self, X: np.zeros(len(X)),
            "score": lambda self, X, y: 0.5,
        },
    )
    FakeClassifier = type(
        "GradientBoostingClassifier",
        (),
        {
            "__init__": lambda self, *args, **kwargs: None,
            "fit": lambda self, X, y: self,
            "predict": lambda self, X: np.ones(len(X)),
            "score": lambda self, X, y: 0.5,
        },
    )

    monkeypatch.setattr(gbt, "GradientBoostingRegressor", FakeRegressor)
    monkeypatch.setattr(gbt, "GradientBoostingClassifier", FakeClassifier)
    monkeypatch.setattr(
        gbt,
        "ProgBar",
        lambda *_, **__: type(
            "PB", (), {"start_subtask": lambda self, *a, **k: self, "update_subtask": lambda *a, **k: None, "remove": lambda *a, **k: None}
        )(),
    )
    monkeypatch.setattr(gbt.utils, "get_feature_names", lambda X: list(X.columns))
    monkeypatch.setattr(
        gbt.utils,
        "get_feature_types",
        lambda X: {list(X.columns)[0]: "numerical", list(X.columns)[1]: "binary"},
    )

    df = _dataframe()
    model = gbt.GBTRegressor(prog_bar=True, correlation_th=0.5, n_estimators=2, verbose=False)
    model.fit(df)
    preds = model.predict(df)
    scores = model.score(df)
    assert preds.shape[0] == len(df.columns)
    assert len(scores) == len(df.columns)


def test_gbt_fit_sets_caller_name_from_call(monkeypatch):
    monkeypatch.setattr(gbt.inspect, "getouterframes", lambda *_, **__: [(None, None, None, "__call__", None, None)])
    FakeRegressor = type(
        "GradientBoostingRegressor",
        (),
        {
            "__init__": lambda self, *args, **kwargs: None,
            "fit": lambda self, X, y: self,
            "predict": lambda self, X: np.zeros(len(X)),
            "score": lambda self, X, y: 0.5,
        },
    )
    monkeypatch.setattr(gbt, "GradientBoostingRegressor", FakeRegressor)
    monkeypatch.setattr(gbt, "GradientBoostingClassifier", FakeRegressor)
    monkeypatch.setattr(gbt.utils, "get_feature_names", lambda X: list(X.columns))
    monkeypatch.setattr(gbt.utils, "get_feature_types", lambda X: {c: "numerical" for c in X.columns})
    df = _dataframe()
    model = gbt.GBTRegressor(prog_bar=False, n_estimators=1)
    model.fit(df)


def test_gbt_fit_parallel_jobs(monkeypatch):
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

    FakeRegressor = type(
        "GradientBoostingRegressor",
        (),
        {
            "__init__": lambda self, *args, **kwargs: None,
            "fit": lambda self, X, y: setattr(self, "n_features_in_", X.shape[1]) or self,
            "predict": lambda self, X: np.zeros(len(X)),
            "score": lambda self, X, y: 0.5,
        },
    )
    FakeClassifier = type(
        "GradientBoostingClassifier",
        (),
        {
            "__init__": lambda self, *args, **kwargs: None,
            "fit": lambda self, X, y: setattr(self, "n_features_in_", X.shape[1]) or self,
            "predict": lambda self, X: np.ones(len(X)),
            "score": lambda self, X, y: 0.5,
        },
    )

    monkeypatch.setattr(gbt, "GradientBoostingRegressor", FakeRegressor)
    monkeypatch.setattr(gbt, "GradientBoostingClassifier", FakeClassifier)
    monkeypatch.setattr(gbt.utils, "get_feature_names", lambda X: list(X.columns))
    monkeypatch.setattr(
        gbt.utils,
        "get_feature_types",
        lambda X: {list(X.columns)[0]: "numerical", list(X.columns)[1]: "binary"},
    )

    df = _dataframe()
    model = gbt.GBTRegressor(prog_bar=False, parallel_jobs=2, n_estimators=1)
    model.fit(df)

    assert set(model.regressor.keys()) == set(df.columns)
    assert model.regressor["x"].n_features_in_ == 1
    assert model.regressor["y"].n_features_in_ == 1


def test_gbt_custom_main_runs_with_mocks(monkeypatch):
    monkeypatch.setattr(gbt.utils, "graph_from_dot_file", lambda *_: None)

    class DummyScaler:
        def fit_transform(self, data):
            return data.values

    monkeypatch.setattr(gbt, "StandardScaler", lambda *_, **__: DummyScaler())
    monkeypatch.setattr(gbt.pd, "read_csv", lambda *_: _dataframe())
    gbt.custom_main(experiment_name="dummy", tune=False, score=False)


def test_gbt_custom_main_tune_branch(monkeypatch):
    monkeypatch.setattr(gbt.utils, "graph_from_dot_file", lambda *_: None)
    monkeypatch.setattr(gbt.pd, "read_csv", lambda *_: _dataframe())

    class DummyScaler:
        def fit_transform(self, data):
            return data.values

    class DummyGBT:
        def __init__(self, *_, **__):
            self.tuned = False

        def tune_fit(self, *_, **__):
            self.tuned = True

        def score(self, *_):
            return [0.0]

    monkeypatch.setattr(gbt, "StandardScaler", lambda *_, **__: DummyScaler())
    monkeypatch.setattr(gbt, "GBTRegressor", DummyGBT)
    gbt.custom_main(experiment_name="dummy", tune=True, score=False)
