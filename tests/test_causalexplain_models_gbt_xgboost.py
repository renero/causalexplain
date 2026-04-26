import numpy as np
import pandas as pd
import pytest

import causalexplain.models.gbt as gbt_module
from causalexplain.models.gbt import GBTRegressor, XGBoostGBTBackend

xgb = pytest.importorskip("xgboost", reason="xgboost not installed")


def _df():
    return pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0, 1, 0, 1]})


def _fake_progbar():
    return type(
        "PB",
        (),
        {
            "start_subtask": lambda *a, **k: None,
            "update_subtask": lambda *a, **k: None,
            "remove": lambda *a, **k: None,
        },
    )()


# ---------------------------------------------------------------------------
# Backend unit tests
# ---------------------------------------------------------------------------


def test_make_regressor_returns_xgbregressor():
    b = XGBoostGBTBackend()
    m = b.make_regressor(loss="squared_error")
    assert m.__class__.__name__ == "XGBRegressor"


def test_make_classifier_returns_xgbclassifier():
    b = XGBoostGBTBackend()
    m = b.make_classifier(loss="log_loss")
    assert m.__class__.__name__ == "XGBClassifier"


def test_is_classifier_distinguishes_types():
    b = XGBoostGBTBackend()
    assert b.is_classifier(b.make_classifier(loss="log_loss"))
    assert not b.is_classifier(b.make_regressor(loss="squared_error"))


def test_is_classifier_works_with_name_matched_fakes():
    b = XGBoostGBTBackend()
    FakeCls = type("XGBClassifier", (), {})()
    FakeReg = type("XGBRegressor", (), {})()
    assert b.is_classifier(FakeCls)
    assert not b.is_classifier(FakeReg)


def test_suggest_hparams_calls_all_trial_params():
    b = XGBoostGBTBackend()
    recorded = {}

    class FakeTrial:
        def suggest_float(self, name, low, high, log=False):
            recorded[name] = low
            return low

        def suggest_int(self, name, low, high):
            recorded[name] = low
            return low

    returned = b.suggest_hparams(FakeTrial())

    # XGBoost-specific params must have been suggested (stored on backend)
    for xgb_param in (
        "min_child_weight", "gamma", "colsample_bytree",
        "colsample_bylevel", "reg_lambda", "reg_alpha",
    ):
        assert xgb_param in recorded, f"Expected {xgb_param} to be suggested"

    # Returned dict must contain exactly the GBTRegressor-compatible subset
    assert set(returned.keys()) == {"learning_rate", "n_estimators", "max_depth", "subsample"}


def test_suggest_hparams_stores_xgb_params_on_backend():
    b = XGBoostGBTBackend()

    class FakeTrial:
        def suggest_float(self, name, low, high, log=False):
            return low
        def suggest_int(self, name, low, high):
            return low

    b.suggest_hparams(FakeTrial())
    # Stored defaults are the lower bounds of each range
    assert b.min_child_weight == 1
    assert b.gamma == 0.0
    assert b.colsample_bytree == 0.5
    assert b.reg_alpha == 0.0


def test_apply_best_params_updates_backend_state():
    b = XGBoostGBTBackend()
    b.apply_best_params({
        "learning_rate": 0.05,
        "n_estimators": 200,
        "max_depth": 6,
        "subsample": 0.8,
        "min_child_weight": 3,
        "gamma": 1.0,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.6,
        "reg_lambda": 2.5,
        "reg_alpha": 0.1,
    })
    assert b.learning_rate == 0.05
    assert b.n_estimators == 200
    assert b.gamma == 1.0
    assert b.reg_lambda == 2.5


def test_build_regressor_args_returns_overlapping_subset():
    b = XGBoostGBTBackend()
    args = b.build_regressor_args({
        "learning_rate": 0.05,
        "n_estimators": 200,
        "max_depth": 6,
        "subsample": 0.8,
        "min_child_weight": 3,  # XGBoost-specific; must NOT appear in output
    })
    assert set(args.keys()) == {"learning_rate", "n_estimators", "max_depth", "subsample"}
    assert "min_child_weight" not in args


# ---------------------------------------------------------------------------
# GBTRegressor end-to-end with XGBoost backend
# ---------------------------------------------------------------------------


def test_fit_predict_score_with_xgb_backend(monkeypatch):
    monkeypatch.setattr(gbt_module, "ProgBar", lambda *_, **__: _fake_progbar())
    df = _df()
    model = GBTRegressor(
        random_state=0, n_estimators=5, prog_bar=False,
        backend=XGBoostGBTBackend(),
    )
    model.fit(df)

    # Per-target models must be the XGBoost types
    assert model.regressor["x"].__class__.__name__ == "XGBRegressor"
    assert model.regressor["y"].__class__.__name__ == "XGBClassifier"

    preds = model.predict(df)
    assert preds.shape[0] == len(df.columns)

    scores = model.score(df)
    assert scores.shape[0] == len(df.columns)
    assert np.all(scores <= 1.0)


# ---------------------------------------------------------------------------
# HPO (tune) end-to-end with XGBoost backend
# ---------------------------------------------------------------------------


def test_tune_end_to_end_with_xgb_backend(monkeypatch):
    monkeypatch.setattr(gbt_module.utils, "cast_categoricals_to_int", lambda X: X)

    class FakeTrial:
        def __init__(self):
            self.params = {}

        def suggest_float(self, name, low, high, log=False):
            self.params[name] = low
            return low

        def suggest_int(self, name, low, high):
            self.params[name] = low
            return low

        def report(self, *_, **__):
            pass

        def should_prune(self):
            return False

    class FakeFrozen:
        def __init__(self, params, value):
            self.params = params
            self.values = [value]

    FakeXGBReg = type(
        "XGBRegressor",
        (),
        {"predict": lambda self, X: [0.5] * len(X), "score": lambda self, X, y: 0.8},
    )
    FakeXGBCls = type(
        "XGBClassifier",
        (),
        {"predict": lambda self, X: [1] * len(X)},
    )

    def fake_fit(self, X):
        self.feature_names = list(X.columns)
        self.regressor = {
            self.feature_names[0]: FakeXGBReg(),
            self.feature_names[1]: FakeXGBCls(),
        }
        self.is_fitted_ = True
        return self

    monkeypatch.setattr(gbt_module.GBTRegressor, "fit", fake_fit)

    class FakeStudy:
        best_value = float("inf")

        def __init__(self):
            self.best_trials = []

        def optimize(self, objective, n_trials, show_progress_bar=None, callbacks=None, **kw):
            trial = FakeTrial()
            value = objective(trial)
            self.best_trials = [FakeFrozen(trial.params, value)]
            self.best_value = value

    monkeypatch.setattr(gbt_module.optuna, "create_study", lambda **kw: FakeStudy())
    monkeypatch.setattr(gbt_module.optuna.logging, "set_verbosity", lambda *_, **__: None)

    backend = XGBoostGBTBackend()
    df = _df()
    model = GBTRegressor(prog_bar=False, silent=True, backend=backend)
    args = model.tune(df, df, n_trials=1)

    assert "learning_rate" in args
    assert model.best_params
    # XGBoost-specific params must be in best_params
    assert "min_child_weight" in model.best_params
    assert "gamma" in model.best_params
    # build_regressor_args must NOT include xgb-specific params
    assert "min_child_weight" not in args


# ---------------------------------------------------------------------------
# SHAP compatibility
# ---------------------------------------------------------------------------


def test_shap_tree_explainer_works_with_xgb_backend(monkeypatch):
    """shap.TreeExplainer must accept XGBRegressor produced by the backend."""
    import shap

    monkeypatch.setattr(gbt_module, "ProgBar", lambda *_, **__: _fake_progbar())
    rng = np.random.default_rng(0)
    data = pd.DataFrame({
        "a": rng.standard_normal(40),
        "b": rng.standard_normal(40),
        "c": rng.standard_normal(40),
    })
    model = GBTRegressor(
        random_state=0, n_estimators=10, prog_bar=False,
        backend=XGBoostGBTBackend(),
    )
    model.fit(data)

    fitted_reg = model.regressor["a"]
    assert fitted_reg.__class__.__name__ == "XGBRegressor"

    background = data.drop("a", axis=1).values
    explainer = shap.TreeExplainer(fitted_reg, background)
    shap_vals = explainer.shap_values(background)
    assert shap_vals.shape == (len(data), data.shape[1] - 1)
