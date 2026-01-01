import warnings

import numpy as np

from causalexplain.explainability import shapley


def _patch_generic_explainer(monkeypatch):
    monkeypatch.setattr(
        shapley, "build_generic_explainer",
        lambda model, X_bg: "explainer")

    def _fake_compute_generic_shap(explainer, X_explain):
        X_arr = np.asarray(X_explain)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return np.zeros((X_arr.shape[0], X_arr.shape[1]))

    monkeypatch.setattr(
        shapley, "compute_generic_shap", _fake_compute_generic_shap)


def test_adaptive_sampling_disabled_warns_large(monkeypatch):
    _patch_generic_explainer(monkeypatch)
    X = np.zeros((2001, 3))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _, diagnostics = shapley.compute_shap_adaptive(
            X,
            model=object(),
            backend="explainer",
            adaptive_shap_sampling=False,
        )
    assert any(
        "Adaptive SHAP sampling is disabled" in str(item.message)
        for item in recorded)
    assert diagnostics.mode == "no_sampling"
    assert diagnostics.n_background == 2001
    assert diagnostics.K == 1
    assert diagnostics.stability.get("skipped") is True
    assert diagnostics.warnings


def test_adaptive_sampling_disabled_no_warning_small(monkeypatch):
    _patch_generic_explainer(monkeypatch)
    X = np.zeros((2000, 3))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _, diagnostics = shapley.compute_shap_adaptive(
            X,
            model=object(),
            backend="explainer",
            adaptive_shap_sampling=False,
        )
    assert recorded == []
    assert diagnostics.mode == "no_sampling"
    assert diagnostics.n_background == 2000
    assert diagnostics.K == 1
