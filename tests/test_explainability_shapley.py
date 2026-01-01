import numpy as np
import pandas as pd
import pytest

from causalexplain.explainability import shapley


class DummyModels:
    def __init__(self):
        self.regressor = {"a": object(), "b": object()}


def test_predict_requires_fit():
    estimator = shapley.ShapEstimator(models=DummyModels(), mean_shap_percentile=0.0, prog_bar=False, verbose=False)
    estimator.is_fitted_ = False
    with pytest.raises(ValueError):
        estimator.predict(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))


def test_compute_scaled_shap_threshold():
    estimator = shapley.ShapEstimator(models=DummyModels(), mean_shap_percentile=0.5, prog_bar=False, verbose=False)
    estimator.all_mean_shap_values = np.array([0.0, 1.0, 3.0])
    estimator._compute_scaled_shap_threshold()
    assert estimator.mean_shap_threshold == pytest.approx(1.0)


@pytest.mark.internal
def test_run_selected_shap_explainer_invalid(monkeypatch):
    estimator = shapley.ShapEstimator(models=DummyModels(), explainer="invalid", prog_bar=False, verbose=False)
    with pytest.raises(ValueError):
        estimator._run_selected_shap_explainer("t", object(), np.zeros((1, 1)), np.zeros((1, 1)))


@pytest.mark.internal
def test_compute_discrepancy_outputs_dataclass():
    estimator = shapley.ShapEstimator(models=DummyModels(), prog_bar=False, verbose=False)
    result = estimator._compute_discrepancy(
        x=np.array([0.0, 1.0, 2.0, 3.0]),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
        s=np.array([[0.5], [1.0], [1.5], [2.0]]),
        target_name="t",
        parent_name="p",
    )
    assert isinstance(result, shapley.ShapDiscrepancy)
    assert 0.0 <= result.ks_pvalue <= 1.0


@pytest.mark.internal
def test_compute_discrepancy_handles_nan_shap_values():
    estimator = shapley.ShapEstimator(models=DummyModels(), prog_bar=False, verbose=False)
    result = estimator._compute_discrepancy(
        x=np.array([0.0, 1.0, 2.0, 3.0]),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
        s=np.array([[0.5], [np.nan], [1.5], [2.0]]),
        target_name="t",
        parent_name="p",
    )
    assert isinstance(result, shapley.ShapDiscrepancy)
    assert np.isfinite(result.shap_gof)


@pytest.mark.internal
def test_call_explainer_retries_without_additivity_check():
    estimator = shapley.ShapEstimator(models=DummyModels(), prog_bar=False, verbose=False)

    class DummyExplainer:
        def __call__(self, X, silent=True, check_additivity=True):
            if check_additivity:
                explainer_error = getattr(getattr(shapley.shap, "utils", None), "_exceptions", None)
                explainer_error_cls = getattr(explainer_error, "ExplainerError", RuntimeError)
                raise explainer_error_cls("additivity failed")
            return "ok"

    result = estimator._call_explainer(DummyExplainer(), np.zeros((2, 2)), silent=True)
    assert result == "ok"


@pytest.mark.internal
def test_compute_discrepancies_builds_matrix():
    estimator = shapley.ShapEstimator(models=DummyModels(), prog_bar=False, verbose=False)
    estimator.feature_names = ["a", "b"]
    estimator.shap_values = {
        "a": np.array([[1.0], [2.0], [3.0], [4.0]]),
        "b": np.array([[4.0], [3.0], [2.0], [1.0]]),
    }
    estimator.is_fitted_ = True
    data = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})

    discrepancies = estimator._compute_discrepancies(data)
    assert set(discrepancies.columns) == {"a", "b"}
    assert discrepancies.isna().sum().sum() == 0
