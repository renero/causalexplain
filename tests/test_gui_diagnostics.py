from types import SimpleNamespace

import pandas as pd

from causalexplain.gui.diagnostics import (
    bootstrap_details_for_regressor,
    normalize_rex_diagnostics,
    regression_errors_for_target,
    shap_values_for_target,
)
from causalexplain.gui.tabs.diagnostics import DiagnosticsTab
from causalexplain.gui.tabs.load import LoadTab
from causalexplain.gui.tabs.train import TrainTab


def _diagnostics_by_regressor():
    return {
        "nn": {
            "regressor_errors": pd.DataFrame({
                "target": ["A", "B"],
                "error": [0.1, 0.2],
            }),
            "shap_mean_long": pd.DataFrame({
                "target": ["A", "A", "B", "B"],
                "predictor": ["B", "C", "A", "C"],
                "mean_shap": [0.7, 0.4, 0.3, 0.2],
            }),
            "bootstrap_matrix": pd.DataFrame(
                [[0.0, 0.6, 0.0], [0.0, 0.0, 0.5], [0.1, 0.0, 0.0]],
                index=["A", "B", "C"],
                columns=["A", "B", "C"],
            ),
            "bootstrap_edges": pd.DataFrame({
                "source": ["A", "B", "C"],
                "target": ["B", "C", "A"],
                "weight": [0.6, 0.5, 0.1],
            }),
        },
        "gbt": {
            "regressor_errors": pd.DataFrame({
                "target": ["A", "B"],
                "error": [0.05, 0.25],
            }),
            "shap_mean_long": pd.DataFrame({
                "target": ["A", "A", "B", "B"],
                "predictor": ["B", "C", "A", "C"],
                "mean_shap": [0.8, 0.6, 0.4, 0.1],
            }),
            "bootstrap_matrix": pd.DataFrame(
                [[0.0, 0.4, 0.0], [0.2, 0.0, 0.3], [0.0, 0.0, 0.0]],
                index=["A", "B", "C"],
                columns=["A", "B", "C"],
            ),
            "bootstrap_edges": pd.DataFrame({
                "source": ["A", "B", "B"],
                "target": ["B", "A", "C"],
                "weight": [0.4, 0.2, 0.3],
            }),
        },
    }


def test_normalize_rex_diagnostics_stacks_regressor_frames() -> None:
    normalized = normalize_rex_diagnostics(_diagnostics_by_regressor())

    assert normalized["feature_names"] == ["A", "B", "C"]
    assert normalized["regressors"] == ["nn", "gbt"]
    assert list(normalized["bootstrap_matrices"].keys()) == ["nn", "gbt"]
    assert set(normalized["errors_long"]["regressor"]) == {"nn", "gbt"}
    assert set(normalized["shap_long"]["regressor"]) == {"nn", "gbt"}
    assert set(normalized["bootstrap_edges_long"]["regressor"]) == {"nn", "gbt"}


def test_regression_errors_for_target_filters_target() -> None:
    normalized = normalize_rex_diagnostics(_diagnostics_by_regressor())

    frame = regression_errors_for_target(normalized["errors_long"], "A")

    assert frame.to_dict("records") == [
        {"target": "A", "regressor": "nn", "error": 0.1},
        {"target": "A", "regressor": "gbt", "error": 0.05},
    ]


def test_shap_values_for_target_filters_and_orders_predictors() -> None:
    normalized = normalize_rex_diagnostics(_diagnostics_by_regressor())

    frame = shap_values_for_target(normalized["shap_long"], "A")

    assert frame["target"].tolist() == ["A", "A", "A", "A"]
    assert frame["predictor"].tolist() == ["B", "B", "C", "C"]
    assert frame["regressor"].tolist() == ["gbt", "nn", "gbt", "nn"]


def test_bootstrap_details_for_regressor_returns_matrix_and_edges() -> None:
    normalized = normalize_rex_diagnostics(_diagnostics_by_regressor())

    matrix, edges = bootstrap_details_for_regressor(
        normalized["bootstrap_matrices"],
        normalized["bootstrap_edges_long"],
        "gbt",
    )

    assert float(matrix.loc["B", "A"]) == 0.2
    assert edges["regressor"].unique().tolist() == ["gbt"]
    assert edges["source"].tolist() == ["A", "B", "B"]


def test_diagnostics_tab_uses_none_for_empty_select_defaults() -> None:
    tab = DiagnosticsTab(
        ui=None,
        run=None,
        storage={},
        settings={
            "selected_target": "",
            "selected_regressor": None,
            "selected_source": "A",
            "selected_pair_target": "",
        },
        active_model_state={},
    )

    assert tab._select_setting_value("selected_target") is None
    assert tab._select_setting_value("selected_regressor") is None
    assert tab._select_setting_value("selected_source") == "A"


def test_diagnostics_tab_binds_render_refresh_with_change_fallback() -> None:
    calls = []

    class DummyElement:
        def on(self, event_name, handler):
            calls.append((event_name, handler))

    tab = DiagnosticsTab(
        ui=None,
        run=None,
        storage={},
        settings={},
        active_model_state={},
    )

    tab._bind_render_refresh(DummyElement())

    assert len(calls) == 1
    assert calls[0][0] == "change"


def test_diagnostics_tab_mutates_existing_chart_options() -> None:
    updated = []

    class DummyChart:
        def __init__(self):
            self._options = {"title": {"text": "old"}, "series": [1]}

        @property
        def options(self):
            return self._options

        def update(self):
            updated.append(True)

    tab = DiagnosticsTab(
        ui=None,
        run=None,
        storage={},
        settings={},
        active_model_state={},
    )
    tab.chart = DummyChart()

    tab._set_chart_options({"title": {"text": "new"}, "series": []})

    assert tab.chart.options == {"title": {"text": "new"}, "series": []}
    assert updated == [True]


def test_train_tab_publishes_active_model() -> None:
    published = []
    active_state = {
        "set_active_model": lambda discoverer, ref_graph, source: published.append(
            (discoverer, ref_graph, source)
        )
    }
    tab = TrainTab(None, None, {}, {}, "", active_state)
    discoverer = SimpleNamespace(estimator="rex")

    tab._publish_active_model(discoverer, "ref_graph", "trained model")

    assert published == [(discoverer, "ref_graph", "trained model")]


def test_load_tab_publishes_active_model() -> None:
    published = []
    active_state = {
        "set_active_model": lambda discoverer, ref_graph, source: published.append(
            (discoverer, ref_graph, source)
        )
    }
    tab = LoadTab(None, None, {}, {}, "", active_state)
    discoverer = SimpleNamespace(estimator="rex")

    tab._publish_active_model(discoverer, "ref_graph", "loaded model")

    assert published == [(discoverer, "ref_graph", "loaded model")]
