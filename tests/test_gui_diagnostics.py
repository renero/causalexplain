from types import SimpleNamespace

import pandas as pd

from causalexplain.gui.app import has_active_model
from causalexplain.gui.diagnostics import (
    bootstrap_details_for_regressor,
    normalize_rex_diagnostics,
    regression_errors_for_all_targets,
    shap_values_for_target,
)
from causalexplain.gui.rendering import (
    render_bootstrap_heatmap,
    render_regression_error_chart,
    render_shap_mean_chart,
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
    assert normalized["errors_long"]["regressor"].tolist() == ["nn", "gbt", "nn", "gbt"]
    assert normalized["shap_long"]["regressor"].tolist() == ["nn", "gbt", "nn", "gbt", "nn", "gbt", "nn", "gbt"]
    assert normalized["bootstrap_edges_long"]["regressor"].tolist() == ["nn", "nn", "nn", "gbt", "gbt", "gbt"]


def test_regression_errors_for_all_targets_keeps_all_variables() -> None:
    normalized = normalize_rex_diagnostics(_diagnostics_by_regressor())

    frame = regression_errors_for_all_targets(normalized["errors_long"])

    assert frame.head(4).to_dict("records") == [
        {"target": "A", "regressor": "nn", "error": 0.1},
        {"target": "A", "regressor": "gbt", "error": 0.05},
        {"target": "B", "regressor": "nn", "error": 0.2},
        {"target": "B", "regressor": "gbt", "error": 0.25},
    ]


def test_shap_values_for_target_filters_and_orders_predictors() -> None:
    normalized = normalize_rex_diagnostics(_diagnostics_by_regressor())

    frame = shap_values_for_target(normalized["shap_long"], "A")

    assert frame["target"].tolist() == ["A", "A", "A", "A"]
    assert frame["predictor"].tolist() == ["B", "B", "C", "C"]
    assert frame["regressor"].tolist() == ["nn", "gbt", "nn", "gbt"]


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


def test_render_regression_error_chart_uses_nn_then_gbt_and_expected_colors() -> None:
    normalized = normalize_rex_diagnostics(_diagnostics_by_regressor())
    frame = regression_errors_for_all_targets(normalized["errors_long"])

    options = render_regression_error_chart(frame)

    assert options["legend"]["data"] == ["NN", "GBT"]
    assert options["series"][0]["name"] == "NN"
    assert options["series"][0]["itemStyle"]["color"] == "#2563eb"
    assert options["series"][1]["name"] == "GBT"
    assert options["series"][1]["itemStyle"]["color"] == "#16a34a"


def test_render_shap_mean_chart_uses_nn_then_gbt_and_expected_colors() -> None:
    normalized = normalize_rex_diagnostics(_diagnostics_by_regressor())
    frame = shap_values_for_target(normalized["shap_long"], "A")

    options = render_shap_mean_chart(frame, "A")

    assert options["legend"]["data"] == ["NN", "GBT"]
    assert options["series"][0]["name"] == "NN"
    assert options["series"][0]["itemStyle"]["color"] == "#2563eb"
    assert options["series"][1]["name"] == "GBT"
    assert options["series"][1]["itemStyle"]["color"] == "#16a34a"


def test_render_bootstrap_heatmap_labels_axes_as_parent_and_child() -> None:
    normalized = normalize_rex_diagnostics(_diagnostics_by_regressor())
    matrix, _ = bootstrap_details_for_regressor(
        normalized["bootstrap_matrices"],
        normalized["bootstrap_edges_long"],
        "nn",
    )

    options = render_bootstrap_heatmap(matrix, "nn")

    assert options["xAxis"]["name"] == "Child"
    assert options["yAxis"]["name"] == "Parent"
    assert options["visualMap"]["show"] is False
    assert options["visualMap"]["calculable"] is False


def test_diagnostics_tab_uses_blank_placeholder_option_for_empty_selects() -> None:
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

    assert tab._EMPTY_OPTION == [""]


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


def test_has_active_model_checks_active_discoverer() -> None:
    assert has_active_model({"active_discoverer": object()}) is True
    assert has_active_model({"active_discoverer": None}) is False


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
