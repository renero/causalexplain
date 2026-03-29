"""Diagnostics tab for ReX per-regressor internals."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

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
from causalexplain.gui.ui_helpers import bind_setting


class DiagnosticsTab:
    """Build and manage the Diagnostics tab."""

    _VIEWS = [
        "Regression Errors",
        "SHAP Means",
        "Bootstrap Matrix",
    ]
    _EMPTY_OPTION = [""]

    def __init__(
        self,
        ui: Any,
        run: Any,
        storage: Any,
        settings: Dict[str, Any],
        active_model_state: Dict[str, Any],
    ) -> None:
        """Initialize the diagnostics tab with shared GUI state."""
        self.ui = ui
        self.run = run
        self.storage = storage
        self.settings = settings
        self.active_model_state = active_model_state
        self.state: Dict[str, Any] = {
            "cache_key": None,
            "normalized": None,
        }
        self.status_label: Optional[Any] = None
        self.view_select: Optional[Any] = None
        self.target_select: Optional[Any] = None
        self.regressor_select: Optional[Any] = None
        self.chart: Optional[Any] = None
        self.table: Optional[Any] = None

    def build(self) -> None:
        """Render the diagnostics tab."""
        self.active_model_state.setdefault("_listeners", []).append(
            self.refresh_from_state)

        with self.ui.element("div").classes("section-card w-full"):
            self.ui.label("Diagnostics").classes("section-title")
            self.status_label = self.ui.label(
                "Train or load a ReX model to inspect diagnostics."
            ).classes("subtle")
            with self.ui.element("div").classes("form-grid w-full"):
                self.view_select = self.ui.select(
                    self._VIEWS,
                    value=self.settings.get("view", self._VIEWS[0]),
                    label="Diagnostic view",
                    on_change=lambda _e: self._render_current_view(),
                ).classes("w-full")
                self.target_select = self.ui.select(
                    self._EMPTY_OPTION,
                    value="",
                    label="Target variable",
                    on_change=lambda _e: self._render_current_view(),
                ).classes("w-full")
                self.regressor_select = self.ui.select(
                    self._EMPTY_OPTION,
                    value="",
                    label="Regressor",
                    on_change=lambda _e: self._render_current_view(),
                ).classes("w-full")
            self.ui.button(
                "Refresh diagnostics",
                on_click=self.refresh_from_state,
            ).props("flat")

        with self.ui.element("div").classes("section-card w-full"):
            self.ui.label("Visualization").classes("section-title")
            self.chart = self.ui.echart(self._empty_chart(
                "No diagnostics available."
            )).classes("w-full")
            self.chart.style("height: 420px")

        with self.ui.element("div").classes("section-card w-full"):
            self.ui.label("Data").classes("section-title")
            self.table = self.ui.table(
                columns=[],
                rows=[],
                row_key="_row_id",
                pagination=10,
            ).classes("w-full")

        if self.view_select is not None:
            bind_setting(
                self.view_select,
                self.storage,
                "diagnostics_settings",
                self.settings,
                "view",
            )

        for widget, key in (
            (self.target_select, "selected_target"),
            (self.regressor_select, "selected_regressor"),
        ):
            if widget is None:
                continue
            bind_setting(
                widget,
                self.storage,
                "diagnostics_settings",
                self.settings,
                key,
            )

        self.refresh_from_state()

    @staticmethod
    def _empty_chart(title: str) -> Dict[str, Any]:
        """Build an empty-state chart configuration."""
        return {
            "title": {"text": title},
            "xAxis": {"type": "category", "data": []},
            "yAxis": {"type": "value"},
            "series": [],
        }

    def _set_status(self, message: str) -> None:
        """Update the status label text."""
        if self.status_label is None:
            return
        self.status_label.text = message
        self.status_label.update()

    def _set_chart_options(self, options: Dict[str, Any]) -> None:
        """Replace the current chart configuration."""
        if self.chart is None:
            return
        current_options = self.chart.options
        current_options.clear()
        current_options.update(options)
        self.chart.update()

    def _set_table_frame(self, frame: pd.DataFrame) -> None:
        """Render a dataframe in the diagnostics table."""
        if self.table is None:
            return
        render_frame = frame.copy()
        render_frame["_row_id"] = range(len(render_frame))
        self.table.columns = [
            {"name": col, "label": col.replace("_", " ").title(), "field": col}
            for col in render_frame.columns
            if col != "_row_id"
        ]
        self.table.rows = render_frame.to_dict("records")

    def _update_select_options(self, widget: Any, options: list[str], setting_key: str) -> None:
        """Set select options and clamp current value to the available set."""
        if widget is None:
            return
        widget.options = options
        current = getattr(widget, "value", None)
        if current not in options:
            current = options[0] if options else None
            widget.value = current
        self.settings[setting_key] = current or ""
        self.storage["diagnostics_settings"] = self.settings
        widget.update()

    def _apply_selector_state(self) -> None:
        """Refresh selector options and visibility based on diagnostics state."""
        normalized = self.state.get("normalized")
        feature_names = list(normalized["feature_names"]) if normalized else []
        regressors = list(normalized["regressors"]) if normalized else []

        self._update_select_options(
            self.target_select, feature_names, "selected_target")
        self._update_select_options(
            self.regressor_select, regressors, "selected_regressor")

        selected_view = self.view_select.value if self.view_select is not None else self._VIEWS[0]
        target_visible = selected_view == "SHAP Means"
        regressor_visible = selected_view == "Bootstrap Matrix"

        for widget, visible in (
            (self.target_select, target_visible),
            (self.regressor_select, regressor_visible),
        ):
            if widget is None:
                continue
            widget.visible = visible
            widget.update()

    def refresh_from_state(self) -> None:
        """Reload diagnostics from the active shared model state."""
        discoverer = self.active_model_state.get("active_discoverer")
        cache_key = (
            id(discoverer) if discoverer is not None else None,
            self.active_model_state.get("_version"),
        )
        if cache_key == self.state.get("cache_key"):
            self._render_current_view()
            return

        self.state["cache_key"] = cache_key
        self.state["normalized"] = None

        if discoverer is None:
            self._set_status("Train or load a ReX model to inspect diagnostics.")
            self._set_chart_options(self._empty_chart("No diagnostics available."))
            self._set_table_frame(pd.DataFrame())
            self._apply_selector_state()
            return

        estimator = self.active_model_state.get("active_estimator")
        if estimator != "rex":
            self._set_status("Diagnostics are currently available only for ReX models.")
            self._set_chart_options(self._empty_chart("Diagnostics unavailable for this estimator."))
            self._set_table_frame(pd.DataFrame())
            self._apply_selector_state()
            return

        try:
            diagnostics = discoverer.get_rex_diagnostics_by_regressor()
            self.state["normalized"] = normalize_rex_diagnostics(diagnostics)
        except Exception as exc:
            self._set_status(
                "Diagnostics are unavailable for this model. "
                f"{str(exc)}"
            )
            self._set_chart_options(self._empty_chart("Diagnostics unavailable."))
            self._set_table_frame(pd.DataFrame())
            self._apply_selector_state()
            return

        source = self.active_model_state.get("active_source") or "current model"
        self._set_status(f"Showing ReX diagnostics from the active {source}.")
        self._apply_selector_state()
        self._render_current_view()

    def _render_current_view(self) -> None:
        """Render the currently selected diagnostics view."""
        normalized = self.state.get("normalized")
        self._apply_selector_state()

        if normalized is None:
            return

        selected_view = self.view_select.value if self.view_select is not None else self._VIEWS[0]
        if selected_view == "Regression Errors":
            frame = regression_errors_for_all_targets(normalized["errors_long"])
            self._set_chart_options(render_regression_error_chart(frame))
            self._set_table_frame(frame)
            return

        if selected_view == "SHAP Means":
            target = self.target_select.value if self.target_select is not None else ""
            frame = shap_values_for_target(normalized["shap_long"], target)
            self._set_chart_options(render_shap_mean_chart(frame, target))
            self._set_table_frame(frame)
            return

        regressor = self.regressor_select.value if self.regressor_select is not None else ""
        matrix, edges = bootstrap_details_for_regressor(
            normalized["bootstrap_matrices"],
            normalized["bootstrap_edges_long"],
            regressor,
        )
        self._set_chart_options(render_bootstrap_heatmap(matrix, regressor))
        self._set_table_frame(edges)
