"""Diagnostics tab for ReX per-regressor internals."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import pandas as pd

from causalexplain.gui.diagnostics import (
    bootstrap_details_for_regressor,
    normalize_rex_diagnostics,
    regression_errors_for_target,
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
        self.pair_value_label: Optional[Any] = None
        self.view_select: Optional[Any] = None
        self.target_select: Optional[Any] = None
        self.regressor_select: Optional[Any] = None
        self.source_select: Optional[Any] = None
        self.pair_target_select: Optional[Any] = None
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
                ).classes("w-full")
                self.target_select = self.ui.select(
                    [],
                    value=self._select_setting_value("selected_target"),
                    label="Target variable",
                ).classes("w-full")
                self.regressor_select = self.ui.select(
                    [],
                    value=self._select_setting_value("selected_regressor"),
                    label="Regressor",
                ).classes("w-full")
                self.source_select = self.ui.select(
                    [],
                    value=self._select_setting_value("selected_source"),
                    label="Source variable",
                ).classes("w-full")
                self.pair_target_select = self.ui.select(
                    [],
                    value=self._select_setting_value("selected_pair_target"),
                    label="Pair target variable",
                ).classes("w-full")
            self.pair_value_label = self.ui.label("").classes("subtle")
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
            self._bind_render_refresh(self.view_select)

        for widget, key in (
            (self.target_select, "selected_target"),
            (self.regressor_select, "selected_regressor"),
            (self.source_select, "selected_source"),
            (self.pair_target_select, "selected_pair_target"),
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
            self._bind_render_refresh(widget)

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

    def _select_setting_value(self, key: str) -> Optional[str]:
        """Return a valid initial select value for dynamic choice widgets."""
        value = self.settings.get(key)
        if value in ("", None):
            return None
        return str(value)

    def _bind_render_refresh(self, element: Any) -> None:
        """Attach a change handler in a NiceGUI-version-compatible way."""
        handler: Callable[[Any], None] = lambda _e: self._render_current_view()
        if hasattr(element, "on_value_change"):
            element.on_value_change(handler)
            return
        element.on("change", handler)

    def _set_status(self, message: str) -> None:
        """Update the status label text."""
        if self.status_label is None:
            return
        self.status_label.text = message
        self.status_label.update()

    def _set_pair_value(self, message: str) -> None:
        """Update the pair-weight label."""
        if self.pair_value_label is None:
            return
        self.pair_value_label.text = message
        self.pair_value_label.update()

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
        self._update_select_options(
            self.source_select, feature_names, "selected_source")
        self._update_select_options(
            self.pair_target_select, feature_names, "selected_pair_target")

        selected_view = self.view_select.value if self.view_select is not None else self._VIEWS[0]
        target_visible = selected_view in {"Regression Errors", "SHAP Means"}
        regressor_visible = selected_view == "Bootstrap Matrix"
        pair_visible = selected_view == "Bootstrap Matrix"

        for widget, visible in (
            (self.target_select, target_visible),
            (self.regressor_select, regressor_visible),
            (self.source_select, pair_visible),
            (self.pair_target_select, pair_visible),
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
        self._set_pair_value("")

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
        self._set_pair_value("")

        if normalized is None:
            return

        selected_view = self.view_select.value if self.view_select is not None else self._VIEWS[0]
        if selected_view == "Regression Errors":
            target = self.target_select.value if self.target_select is not None else ""
            frame = regression_errors_for_target(normalized["errors_long"], target)
            self._set_chart_options(render_regression_error_chart(frame, target))
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

        source = self.source_select.value if self.source_select is not None else ""
        target = self.pair_target_select.value if self.pair_target_select is not None else ""
        if source and target and source in matrix.index and target in matrix.columns:
            weight = float(matrix.loc[source, target])
            self._set_pair_value(
                f"Selected edge weight for {source} -> {target}: {weight:.4f}"
            )
