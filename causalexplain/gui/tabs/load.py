"""Load/evaluate tab for the causalexplain GUI."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from causalexplain.causalexplainer import GraphDiscovery
from causalexplain.gui.graph_utils import graph_from_dot
from causalexplain.gui.io_utils import ensure_file
from causalexplain.gui.rendering import (
    render_cytoscape_overlay,
    update_metrics_log,
)
from causalexplain.gui.ui_helpers import bind_setting, make_upload_handler
from causalexplain.metrics.compare_graphs import evaluate_graph


class LoadTab:
    """Build and manage the Load + Evaluate tab."""

    def __init__(
        self,
        ui: Any,
        run: Any,
        storage: Any,
        settings: Dict[str, Any],
        upload_dir: str,
    ) -> None:
        """Initialize the load tab with shared GUI dependencies."""
        self.ui = ui
        self.run = run
        self.storage = storage
        self.settings = settings
        self.upload_dir = upload_dir
        self.load_metrics_log: Optional[Any] = None
        self.load_overlay_container: Optional[Any] = None
        self.load_overlay_status: Optional[Any] = None

    def build(self) -> None:
        """Render the Load + Evaluate tab UI."""
        with self.ui.element("div").classes("section-card w-full"):
            self.ui.label("Load Model").classes("section-title")
            with self.ui.element("div").classes("file-row"):
                self.ui.label("Model pickle").classes("file-label")
                model_load_input = self.ui.input(
                    value=self.settings.get("model_path", "")
                ).props("dense").classes("file-input w-full")
                self.ui.upload(
                    label="Browse",
                    on_upload=make_upload_handler(
                        model_load_input,
                        self.storage,
                        "load_settings",
                        self.settings,
                        "model_path",
                        self.upload_dir,
                    ),
                    auto_upload=True,
                ).props("accept=.pickle,.pkl").classes("upload-inline")
            with self.ui.element("div").classes("file-row"):
                self.ui.label("True DAG").classes("file-label")
                dag_load_input = self.ui.input(
                    value=self.settings.get("true_dag_path", "")
                ).props("dense").classes("file-input w-full")
                self.ui.upload(
                    label="Browse",
                    on_upload=make_upload_handler(
                        dag_load_input,
                        self.storage,
                        "load_settings",
                        self.settings,
                        "true_dag_path",
                        self.upload_dir,
                        ".dot",
                    ),
                    auto_upload=True,
                ).props("accept=.dot").classes("upload-inline")

            bind_setting(
                model_load_input,
                self.storage,
                "load_settings",
                self.settings,
                "model_path",
            )
            bind_setting(
                dag_load_input,
                self.storage,
                "load_settings",
                self.settings,
                "true_dag_path",
            )
            self.ui.button(
                "Load + Evaluate",
                on_click=lambda: asyncio.create_task(self.run_load()),
            )

        with self.ui.element("div").classes("load-grid w-full"):
            with self.ui.element("div").classes("section-card"):
                self.ui.label("Metrics").classes("section-title")
                self.load_metrics_log = self.ui.log(
                    max_lines=120
                ).classes("w-full train-log")
                update_metrics_log(self.load_metrics_log, None)

            with self.ui.element("div").classes("section-card"):
                self.ui.label("Graphs").classes("section-title")
                self.ui.label("Overlay vs True DAG").classes("subtle")
                self.load_overlay_container = self.ui.element("div").classes(
                    "dag-frame"
                )
                self.load_overlay_status = self.ui.label("").classes("subtle")

    def _load_job(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Load the model and compute metrics in a worker thread."""
        model_path = ensure_file(settings["model_path"], (".pickle", ".pkl"))
        true_dag_path = settings.get("true_dag_path") or ""
        ref_graph = None
        if true_dag_path:
            true_dag_path = ensure_file(true_dag_path, ".dot")
            ref_graph = graph_from_dot(true_dag_path)
        discoverer = GraphDiscovery()
        discoverer.load_model(model_path)
        experiment = next(reversed(discoverer.trainer.values()))
        dag = experiment.dag or discoverer.dag
        metrics = experiment.metrics
        if metrics is None and ref_graph is not None:
            data_cols = None
            if getattr(experiment, "data", None) is not None:
                data_cols = list(experiment.data.columns)
            metrics = evaluate_graph(ref_graph, dag, data_cols)
        discoverer.ref_graph = ref_graph
        discoverer.model.ref_graph = ref_graph
        return {
            "dag": dag,
            "metrics": metrics,
            "ref_graph": ref_graph,
            "discoverer": discoverer,
        }

    async def run_load(self) -> None:
        """Load a model and refresh the UI."""
        try:
            result = await self.run.io_bound(self._load_job, self.settings)
        except Exception as exc:
            self.ui.notify(str(exc), type="negative")
            return

        update_metrics_log(self.load_metrics_log, result.get("metrics"))
        discoverer = result.get("discoverer")
        ref_graph = result.get("ref_graph")
        render_cytoscape_overlay(
            self.load_overlay_container,
            self.load_overlay_status,
            discoverer,
            ref_graph,
            persist_positions=False,
        )
