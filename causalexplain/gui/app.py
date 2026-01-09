"""NiceGUI app for local causalexplain workflows."""
from __future__ import annotations

import asyncio
import contextlib
import io
import os
import time
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pydot

from causalexplain.causalexplainer import GraphDiscovery, ensure_cytoscape_assets
from causalexplain.generators.generators import AcyclicGraphGenerator
from causalexplain.common import (
    DEFAULT_BOOTSTRAP_TOLERANCE,
    DEFAULT_BOOTSTRAP_TRIALS,
    DEFAULT_HPO_TRIALS,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_REGRESSORS,
    DEFAULT_SEED,
    SUPPORTED_METHODS,
    utils,
)
from causalexplain.metrics.compare_graphs import evaluate_graph


def _default_train_settings() -> Dict[str, Any]:
    return {
        "dataset_path": "",
        "true_dag_path": "",
        "prior_path": "",
        "method": "rex",
        "hpo_iterations": DEFAULT_HPO_TRIALS,
        "bootstrap_iterations": DEFAULT_BOOTSTRAP_TRIALS,
        "bootstrap_tolerance": DEFAULT_BOOTSTRAP_TOLERANCE,
        "combine_op": "union",
        "device": "cpu",
        "parallel_jobs": 0,
        "bootstrap_parallel_jobs": 0,
        "adaptive_shap_sampling": True,
        "max_shap_samples": DEFAULT_MAX_SAMPLES,
        "seed": DEFAULT_SEED,
        "regressors": DEFAULT_REGRESSORS[:],
        "explainer": "gradient",
        "corr_method": "spearman",
        "corr_alpha": 0.6,
        "corr_clusters": 15,
        "condlen": 1,
        "condsize": 0,
        "mean_pi_percentile": 0.8,
        "discrepancy_threshold": 0.99,
        "bootstrap_sampling_split": "auto",
        "save_model_path": "",
        "output_dag_path": "",
    }


def _default_load_settings() -> Dict[str, Any]:
    return {
        "model_path": "",
        "true_dag_path": "",
    }


def _default_generate_settings() -> Dict[str, Any]:
    return {
        "mechanism": "linear",
        "nodes": 10,
        "samples": 500,
        "max_parents": 3,
        "seed": DEFAULT_SEED,
        "rescale": True,
        "output_base": "",
    }


def _merge_settings(stored: Any, defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = defaults.copy()
    if isinstance(stored, dict):
        for key, value in stored.items():
            merged[key] = value
    return merged


def _clean_node_name(name: Any) -> str:
    text = str(name).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text


def _normalize_graph(graph: nx.Graph) -> nx.DiGraph:
    cleaned = nx.DiGraph()
    for node in graph.nodes:
        cleaned.add_node(_clean_node_name(node))
    for edge in graph.edges:
        if not isinstance(edge, (tuple, list)) or len(edge) < 2:
            continue
        src, dst = edge[0], edge[1]
        cleaned.add_edge(_clean_node_name(src), _clean_node_name(dst))
    if cleaned.has_node("\\n"):
        cleaned.remove_node("\\n")
    return cleaned


def _graph_from_dot(path: str) -> Optional[nx.DiGraph]:
    if not path:
        return None
    graph = nx.drawing.nx_pydot.read_dot(path)
    return _normalize_graph(graph)


def _graph_to_svg(graph: Optional[nx.Graph], title: Optional[str] = None) -> str:
    if graph is None or len(graph.nodes) == 0:
        return "<div class='empty-panel'>No graph available.</div>"
    normalized = _normalize_graph(graph)
    try:
        dot = nx.nx_pydot.to_pydot(normalized)
        dot.set("rankdir", "LR")
        if title:
            dot.set("labelloc", "t")
            dot.set("label", title)
        svg_bytes = dot.create_svg()
        return svg_bytes.decode("utf-8")
    except Exception as exc:  # pragma: no cover - best effort rendering
        return (
            "<div class='empty-panel'>Graph render failed: "
            f"{str(exc)}</div>"
        )


def _plot_svg(
    discoverer: GraphDiscovery,
    title: Optional[str] = None,
    use_reference: bool = True,
    reference_graph: Optional[nx.Graph] = None,
) -> str:
    model = discoverer.model
    original_ref = getattr(model, "ref_graph", None)
    if reference_graph is not None:
        model.ref_graph = reference_graph
    elif not use_reference:
        model.ref_graph = None
    try:
        fig = plt.figure(figsize=(6, 4), dpi=96)
        ax = fig.add_subplot(1, 1, 1)
        discoverer.plot(show_metrics=False, title=title or "", ax=ax)
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue().decode("utf-8")
    except Exception as exc:  # pragma: no cover - best effort rendering
        return (
            "<div class='empty-panel'>Graph render failed: "
            f"{str(exc)}</div>"
        )
    finally:
        model.ref_graph = original_ref


def _overlay_svg(
    predicted: Optional[nx.Graph],
    truth: Optional[nx.Graph]
) -> str:
    if predicted is None or truth is None:
        return "<div class='empty-panel'>Overlay requires two graphs.</div>"
    pred_graph = _normalize_graph(predicted)
    truth_graph = _normalize_graph(truth)
    try:
        dot = pydot.Dot(graph_type="digraph", strict=True, rankdir="LR")
        node_names = sorted(set(pred_graph.nodes) | set(truth_graph.nodes))
        for node in node_names:
            dot.add_node(pydot.Node(node))
        pred_edges = set(pred_graph.edges)
        truth_edges = set(truth_graph.edges)
        all_edges = pred_edges | truth_edges
        for src, dst in sorted(all_edges):
            if (src, dst) in pred_edges and (src, dst) in truth_edges:
                color = "seagreen"
                style = "solid"
            elif (src, dst) in pred_edges:
                color = "indianred"
                style = "solid"
            else:
                color = "goldenrod"
                style = "dashed"
            dot.add_edge(pydot.Edge(src, dst, color=color, style=style))
        svg_bytes = dot.create_svg()
        return svg_bytes.decode("utf-8")
    except Exception as exc:  # pragma: no cover - best effort rendering
        return (
            "<div class='empty-panel'>Overlay render failed: "
            f"{str(exc)}</div>"
        )


def _metrics_rows(metrics: Any) -> List[Dict[str, str]]:
    if metrics is None:
        return []
    data = metrics.to_dict() if hasattr(metrics, "to_dict") else metrics
    order = [
        ("Tp", "TP"),
        ("Tn", "TN"),
        ("Fp", "FP"),
        ("Fn", "FN"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("aupr", "auPR"),
        ("shd", "SHD"),
        ("sid", "SID"),
    ]
    rows = []
    for key, label in order:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, float):
            formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        rows.append({"metric": label, "value": formatted})
    return rows


def run_gui(host: str = "127.0.0.1", port: int = 8080) -> None:
    try:
        from nicegui import app, run, ui
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "NiceGUI is required for the GUI. Install it with: pip install nicegui"
        ) from exc

    upload_dir = os.path.join(os.getcwd(), ".gui_uploads")
    os.makedirs(upload_dir, exist_ok=True)

    ui.add_head_html(
        """
        <style>
        :root {
          --bg: white;
          --surface: white;
          --surface-strong: white;
          --sidebar-bg: white;
          --text-primary: CanvasText;
          --text-secondary: color-mix(in srgb, CanvasText 60%, transparent);
          --separator: color-mix(in srgb, CanvasText 12%, transparent);
          --hover: color-mix(in srgb, CanvasText 8%, transparent);
          --selected: color-mix(in srgb, Highlight 35%, transparent);
          --control-bg: white;
          --control-ring: color-mix(in srgb, Highlight 55%, transparent);
        }

        body, .nicegui-app {
          font-family: -apple-system, "SF Pro Text", "SF Pro Display",
            "SF Pro", "Helvetica Neue", Helvetica, Arial, sans-serif;
          background: var(--bg);
          color: var(--text-primary);
        }

        .app-root {
          display: flex;
          height: 100vh;
          width: 100%;
        }

        .sidebar {
          width: 220px;
          flex-shrink: 0;
          border-right: 1px solid var(--separator);
          background: var(--sidebar-bg);
          display: flex;
          flex-direction: column;
          gap: 12px;
          padding: 16px 12px;
        }

        .material {
          backdrop-filter: blur(18px);
        }

        @media (prefers-reduced-transparency: reduce) {
          .material {
            backdrop-filter: none;
            background: var(--surface);
          }
        }

        .sidebar-header {
          font-size: 18px;
          font-weight: 700;
          letter-spacing: 0.2px;
        }

        .sidebar-header + .subtle {
          margin-top: -6px;
        }

        .sidebar-list {
          display: flex;
          flex-direction: column;
          gap: 10px;
          overflow-y: auto;
          padding-right: 4px;
        }

        .sidebar-item {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 10px 12px;
          border-radius: 12px;
          transition: background 160ms ease;
          cursor: pointer;
        }

        .sidebar-item:hover {
          background: var(--hover);
        }

        .sidebar-item.selected {
          background: var(--selected);
        }

        .sidebar-icon {
          width: 32px;
          height: 32px;
          border-radius: 10px;
          background: var(--surface-strong);
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: 600;
          font-size: 12px;
        }

        .sidebar-title {
          font-size: 14px;
          font-weight: 600;
        }

        .sidebar-subtitle {
          font-size: 12px;
          color: var(--text-secondary);
        }

        .main-panel {
          display: flex;
          flex-direction: column;
          flex: 1;
          min-width: 0;
        }

        .content-scroll {
          flex: 1;
          overflow-y: auto;
          padding: 0 24px 24px 24px;
        }

        .content {
          width: 100%;
          max-width: 1280px;
          margin: 0;
          padding: 0 0 32px;
          display: flex;
          flex-direction: column;
          gap: 18px;
          align-items: stretch;
        }

        .section-card {
          border-radius: 16px;
          border: 1px solid var(--separator);
          background: var(--surface);
          padding: 12px 14px;
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .section-title {
          font-size: 16px;
          font-weight: 600;
        }

        .subtle {
          font-size: 12px;
          color: var(--text-secondary);
        }

        .field-row {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 10px;
        }

        .train-grid {
          display: grid;
          grid-template-columns: minmax(320px, 1fr) minmax(0, 2fr);
          gap: 18px;
          align-items: stretch;
        }

        .span-full {
          grid-column: 1 / -1;
        }

        .form-grid {
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
          gap: 16px;
        }

        .form-col {
          display: flex;
          flex-direction: column;
          gap: 12px;
          min-width: 0;
        }

        .file-row {
          display: grid;
          grid-template-columns: 150px minmax(0, 1fr) 210px;
          gap: 10px;
          align-items: center;
        }

        .file-label {
          font-size: 12px;
          color: var(--text-secondary);
        }

        .file-input .q-field__control {
          min-height: 36px;
        }

        .upload-inline {
          width: 210px;
          min-height: 38px;
          max-height: 38px;
          border: none;
          background: transparent;
          box-shadow: none;
          overflow: hidden;
        }

        .upload-inline .q-uploader__header {
          min-height: 38px;
          padding: 0 14px;
          border-radius: 10px;
          background: color-mix(in srgb, Highlight 75%, Canvas 25%);
          color: white;
          width: 100%;
        }

        .upload-inline .q-uploader__header-content {
          justify-content: center;
          gap: 6px;
        }

        .upload-inline .q-uploader__title {
          font-size: 13px;
          font-weight: 600;
          white-space: nowrap;
        }

        .upload-inline .q-uploader__icon {
          display: none;
        }

        .upload-inline .q-uploader__subtitle,
        .upload-inline .q-uploader__progress {
          display: none;
        }

        .upload-inline .q-uploader__list {
          display: none;
        }

        @media (max-width: 1200px) {
          .train-grid {
            grid-template-columns: minmax(0, 1fr);
          }
        }

        @media (max-width: 1100px) {
          .form-grid {
            grid-template-columns: minmax(0, 1fr);
          }

          .file-row {
            grid-template-columns: minmax(0, 1fr);
          }
        }

        .action-row {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .chip {
          padding: 2px 10px;
          border-radius: 999px;
          background: var(--surface-strong);
          font-size: 12px;
        }

        .empty-panel {
          padding: 24px;
          border: 1px dashed var(--separator);
          border-radius: 12px;
          color: var(--text-secondary);
          text-align: center;
        }

        .dag-frame {
          border: 1px solid var(--separator);
          border-radius: 12px;
          padding: 8px;
          background: var(--surface);
          overflow: auto;
        }

        .mini-button {
          border-radius: 999px;
          padding: 6px 10px;
          border: 1px solid var(--separator);
          background: var(--surface);
        }

        .nicegui-log {
          background: var(--surface);
        }

        </style>
        """,
        shared=True,
    )

    ensure_cytoscape_assets()

    @ui.page("/")
    def main_page() -> None:
        storage = app.storage.user

        train_settings = _merge_settings(
            storage.get("train_settings"), _default_train_settings()
        )
        load_settings = _merge_settings(
            storage.get("load_settings"), _default_load_settings()
        )
        generate_settings = _merge_settings(
            storage.get("generate_settings"), _default_generate_settings()
        )
        selected_panel = "train"

        def update_settings(
            settings_key: str, target: Dict[str, Any], key: str, value: Any
        ) -> None:
            target[key] = value
            storage[settings_key] = target

        def bind_setting(
            element: Any,
            settings_key: str,
            settings_ref: Dict[str, Any],
            field: str,
        ) -> None:
            if hasattr(element, "on_value_change"):
                element.on_value_change(
                    lambda e, key=field: update_settings(
                        settings_key, settings_ref, key, e.value
                    )
                )
            else:
                element.on(
                    "change",
                    lambda e, key=field, el=element: update_settings(
                        settings_key,
                        settings_ref,
                        key,
                        getattr(e, "value", getattr(el, "value", None)),
                    ),
                )

        async def save_upload(file_upload: Any, suffix: Optional[str] = None) -> str:
            filename = getattr(file_upload, "name", "upload")
            if suffix and not filename.lower().endswith(suffix):
                filename = f"{filename}{suffix}"
            timestamp = int(time.time() * 1000)
            base = os.path.basename(filename)
            path = os.path.join(upload_dir, f"{timestamp}_{base}")
            await file_upload.save(path)
            return path

        def set_input_value(input_el: Any, value: str) -> None:
            input_el.value = value
            input_el.update()

        def make_upload_handler(
            input_el: Any,
            settings_key: str,
            settings_ref: Dict[str, Any],
            field: str,
            suffix: Optional[str] = None,
            status_label: Optional[Any] = None,
        ):
            async def _handler(event: Any) -> None:
                file_upload = getattr(event, "file", None)
                if file_upload is None:
                    raise ValueError("Upload event missing file payload.")
                path = await save_upload(file_upload, suffix)
                set_input_value(input_el, path)
                update_settings(settings_key, settings_ref, field, path)
                if status_label is not None:
                    status_label.text = f"Loaded: {os.path.basename(path)}"
                    status_label.update()

            return _handler

        def ensure_file(path: str, suffixes: Any) -> str:
            if not path:
                raise ValueError("File path is required.")
            if suffixes:
                if isinstance(suffixes, str):
                    suffix_list = (suffixes,)
                else:
                    suffix_list = tuple(suffixes)
                if not any(path.lower().endswith(suffix) for suffix in suffix_list):
                    raise ValueError(
                        "Expected file extension: "
                        + ", ".join(suffix_list)
                    )
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")
            return path

        def ensure_output_dir(path: str) -> None:
            output_dir = os.path.dirname(path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

        with ui.element("div").classes("app-root"):
            with ui.element("div").classes("sidebar material"):
                ui.label("CausalExplain").classes("sidebar-header")
                ui.label("Choose a task").classes("subtle")

                panel_list = ui.column().classes("sidebar-list")
                panel_rows: Dict[str, Any] = {}

                def select_panel(panel_id: str) -> None:
                    nonlocal selected_panel
                    selected_panel = panel_id
                    for key, row in panel_rows.items():
                        if key == panel_id:
                            row.classes(add="selected")
                        else:
                            row.classes(remove="selected")
                    if panel_id == "train":
                        tabs.value = tab_train
                    elif panel_id == "load":
                        tabs.value = tab_load
                    else:
                        tabs.value = tab_generate
                    tabs.update()

                def add_panel_row(panel_id: str, icon: str, title: str, subtitle: str) -> None:
                    row = ui.element("div").classes("sidebar-item")
                    with row:
                        ui.label(icon).classes("sidebar-icon")
                        with ui.element("div"):
                            ui.label(title).classes("sidebar-title")
                            ui.label(subtitle).classes("sidebar-subtitle")
                    row.on("click", lambda _: select_panel(panel_id))
                    panel_rows[panel_id] = row

                with panel_list:
                    add_panel_row(
                        "train",
                        "T",
                        "Train Model",
                        "Fit a new causal graph",
                    )
                    add_panel_row(
                        "load",
                        "L",
                        "Load Model",
                        "Evaluate existing runs",
                    )
                    add_panel_row(
                        "generate",
                        "G",
                        "Generate Dataset",
                        "Create synthetic data",
                    )

            with ui.element("div").classes("main-panel"):
                with ui.element("div").classes("content-scroll"):
                    with ui.element("div").classes("content"):
                        tabs = ui.tabs().classes("text-sm")
                        tab_train = ui.tab("Train Model")
                        tab_load = ui.tab("Load + Evaluate")
                        tab_generate = ui.tab("Generate Dataset")

                        train_metrics_table = None
                        train_dag_html = None
                        train_overlay_container = None
                        train_log = None
                        train_progress = None
                        run_button = None
                        cancel_button = None
                        load_metrics_table = None
                        load_dag_html = None
                        load_overlay_html = None
                        generate_preview_table = None
                        generate_dag_html = None
                        generate_log = None

                        train_state: Dict[str, Any] = {
                            "task": None,
                            "running": False,
                        }

                        def update_metrics_table(table: Any, metrics: Any) -> None:
                            if table is None:
                                return
                            table.rows = _metrics_rows(metrics)
                            table.update()

                        def update_dag_view(
                            html_el: Any,
                            graph: Optional[nx.Graph],
                            svg: Optional[str] = None,
                        ) -> None:
                            if html_el is None:
                                return
                            if svg:
                                html_el.content = svg
                            else:
                                html_el.content = _graph_to_svg(graph)
                            html_el.update()

                        def update_overlay_view(
                            html_el: Any,
                            pred_graph: Optional[nx.Graph],
                            ref_graph: Optional[nx.Graph],
                            svg: Optional[str] = None,
                        ) -> None:
                            if html_el is None:
                                return
                            if svg:
                                html_el.content = svg
                            else:
                                html_el.content = _overlay_svg(pred_graph, ref_graph)
                            html_el.update()

                        async def run_training() -> None:
                            if train_state["running"]:
                                return

                            def _train_job(settings: Dict[str, Any]) -> Dict[str, Any]:
                                output: Dict[str, Any] = {}
                                buffer = io.StringIO()
                                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                                    dataset_path = ensure_file(settings["dataset_path"], ".csv")
                                    true_dag_path = settings.get("true_dag_path") or None
                                    if true_dag_path:
                                        true_dag_path = ensure_file(true_dag_path, ".dot")
                                    prior_path = settings.get("prior_path") or ""
                                    prior = None
                                    if prior_path:
                                        prior_path = ensure_file(prior_path, ".json")
                                        prior = utils.read_json_file(prior_path)
                                    dataset_name = os.path.splitext(
                                        os.path.basename(dataset_path)
                                    )[0]
                                    discoverer = GraphDiscovery(
                                        experiment_name=dataset_name,
                                        model_type=settings["method"],
                                        csv_filename=dataset_path,
                                        true_dag_filename=true_dag_path,
                                        verbose=False,
                                        seed=int(settings["seed"]),
                                        device=settings["device"],
                                        parallel_jobs=int(settings["parallel_jobs"]),
                                        bootstrap_parallel_jobs=int(
                                            settings["bootstrap_parallel_jobs"]
                                        ),
                                        max_shap_samples=int(settings["max_shap_samples"]),
                                    )
                                    if settings["method"] == "rex":
                                        regressors = settings.get("regressors", DEFAULT_REGRESSORS)
                                        if len(regressors) != 2:
                                            raise ValueError(
                                                "ReX requires exactly two regressors."
                                            )
                                        discoverer.regressors = regressors
                                    extra_kwargs: Dict[str, Any] = {}
                                    if settings["method"] == "rex":
                                        split = settings.get("bootstrap_sampling_split", "auto")
                                        if isinstance(split, str) and split.lower() != "auto":
                                            split = float(split)
                                        extra_kwargs = {
                                            "adaptive_shap_sampling": settings.get(
                                                "adaptive_shap_sampling", True
                                            ),
                                            "max_shap_samples": int(
                                                settings.get("max_shap_samples", DEFAULT_MAX_SAMPLES)
                                            ),
                                            "explainer": settings.get("explainer", "gradient"),
                                            "corr_method": settings.get("corr_method", "spearman"),
                                            "corr_alpha": float(settings.get("corr_alpha", 0.6)),
                                            "corr_clusters": int(settings.get("corr_clusters", 15)),
                                            "condlen": int(settings.get("condlen", 1)),
                                            "condsize": int(settings.get("condsize", 0)),
                                            "mean_pi_percentile": float(
                                                settings.get("mean_pi_percentile", 0.8)
                                            ),
                                            "discrepancy_threshold": float(
                                                settings.get("discrepancy_threshold", 0.99)
                                            ),
                                            "bootstrap_sampling_split": split,
                                        }
                                    start_time = time.time()
                                    discoverer.run(
                                        hpo_iterations=int(settings["hpo_iterations"]),
                                        bootstrap_iterations=int(
                                            settings["bootstrap_iterations"]
                                        ),
                                        prior=prior,
                                        bootstrap_tolerance=float(
                                            settings["bootstrap_tolerance"]
                                        ),
                                        combine_op=settings.get("combine_op", "union"),
                                        **extra_kwargs,
                                    )
                                    elapsed_seconds = time.time() - start_time
                                    ref_graph = None
                                    if true_dag_path:
                                        ref_graph = _graph_from_dot(true_dag_path)
                                    discoverer.ref_graph = ref_graph
                                    discoverer.model.ref_graph = ref_graph
                                    num_variables = 0
                                    if discoverer.data_columns:
                                        num_variables = len(discoverer.data_columns)
                                    output["dag"] = discoverer.dag
                                    output["discoverer"] = discoverer
                                    output["metrics"] = discoverer.metrics
                                    output["ref_graph"] = ref_graph
                                    output["elapsed_seconds"] = elapsed_seconds
                                    output["num_variables"] = num_variables
                                    output["dataset_name"] = dataset_name
                                    output["method"] = settings["method"]
                                    model_path = settings.get("save_model_path") or ""
                                    if model_path:
                                        ensure_output_dir(model_path)
                                        discoverer.save_model(model_path)
                                        output["model_path"] = model_path
                                    dag_path = settings.get("output_dag_path") or ""
                                    if dag_path:
                                        ensure_output_dir(dag_path)
                                        utils.graph_to_dot_file(discoverer.dag, dag_path)
                                        output["dag_path"] = dag_path
                                output["log"] = buffer.getvalue()
                                return output

                            train_state["running"] = True
                            if run_button is not None:
                                run_button.disable()
                            if cancel_button is not None:
                                cancel_button.enable()
                            if train_log is not None:
                                train_log.clear()
                                train_log.push("Starting training...")
                            if train_overlay_container is not None:
                                train_overlay_container.clear()
                            if train_progress is not None:
                                train_progress.props("indeterminate")
                                train_progress.update()
                            try:
                                result = await run.io_bound(
                                    _train_job, train_settings
                                )
                            except asyncio.CancelledError:
                                if train_log is not None:
                                    train_log.push("Training canceled.")
                                if train_progress is not None:
                                    train_progress.value = 0
                                    train_progress.props(remove="indeterminate")
                                    train_progress.update()
                                return
                            except Exception as exc:
                                if train_log is not None:
                                    train_log.push(f"Error: {str(exc)}")
                                if train_progress is not None:
                                    train_progress.value = 0
                                    train_progress.props(remove="indeterminate")
                                    train_progress.update()
                                return
                            finally:
                                train_state["running"] = False
                                train_state["task"] = None
                                if run_button is not None:
                                    run_button.enable()
                                if cancel_button is not None:
                                    cancel_button.disable()

                            if train_progress is not None:
                                train_progress.value = 0
                                train_progress.props(remove="indeterminate")
                                train_progress.update()
                            if train_log is not None:
                                log_text = result.get("log") or ""
                                if log_text:
                                    for line in log_text.splitlines():
                                        train_log.push(line)
                            discoverer = result.get("discoverer")
                            metrics = result.get("metrics")
                            ref_graph = result.get("ref_graph")
                            overlay_error = None
                            if train_overlay_container is not None:
                                train_overlay_container.clear()
                                if discoverer is not None:
                                    try:
                                        discoverer.plot_interactive(
                                            train_overlay_container,
                                            title=None,
                                            layout="dagre",
                                            rank_dir="TB",
                                            width="100%",
                                            height="420px",
                                            persist_positions=True,
                                        )
                                    except Exception as exc:
                                        overlay_error = exc
                            if train_log is not None:
                                num_variables = int(result.get("num_variables") or 0)
                                elapsed_seconds = float(
                                    result.get("elapsed_seconds") or 0.0
                                )
                                elapsed_minutes = elapsed_seconds / 60.0
                                train_log.push(
                                    "Training completed for "
                                    f"{num_variables} variables, in "
                                    f"{elapsed_minutes:.2f} minutes."
                                )
                                if overlay_error is not None:
                                    train_log.push(
                                        f"Overlay render failed: {str(overlay_error)}"
                                    )
                                if metrics is not None:
                                    train_log.push("\n\nEvaluation Metrics:")
                                    metrics_buffer = io.StringIO()
                                    with contextlib.redirect_stdout(metrics_buffer):
                                        print(metrics)
                                    for line in metrics_buffer.getvalue().splitlines():
                                        train_log.push(line)

                        def start_training_task() -> None:
                            if train_state.get("task") and not train_state["task"].done():
                                return
                            train_state["task"] = asyncio.create_task(run_training())

                        def cancel_training_task() -> None:
                            task = train_state.get("task")
                            if task and not task.done():
                                task.cancel()
                                if train_log is not None:
                                    train_log.push("Cancel requested.")

                        async def run_load() -> None:

                            def _load_job(settings: Dict[str, Any]) -> Dict[str, Any]:
                                model_path = ensure_file(
                                    settings["model_path"], (".pickle", ".pkl")
                                )
                                true_dag_path = settings.get("true_dag_path") or ""
                                ref_graph = None
                                if true_dag_path:
                                    true_dag_path = ensure_file(true_dag_path, ".dot")
                                    ref_graph = _graph_from_dot(true_dag_path)
                                discoverer = GraphDiscovery()
                                discoverer.load_model(model_path)
                                experiment = next(reversed(discoverer.trainer.values()))
                                dag = experiment.dag or discoverer.dag
                                metrics = experiment.metrics
                                if metrics is None and ref_graph is not None:
                                    data_cols = None
                                    if getattr(experiment, "data", None) is not None:
                                        data_cols = list(experiment.data.columns)
                                    metrics = evaluate_graph(
                                        ref_graph, dag, data_cols
                                    )
                                discoverer.ref_graph = ref_graph
                                discoverer.model.ref_graph = ref_graph
                                overlay_plot_svg = None
                                if ref_graph is not None:
                                    overlay_plot_svg = _plot_svg(
                                        discoverer,
                                        title="Overlay vs True DAG",
                                        use_reference=True,
                                        reference_graph=ref_graph,
                                    )
                                return {
                                    "dag": dag,
                                    "dag_plot_svg": _plot_svg(
                                        discoverer, use_reference=False
                                    ),
                                    "overlay_plot_svg": overlay_plot_svg,
                                    "metrics": metrics,
                                    "ref_graph": ref_graph,
                                    "model_name": os.path.basename(model_path),
                                }

                            try:
                                result = await run.io_bound(
                                    _load_job, load_settings
                                )
                            except Exception as exc:
                                ui.notify(str(exc), type="negative")
                                return

                            update_metrics_table(load_metrics_table, result.get("metrics"))
                            update_dag_view(
                                load_dag_html,
                                result.get("dag"),
                                result.get("dag_plot_svg"),
                            )
                            update_overlay_view(
                                load_overlay_html,
                                result.get("dag"),
                                result.get("ref_graph"),
                                result.get("overlay_plot_svg"),
                            )

                        async def run_generate() -> None:

                            def _generate_job(settings: Dict[str, Any]) -> Dict[str, Any]:
                                output_base = settings.get("output_base") or ""
                                if not output_base:
                                    raise ValueError("Output base path is required.")
                                output_dir = os.path.dirname(output_base)
                                if output_dir and not os.path.exists(output_dir):
                                    os.makedirs(output_dir, exist_ok=True)
                                np.random.seed(int(settings.get("seed", DEFAULT_SEED)))
                                generator = AcyclicGraphGenerator(
                                    settings["mechanism"],
                                    points=int(settings["samples"]),
                                    nodes=int(settings["nodes"]),
                                    parents_max=int(settings["max_parents"]),
                                    verbose=False,
                                )
                                graph, data = generator.generate(
                                    rescale=bool(settings["rescale"])
                                )
                                generator.to_csv(output_base, index=False)
                                return {
                                    "graph": graph,
                                    "data": data,
                                    "output_base": output_base,
                                }

                            if generate_log is not None:
                                generate_log.push("Generating dataset...")
                            try:
                                result = await run.io_bound(
                                    _generate_job, generate_settings
                                )
                            except Exception as exc:
                                if generate_log is not None:
                                    generate_log.push(f"Error: {str(exc)}")
                                return

                            data = result.get("data")
                            if isinstance(data, pd.DataFrame):
                                preview = data.head(8)
                                generate_preview_table.columns = [
                                    {"name": col, "label": col, "field": col}
                                    for col in preview.columns
                                ]
                                generate_preview_table.rows = preview.to_dict(
                                    orient="records"
                                )
                                generate_preview_table.update()
                            update_dag_view(generate_dag_html, result.get("graph"))
                            if generate_log is not None:
                                generate_log.push("Generation completed.")

                        with ui.tab_panels(tabs, value=tab_train).classes("w-full"):
                            with ui.tab_panel(tab_train):
                                with ui.element("div").classes("train-grid"):
                                    with ui.element("div").classes(
                                        "section-card span-full"
                                    ):
                                        ui.label("Inputs + Prior").classes("section-title")
                                        # ui.label(
                                        #     "Provide the dataset, true DAG, and prior."
                                        # ).classes("subtle")

                                        with ui.element("div").classes("file-row"):
                                            ui.label("Dataset CSV").classes("file-label")
                                            dataset_input = ui.input(
                                                value=train_settings.get(
                                                    "dataset_path", ""
                                                )
                                            ).props("dense").classes(
                                                "file-input w-full"
                                            )
                                            dataset_upload = ui.upload(
                                                label="Browse",
                                                on_upload=make_upload_handler(
                                                    dataset_input,
                                                    "train_settings",
                                                    train_settings,
                                                    "dataset_path",
                                                    ".csv",
                                                ),
                                                auto_upload=True,
                                            ).props("accept=.csv").classes("upload-inline")

                                        with ui.element("div").classes("file-row"):
                                            ui.label("True DAG").classes("file-label")
                                            dag_input = ui.input(
                                                value=train_settings.get(
                                                    "true_dag_path", ""
                                                )
                                            ).props("dense").classes(
                                                "file-input w-full"
                                            )
                                            dag_upload = ui.upload(
                                                label="Browse",
                                                on_upload=make_upload_handler(
                                                    dag_input,
                                                    "train_settings",
                                                    train_settings,
                                                    "true_dag_path",
                                                    ".dot",
                                                ),
                                                auto_upload=True,
                                            ).props("accept=.dot").classes("upload-inline")

                                        with ui.element("div").classes("file-row"):
                                            ui.label("Prior JSON").classes("file-label")
                                            prior_input = ui.input(
                                                value=train_settings.get(
                                                    "prior_path", ""
                                                )
                                            ).props("dense").classes(
                                                "file-input w-full"
                                            )
                                            prior_upload = ui.upload(
                                                label="Browse",
                                                on_upload=make_upload_handler(
                                                    prior_input,
                                                    "train_settings",
                                                    train_settings,
                                                    "prior_path",
                                                    ".json",
                                                ),
                                                auto_upload=True,
                                            ).props("accept=.json").classes("upload-inline")

                                        bind_setting(
                                            dataset_input,
                                            "train_settings",
                                            train_settings,
                                            "dataset_path",
                                        )
                                        bind_setting(
                                            dag_input,
                                            "train_settings",
                                            train_settings,
                                            "true_dag_path",
                                        )
                                        bind_setting(
                                            prior_input,
                                            "train_settings",
                                            train_settings,
                                            "prior_path",
                                        )

                                    with ui.element("div").classes("section-card"):
                                        ui.label(
                                            "Method + Core Settings"
                                        ).classes("section-title")
                                        with ui.element("div").classes("field-row"):
                                            method_select = ui.select(
                                                SUPPORTED_METHODS,
                                                value=train_settings.get(
                                                    "method", "rex"
                                                ),
                                                label="Method",
                                            )
                                            seed_input = ui.number(
                                                "Seed",
                                                value=train_settings.get(
                                                    "seed", DEFAULT_SEED
                                                ),
                                            ).props("dense")
                                            hpo_input = ui.number(
                                                "HPO iterations",
                                                value=train_settings.get(
                                                    "hpo_iterations", DEFAULT_HPO_TRIALS
                                                ),
                                            ).props("dense")
                                            bootstrap_input = ui.number(
                                                "Bootstrap iterations",
                                                value=train_settings.get(
                                                    "bootstrap_iterations",
                                                    DEFAULT_BOOTSTRAP_TRIALS,
                                                ),
                                            ).props("dense")
                                            tolerance_input = ui.number(
                                                "Bootstrap tolerance",
                                                value=train_settings.get(
                                                    "bootstrap_tolerance",
                                                    DEFAULT_BOOTSTRAP_TOLERANCE,
                                                ),
                                            ).props("dense")
                                            combine_select = ui.select(
                                                ["union", "intersection"],
                                                value=train_settings.get(
                                                    "combine_op", "union"
                                                ),
                                                label="Combine DAGs",
                                            )

                                        bind_setting(
                                            method_select,
                                            "train_settings",
                                            train_settings,
                                            "method",
                                        )
                                        bind_setting(
                                            seed_input,
                                            "train_settings",
                                            train_settings,
                                            "seed",
                                        )
                                        bind_setting(
                                            hpo_input,
                                            "train_settings",
                                            train_settings,
                                            "hpo_iterations",
                                        )
                                        bind_setting(
                                            bootstrap_input,
                                            "train_settings",
                                            train_settings,
                                            "bootstrap_iterations",
                                        )
                                        bind_setting(
                                            tolerance_input,
                                            "train_settings",
                                            train_settings,
                                            "bootstrap_tolerance",
                                        )
                                        bind_setting(
                                            combine_select,
                                            "train_settings",
                                            train_settings,
                                            "combine_op",
                                        )

                                    with ui.element("div").classes("section-card"):
                                        ui.label("Run").classes("section-title")
                                        with ui.element("div").classes("action-row"):
                                            run_button = ui.button(
                                                "Start training", on_click=start_training_task
                                            )
                                            cancel_button = ui.button(
                                                "Cancel", on_click=cancel_training_task
                                            ).props("flat")
                                            cancel_button.disable()
                                        train_progress = ui.linear_progress(
                                            value=0
                                        ).props("instant-feedback")
                                        train_log = ui.log(max_lines=200).classes("w-full")

                                    rex_section = ui.element("div").classes("section-card")
                                    with rex_section:
                                        ui.label("ReX Options").classes("section-title")
                                        with ui.element("div").classes("field-row"):
                                            device_select = ui.select(
                                                ["cpu", "cuda", "mps"],
                                                value=train_settings.get(
                                                    "device", "cpu"
                                                ),
                                                label="Device",
                                            )
                                            parallel_jobs_input = ui.number(
                                                "Parallel jobs",
                                                value=train_settings.get(
                                                    "parallel_jobs", 0
                                                ),
                                            ).props("dense")
                                            bootstrap_jobs_input = ui.number(
                                                "Bootstrap parallel jobs",
                                                value=train_settings.get(
                                                    "bootstrap_parallel_jobs", 0
                                                ),
                                            ).props("dense")
                                            adaptive_switch = ui.switch(
                                                "Adaptive SHAP sampling",
                                                value=train_settings.get(
                                                    "adaptive_shap_sampling", True
                                                ),
                                            )
                                            max_shap_input = ui.number(
                                                "Max SHAP samples",
                                                value=train_settings.get(
                                                    "max_shap_samples",
                                                    DEFAULT_MAX_SAMPLES,
                                                ),
                                            ).props("dense")
                                            regressors_input = ui.select(
                                                ["nn", "gbt"],
                                                value=train_settings.get(
                                                    "regressors", DEFAULT_REGRESSORS
                                                ),
                                                label="Regressors",
                                                multiple=True,
                                            )

                                        bind_setting(
                                            device_select,
                                            "train_settings",
                                            train_settings,
                                            "device",
                                        )
                                        bind_setting(
                                            parallel_jobs_input,
                                            "train_settings",
                                            train_settings,
                                            "parallel_jobs",
                                        )
                                        bind_setting(
                                            bootstrap_jobs_input,
                                            "train_settings",
                                            train_settings,
                                            "bootstrap_parallel_jobs",
                                        )
                                        bind_setting(
                                            adaptive_switch,
                                            "train_settings",
                                            train_settings,
                                            "adaptive_shap_sampling",
                                        )
                                        bind_setting(
                                            max_shap_input,
                                            "train_settings",
                                            train_settings,
                                            "max_shap_samples",
                                        )
                                        bind_setting(
                                            regressors_input,
                                            "train_settings",
                                            train_settings,
                                            "regressors",
                                        )

                                        with ui.expansion(
                                            "Advanced ReX settings", value=False
                                        ):
                                            with ui.element("div").classes("field-row"):
                                                explainer_input = ui.select(
                                                    [
                                                        "gradient",
                                                        "explainer",
                                                        "kernel",
                                                        "tree",
                                                    ],
                                                    value=train_settings.get(
                                                        "explainer", "gradient"
                                                    ),
                                                    label="SHAP explainer backend",
                                                )
                                                corr_method_input = ui.select(
                                                    [
                                                        "spearman",
                                                        "pearson",
                                                        "kendall",
                                                        "mic",
                                                    ],
                                                    value=train_settings.get(
                                                        "corr_method", "spearman"
                                                    ),
                                                    label="Correlation method",
                                                )
                                                corr_alpha_input = ui.number(
                                                    "Correlation alpha",
                                                    value=train_settings.get(
                                                        "corr_alpha", 0.6
                                                    ),
                                                )
                                                corr_clusters_input = ui.number(
                                                    "Correlation clusters",
                                                    value=train_settings.get(
                                                        "corr_clusters", 15
                                                    ),
                                                )
                                                condlen_input = ui.number(
                                                    "Condlen",
                                                    value=train_settings.get(
                                                        "condlen", 1
                                                    ),
                                                )
                                                condsize_input = ui.number(
                                                    "Condsize",
                                                    value=train_settings.get(
                                                        "condsize", 0
                                                    ),
                                                )
                                                mean_pi_input = ui.number(
                                                    "Mean PI percentile",
                                                    value=train_settings.get(
                                                        "mean_pi_percentile",
                                                        0.8,
                                                    ),
                                                )
                                                discrepancy_input = ui.number(
                                                    "Discrepancy threshold",
                                                    value=train_settings.get(
                                                        "discrepancy_threshold",
                                                        0.99,
                                                    ),
                                                )
                                                sampling_input = ui.input(
                                                    "Bootstrap sampling split",
                                                    value=train_settings.get(
                                                        "bootstrap_sampling_split",
                                                        "auto",
                                                    ),
                                                ).props("dense")

                                            bind_setting(
                                                explainer_input,
                                                "train_settings",
                                                train_settings,
                                                "explainer",
                                            )
                                            bind_setting(
                                                corr_method_input,
                                                "train_settings",
                                                train_settings,
                                                "corr_method",
                                            )
                                            bind_setting(
                                                corr_alpha_input,
                                                "train_settings",
                                                train_settings,
                                                "corr_alpha",
                                            )
                                            bind_setting(
                                                corr_clusters_input,
                                                "train_settings",
                                                train_settings,
                                                "corr_clusters",
                                            )
                                            bind_setting(
                                                condlen_input,
                                                "train_settings",
                                                train_settings,
                                                "condlen",
                                            )
                                            bind_setting(
                                                condsize_input,
                                                "train_settings",
                                                train_settings,
                                                "condsize",
                                            )
                                            bind_setting(
                                                mean_pi_input,
                                                "train_settings",
                                                train_settings,
                                                "mean_pi_percentile",
                                            )
                                            bind_setting(
                                                discrepancy_input,
                                                "train_settings",
                                                train_settings,
                                                "discrepancy_threshold",
                                            )
                                            bind_setting(
                                                sampling_input,
                                                "train_settings",
                                                train_settings,
                                                "bootstrap_sampling_split",
                                            )

                                    with ui.element("div").classes("section-card"):
                                        ui.label("Overlay vs True DAG").classes("subtle")
                                        train_overlay_container = ui.element(
                                            "div"
                                        ).classes("dag-frame")

                                    with ui.element("div").classes("section-card"):
                                        ui.label("Outputs").classes("section-title")
                                        with ui.element("div").classes("field-row"):
                                            model_output_input = ui.input(
                                                "Model pickle path",
                                                value=train_settings.get(
                                                    "save_model_path", ""
                                                ),
                                            ).props("dense")
                                            dag_output_input = ui.input(
                                                "Output DAG (.dot) path",
                                                value=train_settings.get(
                                                    "output_dag_path", ""
                                                ),
                                            ).props("dense")

                                        bind_setting(
                                            model_output_input,
                                            "train_settings",
                                            train_settings,
                                            "save_model_path",
                                        )
                                        bind_setting(
                                            dag_output_input,
                                            "train_settings",
                                            train_settings,
                                            "output_dag_path",
                                        )

                            with ui.tab_panel(tab_load):
                                with ui.element("div").classes("section-card"):
                                    ui.label("Load Model").classes("section-title")
                                    with ui.element("div").classes("file-row"):
                                        ui.label("Model pickle").classes("file-label")
                                        model_load_input = ui.input(
                                            value=load_settings.get("model_path", ""),
                                        ).props("dense").classes("file-input w-full")
                                        model_load_upload = ui.upload(
                                            label="Browse",
                                            on_upload=make_upload_handler(
                                                model_load_input,
                                                "load_settings",
                                                load_settings,
                                                "model_path",
                                            ),
                                            auto_upload=True,
                                        ).props("accept=.pickle,.pkl").classes(
                                            "upload-inline"
                                        )
                                    with ui.element("div").classes("file-row"):
                                        ui.label("True DAG").classes("file-label")
                                        dag_load_input = ui.input(
                                            value=load_settings.get("true_dag_path", ""),
                                        ).props("dense").classes("file-input w-full")
                                        dag_load_upload = ui.upload(
                                            label="Browse",
                                            on_upload=make_upload_handler(
                                                dag_load_input,
                                                "load_settings",
                                                load_settings,
                                                "true_dag_path",
                                                ".dot",
                                            ),
                                            auto_upload=True,
                                        ).props("accept=.dot").classes(
                                            "upload-inline"
                                        )
                                    bind_setting(
                                        model_load_input,
                                        "load_settings",
                                        load_settings,
                                        "model_path",
                                    )
                                    bind_setting(
                                        dag_load_input,
                                        "load_settings",
                                        load_settings,
                                        "true_dag_path",
                                    )
                                    ui.button(
                                        "Load + Evaluate",
                                        on_click=lambda: asyncio.create_task(run_load()),
                                    )

                                with ui.element("div").classes("section-card"):
                                    ui.label("Metrics").classes("section-title")
                                    load_metrics_table = ui.table(
                                        columns=[
                                            {
                                                "name": "metric",
                                                "label": "Metric",
                                                "field": "metric",
                                            },
                                            {
                                                "name": "value",
                                                "label": "Value",
                                                "field": "value",
                                            },
                                        ],
                                        rows=[],
                                        row_key="metric",
                                    ).classes("w-full")
                                    update_metrics_table(load_metrics_table, None)

                                with ui.element("div").classes("section-card"):
                                    ui.label("Graphs").classes("section-title")
                                    ui.label("Predicted DAG").classes("subtle")
                                    load_dag_html = ui.html(
                                        "", sanitize=False
                                    ).classes("dag-frame")
                                    update_dag_view(load_dag_html, None)
                                    ui.label("Overlay vs True DAG").classes("subtle")
                                    load_overlay_html = ui.html(
                                        "", sanitize=False
                                    ).classes("dag-frame")
                                    update_overlay_view(load_overlay_html, None, None)

                                with ui.element("div").classes("section-card"):
                                    ui.label("Future Visualizations").classes("section-title")
                                    ui.label(
                                        "SHAP diagnostics and discrepancy matrices will appear here."
                                    ).classes("subtle")
                                    ui.label(
                                        "Reserved space for upcoming visualizations."
                                    ).classes("empty-panel")

                            with ui.tab_panel(tab_generate):
                                with ui.element("div").classes("section-card"):
                                    ui.label("Generate Dataset").classes("section-title")
                                    with ui.element("div").classes("field-row"):
                                        mechanism_select = ui.select(
                                            [
                                                "linear",
                                                "polynomial",
                                                "sigmoid_add",
                                                "sigmoid_mix",
                                                "gp_add",
                                                "gp_mix",
                                            ],
                                            value=generate_settings.get("mechanism", "linear"),
                                            label="Mechanism",
                                        )
                                        nodes_input = ui.number(
                                            "Variables",
                                            value=generate_settings.get("nodes", 10),
                                        ).props("dense")
                                        samples_input = ui.number(
                                            "Samples",
                                            value=generate_settings.get("samples", 500),
                                        ).props("dense")
                                        parents_input = ui.number(
                                            "Max parents",
                                            value=generate_settings.get("max_parents", 3),
                                        ).props("dense")
                                        gen_seed_input = ui.number(
                                            "Seed",
                                            value=generate_settings.get("seed", DEFAULT_SEED),
                                        ).props("dense")
                                        rescale_switch = ui.switch(
                                            "Rescale",
                                            value=generate_settings.get("rescale", True),
                                        )
                                        output_base_input = ui.input(
                                            "Output base path",
                                            value=generate_settings.get("output_base", ""),
                                        ).props("dense")

                                    bind_setting(
                                        mechanism_select,
                                        "generate_settings",
                                        generate_settings,
                                        "mechanism",
                                    )
                                    bind_setting(
                                        nodes_input,
                                        "generate_settings",
                                        generate_settings,
                                        "nodes",
                                    )
                                    bind_setting(
                                        samples_input,
                                        "generate_settings",
                                        generate_settings,
                                        "samples",
                                    )
                                    bind_setting(
                                        parents_input,
                                        "generate_settings",
                                        generate_settings,
                                        "max_parents",
                                    )
                                    bind_setting(
                                        gen_seed_input,
                                        "generate_settings",
                                        generate_settings,
                                        "seed",
                                    )
                                    bind_setting(
                                        rescale_switch,
                                        "generate_settings",
                                        generate_settings,
                                        "rescale",
                                    )
                                    bind_setting(
                                        output_base_input,
                                        "generate_settings",
                                        generate_settings,
                                        "output_base",
                                    )

                                    ui.button(
                                        "Generate",
                                        on_click=lambda: asyncio.create_task(run_generate()),
                                    )
                                    generate_log = ui.log(max_lines=120).classes("w-full")

                                with ui.element("div").classes("section-card"):
                                    ui.label("Preview").classes("section-title")
                                    generate_preview_table = ui.table(
                                        columns=[],
                                        rows=[],
                                    ).classes("w-full")
                                    ui.label("Generated DAG").classes("subtle")
                                    generate_dag_html = ui.html(
                                        "", sanitize=False
                                    ).classes("dag-frame")
                                    update_dag_view(generate_dag_html, None)

                def update_rex_visibility() -> None:
                    is_rex = train_settings.get("method") == "rex"
                    rex_section.visible = is_rex

                update_rex_visibility()
                method_select.on("change", lambda _: update_rex_visibility())
                select_panel("train")

    ui.run(
        host=host,
        port=port,
        title="causalexplain",
        reload=False,
        storage_secret="causalexplain-local-gui",
    )
