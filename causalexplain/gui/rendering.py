"""Rendering helpers for graphs and metrics in the GUI."""

from __future__ import annotations

import contextlib
import io
import uuid
from typing import Any, Dict, List, Optional

import networkx as nx

from causalexplain.gui import cytoscape as cygui


def update_metrics_log(log_el: Any, metrics: Any) -> None:
    """Render metrics into a NiceGUI log element."""
    if log_el is None:
        return
    log_el.clear()
    if metrics is None:
        log_el.push("No metrics available.")
        return
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print(metrics)
    output = buffer.getvalue().strip()
    if not output:
        log_el.push("No metrics available.")
        return
    for line in output.splitlines():
        log_el.push(line)


def overlay_status_message(classes: List[str]) -> str:
    """Return the overlay status message based on edge classes."""
    message = "Edge selected"
    if "edge_true" in classes:
        message = "Correctly predicted edge"
    elif "edge_false_positive" in classes:
        message = "Incorrect prediction"
    elif "edge_reversed" in classes:
        message = "Predicted edge, but direction is reversed"
    elif "edge_false_negative" in classes:
        message = "Edge is missing from the predictions"
    return message


def render_cytoscape_overlay(
    container: Any,
    status_label: Optional[Any],
    discoverer: Any,
    ref_graph: Optional[nx.Graph],
    *,
    persist_positions: bool,
    height: str = "420px",
) -> Optional[Exception]:
    """Render an interactive overlay chart for predicted vs. reference DAGs."""
    if container is None:
        return None
    container.clear()
    if status_label is not None:
        status_label.text = ""
        status_label.update()
    if discoverer is None:
        from nicegui import ui

        with container:
            ui.label("No graph available.").classes("empty-panel")
        return None

    original_ref = discoverer.model.ref_graph

    def handle_edge_click(edge_id: str, classes: List[str]) -> None:
        """Handle edge click events from the Cytoscape overlay."""
        _ = edge_id
        if status_label is None:
            return
        status_label.text = overlay_status_message(classes)
        status_label.update()

    try:
        discoverer.model.ref_graph = ref_graph
        discoverer.plot_interactive(
            container,
            title=None,
            layout="dagre",
            rank_dir="TB",
            width="100%",
            height=height,
            persist_positions=persist_positions,
            on_edge_click=handle_edge_click,
        )
    except Exception as exc:
        from nicegui import ui

        with container:
            ui.label(f"Overlay render failed: {str(exc)}").classes("empty-panel")
        return exc
    finally:
        discoverer.model.ref_graph = original_ref
    return None


def render_cytoscape_graph(
    container: Any,
    graph: Optional[nx.Graph],
    *,
    height: str = "420px",
) -> None:
    """Render a DAG in a NiceGUI container using Cytoscape."""
    if container is None:
        return
    container.clear()
    if graph is None or len(graph.nodes) == 0:
        from nicegui import ui

        with container:
            ui.label("No graph available.").classes("empty-panel")
        return
    graph_id = uuid.uuid4().hex
    elements, _ = cygui._build_cytoscape_elements(
        graph, None, show_node_fill=False
    )
    layout_config = cygui._cytoscape_layout_config("dagre", "TB", False)
    spec: Dict[str, Any] = {
        "elements": elements,
        "style": cygui._cytoscape_stylesheet(),
        "layout": layout_config,
        "width": "100%",
        "height": height,
        "asset_base": cygui._CY_ASSET_BASE_URL or "",
    }
    container_id = f"cytoscape-{graph_id}"
    container_html = (
        f'<div id="{container_id}" style="width: 100%; height: {height};"></div>'
    )
    script = cygui._cytoscape_init_script(container_id, spec, None, None, None)
    from nicegui import ui

    with container:
        ui.html(container_html, sanitize=False)
        ui.run_javascript(script)
