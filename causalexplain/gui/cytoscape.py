"""Cytoscape helpers for NiceGUI graph rendering."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

_CY_ASSETS_LOADED = False
_CY_REGISTERED_ROUTES: Set[str] = set()
_CY_ASSET_BASE_URL: Optional[str] = None


def _hex_from_rgb(r: int, g: int, b: int) -> str:
    """Convert RGB channel values to a hex color string."""
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp_channel(start: int, end: int, t: float) -> int:
    """Linearly interpolate a color channel at t in [0, 1]."""
    return int(round(start + (end - start) * t))


def _score_to_color(score: float, max_score: float) -> str:
    """Map a score to a green-yellow-red color scale."""
    if max_score <= 0:
        return "#ffffff"
    ratio = max(0.0, min(score / max_score, 1.0))
    green = (26, 150, 65)
    yellow = (255, 224, 102)
    red = (215, 48, 39)
    if ratio <= 0.5:
        t = ratio / 0.5
        r = _lerp_channel(green[0], yellow[0], t)
        g = _lerp_channel(green[1], yellow[1], t)
        b = _lerp_channel(green[2], yellow[2], t)
    else:
        t = (ratio - 0.5) / 0.5
        r = _lerp_channel(yellow[0], red[0], t)
        g = _lerp_channel(yellow[1], red[1], t)
        b = _lerp_channel(yellow[2], red[2], t)
    return _hex_from_rgb(r, g, b)


def _luminance_from_hex(hex_color: str) -> float:
    """Compute approximate luminance for a hex color."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return 0.0
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return 0.299 * r + 0.587 * g + 0.114 * b


def _build_cytoscape_elements(
    graph: nx.DiGraph,
    reference: Optional[nx.DiGraph],
    show_node_fill: bool,
    root_causes: Optional[List[str]] = None,
    positions: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Build Cytoscape elements from prediction and reference graphs."""
    elements: List[Dict[str, Any]] = []
    reference_provided = reference is not None
    ref_graph = reference if reference is not None else nx.DiGraph()
    node_ids = sorted(set(graph.nodes()) | set(ref_graph.nodes()))
    has_scores = show_node_fill and all(
        "regr_score" in (graph.nodes[node] if node in graph.nodes else {})
        for node in node_ids
    )
    max_score = 1.0
    if has_scores:
        scores = [float(graph.nodes[node]["regr_score"]) for node in node_ids]
        max_score = max(max(scores), 1.0)

    root_set = set(root_causes or [])
    for node in node_ids:
        node_id = str(node)
        label = node_id
        element: Dict[str, Any] = {"data": {"id": node_id, "label": label}}
        if positions and node_id in positions:
            element["position"] = positions[node_id]
        style: Dict[str, Any] = {}
        if has_scores:
            score = float(graph.nodes[node]["regr_score"])
            color = _score_to_color(score, max_score)
            style["background-color"] = color
            style["color"] = "black" if _luminance_from_hex(color) > 0.5 else "white"
        if node_id in root_set:
            style["border-width"] = 3
        if style:
            element["style"] = style
        elements.append(element)

    pred_edges = set(graph.edges())
    ref_edges = set(ref_graph.edges())

    def edge_id(prefix: str, src: Any, dst: Any) -> str:
        """Build a stable edge id for Cytoscape elements."""
        return f"{prefix}:{src}->{dst}"

    for src, dst in sorted(pred_edges):
        if not reference_provided:
            edge_class = "edge_predicted"
        elif (src, dst) in ref_edges:
            edge_class = "edge_true"
        elif (dst, src) in ref_edges:
            edge_class = "edge_reversed"
        else:
            edge_class = "edge_false_positive"
        elements.append(
            {
                "data": {
                    "id": edge_id(edge_class, src, dst),
                    "source": str(src),
                    "target": str(dst),
                },
                "classes": edge_class,
            }
        )

    if reference_provided:
        for src, dst in sorted(ref_edges):
            if (src, dst) in pred_edges or (dst, src) in pred_edges:
                continue
            elements.append(
                {
                    "data": {
                        "id": edge_id("edge_false_negative", src, dst),
                        "source": str(src),
                        "target": str(dst),
                    },
                    "classes": "edge_false_negative",
                }
            )

    return elements, has_scores


def _cytoscape_stylesheet() -> List[Dict[str, Any]]:
    """Return the default Cytoscape stylesheet."""
    return [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "width": 40,
                "height": 40,
                "background-color": "#ffffff",
                "border-color": "#000000",
                "border-width": 1,
                "font-family": "monospace",
                "font-weight": "bold",
                "font-size": "12px",
                "text-valign": "center",
                "text-halign": "center",
                "color": "#000000",
            },
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "arrow-scale": 0.7,
                "line-color": "#000000",
                "target-arrow-color": "#000000",
                "opacity": 0.8,
                "width": 1,
            },
        },
        {
            "selector": ".edge_predicted",
            "style": {
                "line-color": "#000000",
                "target-arrow-color": "#000000",
                "width": 1,
                "line-style": "solid",
                "opacity": 0.8,
            },
        },
        {
            "selector": ".edge_true",
            "style": {
                "line-color": "#008000",
                "target-arrow-color": "#008000",
                "width": 3,
                "line-style": "solid",
                "opacity": 1.0,
            },
        },
        {
            "selector": ".edge_reversed",
            "style": {
                "line-color": "#ffa500",
                "target-arrow-color": "#ffa500",
                "width": 2,
                "line-style": "dashed",
                "opacity": 0.8,
            },
        },
        {
            "selector": ".edge_false_positive",
            "style": {
                "line-color": "#ff0000",
                "target-arrow-color": "#ff0000",
                "width": 1,
                "line-style": "dashed",
                "line-dash-pattern": [6, 3, 1, 3],
                "opacity": 0.6,
            },
        },
        {
            "selector": ".edge_false_negative",
            "style": {
                "line-color": "#d3d3d3",
                "target-arrow-color": "#d3d3d3",
                "width": 1,
                "line-style": "dotted",
                "opacity": 0.5,
            },
        },
    ]


def _cytoscape_layout_config(
    engine: str,
    rank_dir: str,
    use_preset: bool,
) -> Dict[str, Any]:
    """Return a Cytoscape layout configuration dictionary."""
    if use_preset:
        return {"name": "preset"}
    if engine == "elk":
        direction_map = {
            "LR": "RIGHT",
            "RL": "LEFT",
            "TB": "DOWN",
            "BT": "UP",
        }
        return {
            "name": "elk",
            "elk": {
                "elk.direction": direction_map.get(rank_dir, "RIGHT"),
                "elk.layered.spacing.nodeNodeBetweenLayers": 60,
                "elk.spacing.nodeNode": 30,
            },
            "animate": False,
        }
    return {
        "name": "dagre",
        "rankDir": rank_dir,
        "nodeSep": 50,
        "rankSep": 80,
        "edgeSep": 12,
        "animate": False,
    }


def _ensure_cytoscape_assets() -> None:
    """Register Cytoscape JS assets with NiceGUI."""
    global _CY_ASSETS_LOADED, _CY_ASSET_BASE_URL
    if _CY_ASSETS_LOADED:
        return
    from nicegui import app, ui

    static_root = os.path.join(
        os.path.dirname(__file__), "static", "cytoscape"
    )
    static_url = "/_causalexplain/cytoscape"
    if os.path.isdir(static_root):
        app.add_static_files(static_url, static_root)
        _CY_ASSET_BASE_URL = static_url
        ui.add_head_html(
            f"""
            <script src="{static_url}/cytoscape.min.js"></script>
            <script src="{static_url}/dagre.min.js"></script>
            <script src="{static_url}/cytoscape-dagre.js"></script>
            <script src="{static_url}/elk.bundled.js"></script>
            <script src="{static_url}/cytoscape-elk.js"></script>
            """,
            shared=True,
        )
    else:
        ui.add_head_html(
            "<script>window.__cytoscape_missing_assets = true;</script>",
            shared=True,
        )
    _CY_ASSETS_LOADED = True


def ensure_cytoscape_assets() -> None:
    """Expose Cytoscape asset registration for GUI apps."""
    _ensure_cytoscape_assets()


def _cytoscape_init_script(
    container_id: str,
    spec: Dict[str, Any],
    position_endpoint: Optional[str],
    click_endpoint: Optional[str],
    edge_click_endpoint: Optional[str],
) -> str:
    """Generate the Cytoscape initialization script."""
    payload = json.dumps(spec)
    container_id_json = json.dumps(container_id)
    position_endpoint_json = json.dumps(position_endpoint or "")
    click_endpoint_json = json.dumps(click_endpoint or "")
    edge_click_endpoint_json = json.dumps(edge_click_endpoint or "")
    return f"""
    (() => {{
      const container = document.getElementById({container_id_json});
      if (!container) {{
        return;
      }}
      if (window.__cytoscape_missing_assets) {{
        container.innerHTML = "<div class='empty-panel'>Missing Cytoscape assets. Populate causalexplain/gui/static/cytoscape.</div>";
        return;
      }}
      const init = () => {{
        if (typeof cytoscape === "undefined") {{
          container.innerHTML = "<div class='empty-panel'>Cytoscape failed to load.</div>";
          return;
        }}
        if (container._cy) {{
          container._cy.destroy();
        }}
        if (window.cytoscapeDagre) {{
          cytoscape.use(window.cytoscapeDagre);
        }}
        if (window.cytoscapeElk) {{
          cytoscape.use(window.cytoscapeElk);
        }}
        const spec = {payload};
        const cy = cytoscape({{
          container: container,
          elements: spec.elements,
          style: spec.style,
          layout: spec.layout,
          minZoom: 0.2,
          maxZoom: 2,
          wheelSensitivity: 0.2,
        }});
        container._cy = cy;
        const positionEndpoint = {position_endpoint_json};
        if ({str(bool(position_endpoint)).lower()}) {{
          let saveTimer = null;
          cy.on("dragfree", "node", () => {{
            if (saveTimer) {{
              clearTimeout(saveTimer);
            }}
            saveTimer = setTimeout(() => {{
              const positions = {{}};
              cy.nodes().forEach((node) => {{
                const pos = node.position();
                positions[node.id()] = {{ x: pos.x, y: pos.y }};
              }});
              fetch(positionEndpoint, {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify({{ positions }})
              }});
            }}, 250);
          }});
        }}
        const clickEndpoint = {click_endpoint_json};
        if ({str(bool(click_endpoint)).lower()}) {{
          cy.on("tap", "node", (evt) => {{
            fetch(clickEndpoint, {{
              method: "POST",
              headers: {{ "Content-Type": "application/json" }},
              body: JSON.stringify({{ node_id: evt.target.id() }})
            }});
          }});
        }}
        const edgeClickEndpoint = {edge_click_endpoint_json};
        if ({str(bool(edge_click_endpoint)).lower()}) {{
          cy.on("tap", "edge", (evt) => {{
            const classes = evt.target.classes();
            fetch(edgeClickEndpoint, {{
              method: "POST",
              headers: {{ "Content-Type": "application/json" }},
              body: JSON.stringify({{
                edge_id: evt.target.id(),
                classes: classes,
              }})
            }});
          }});
        }}
      }};
      const spec = {payload};
      const assetBase = spec.asset_base || "";
      const scripts = assetBase
        ? [
            `${{assetBase}}/cytoscape.min.js`,
            `${{assetBase}}/dagre.min.js`,
            `${{assetBase}}/cytoscape-dagre.js`,
            `${{assetBase}}/elk.bundled.js`,
            `${{assetBase}}/cytoscape-elk.js`,
          ]
        : [];
      const loadScript = (src) => new Promise((resolve, reject) => {{
        if (document.querySelector(`script[src="${{src}}"]`)) {{
          resolve();
          return;
        }}
        const script = document.createElement("script");
        script.src = src;
        script.async = true;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error(`Failed to load ${{src}}`));
        document.head.appendChild(script);
      }});
      const ensure = async () => {{
        if (typeof cytoscape === "undefined" && scripts.length > 0) {{
          try {{
            for (const src of scripts) {{
              await loadScript(src);
            }}
          }} catch (err) {{
            container.innerHTML = "<div class='empty-panel'>Cytoscape failed to load.</div>";
            return;
          }}
        }}
        init();
      }};
      setTimeout(ensure, 0);
    }})();
    """


def _cytoscape_sanity_check() -> Dict[str, int]:
    """Run a basic sanity check of edge classification."""
    predicted = nx.DiGraph()
    predicted.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    reference = nx.DiGraph()
    reference.add_edges_from([("A", "B"), ("C", "B"), ("D", "A")])
    elements, _ = _build_cytoscape_elements(
        predicted,
        reference,
        show_node_fill=False,
    )
    counts = {
        "edge_true": 0,
        "edge_reversed": 0,
        "edge_false_positive": 0,
        "edge_false_negative": 0,
    }
    for element in elements:
        classes = element.get("classes")
        if classes in counts:
            counts[classes] += 1
    assert counts["edge_true"] == 1
    assert counts["edge_reversed"] == 1
    assert counts["edge_false_positive"] == 1
    assert counts["edge_false_negative"] == 1
    return counts
