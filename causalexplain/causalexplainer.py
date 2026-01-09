"""
This module contains the GraphDiscovery class which is responsible for
creating, fitting, and evaluating causal discovery experiments.
"""
import json
import os
import pickle
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

import pandas as pd
from matplotlib.axes import Axes
import networkx as nx

from causalexplain.common import (
    DEFAULT_MAX_SAMPLES,
    DEFAULT_REGRESSORS,
    utils,
)
from causalexplain.common import plot
from causalexplain.common.notebook import Experiment
from causalexplain.metrics.compare_graphs import Metrics, evaluate_graph


_CY_ASSETS_LOADED = False
_CY_REGISTERED_ROUTES: Set[str] = set()
_CY_ASSET_BASE_URL: Optional[str] = None


def _hex_from_rgb(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp_channel(start: int, end: int, t: float) -> int:
    return int(round(start + (end - start) * t))


def _score_to_color(score: float, max_score: float) -> str:
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
    global _CY_ASSETS_LOADED, _CY_ASSET_BASE_URL
    if _CY_ASSETS_LOADED:
        return
    from nicegui import app, ui

    static_root = os.path.join(
        os.path.dirname(__file__), "gui", "static", "cytoscape"
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


class GraphDiscovery:
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        model_type: str = 'rex',
        csv_filename: Optional[str] = None,
        true_dag_filename: Optional[str] = None,
        verbose: bool = False,
        seed: int = 42,
        device: Optional[str] = None,
        parallel_jobs: int = 0,
        bootstrap_parallel_jobs: int = 0,
        max_shap_samples: Optional[int] = None
    ) -> None:
        """
        Initialize a graph discovery workflow and optionally load dataset metadata.

        This constructor sets up the estimator, loads the CSV metadata, and
        prepares train/test splits when both an experiment name and CSV path
        are provided. If neither is provided, it leaves the instance in an
        empty state so it can be configured later.

        Args:
            experiment_name (str, optional): The name of the experiment.
            model_type (str, optional): The type of model to use. Valid options
                are: 'rex', 'pc', 'fci', 'ges', 'lingam', 'cam', 'notears'.
            csv_filename (str, optional): The filename of the CSV file containing
                the data.
            true_dag_filename (str, optional): The filename of the DOT file
                containing the true causal graph.
            verbose (bool, optional): Whether to print verbose output.
            seed (int, optional): The random seed for reproducibility.
            device (Optional[str], optional): Device selection for regressors.
            parallel_jobs (int, optional): Number of parallel jobs for CPU training.
            bootstrap_parallel_jobs (int, optional): Number of parallel jobs for bootstrap.
            max_shap_samples (Optional[int], optional): Cap for SHAP background samples.

        Returns:
            None: This method does not return a value.
        """
        self.trainer: Dict[str, Experiment] = {}
        normalized_experiment = self._normalize_optional_str(experiment_name)
        normalized_csv = self._normalize_optional_str(csv_filename)
        resolved_device = utils.resolve_device(device)
        resolved_max_shap = (
            max_shap_samples
            if isinstance(max_shap_samples, int) and max_shap_samples > 0
            else DEFAULT_MAX_SAMPLES
        )

        if normalized_experiment is None and normalized_csv is None:
            self._init_empty_state(
                seed, resolved_device, parallel_jobs, bootstrap_parallel_jobs,
                resolved_max_shap)
            return

        self._validate_experiment_inputs(normalized_experiment, normalized_csv)
        csv_filename = cast(str, normalized_csv)

        self.experiment_name = normalized_experiment
        self.estimator = model_type
        self.csv_filename = csv_filename
        self.dot_filename = true_dag_filename
        self.verbose = verbose
        self.seed = seed
        self.device = resolved_device
        self.parallel_jobs = parallel_jobs
        self.bootstrap_parallel_jobs = bootstrap_parallel_jobs
        self.max_shap_samples = resolved_max_shap
        self.train_size = 0.9
        self.random_state = seed

        self.dataset_path = os.path.dirname(csv_filename)
        self.output_path = os.getcwd()

        self.ref_graph = self._load_reference_graph(true_dag_filename)
        if true_dag_filename is not None and self.ref_graph is None:
            raise ValueError("True DAG could not be loaded from dot file")
        self.data, self.dataset_name, self.data_columns = self._load_dataset_metadata(
            csv_filename)
        self.train_idx, self.test_idx = self._build_split_indices(
            self.data, self.train_size, self.random_state)
        self._validate_dag_nodes(self.ref_graph, self.data_columns)
        self.regressors = self._select_regressors()
        self._cy_positions: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _normalize_optional_str(value: Optional[str]) -> Optional[str]:
        """
        Normalize optional string inputs for consistent downstream checks.

        This strips surrounding whitespace and converts empty strings to None,
        which simplifies validation of optional parameters like filenames.

        Args:
            value (Optional[str]): The input string or None.

        Returns:
            Optional[str]: The normalized string or None if empty/blank.
        """
        if not isinstance(value, str):
            return value
        value = value.strip()
        return value or None

    def _init_empty_state(
        self,
        seed: int,
        device: str,
        parallel_jobs: int,
        bootstrap_parallel_jobs: int,
        max_shap_samples: int
    ) -> None:
        """
        Initialize the instance with placeholder state for deferred configuration.

        This is used when no experiment name or CSV file is provided so that the
        object can be configured later without partially initialized attributes.

        Args:
            seed (int): Random seed to store for later reproducibility.

        Returns:
            None: This method does not return a value.
        """
        self.experiment_name = None
        self.estimator = 'rex'
        self.csv_filename = None
        self.dot_filename = None
        self.data = None
        self.data_columns = None
        self.train_idx = None
        self.test_idx = None
        self.verbose = False
        self.seed = seed
        self.device = device
        self.parallel_jobs = parallel_jobs
        self.bootstrap_parallel_jobs = bootstrap_parallel_jobs
        self.max_shap_samples = max_shap_samples
        self.trainer: Dict[str, Experiment] = {}
        self._cy_positions: Dict[str, Dict[str, float]] = {}

    def _validate_experiment_inputs(
        self,
        experiment_name: Optional[str],
        csv_filename: Optional[str]
    ) -> None:
        """
        Ensure experiment and CSV inputs are provided together.

        This prevents a partially configured instance that would not be able
        to locate data or an experiment name.

        Args:
            experiment_name (Optional[str]): Name of the experiment.
            csv_filename (Optional[str]): Path to the CSV dataset.

        Returns:
            None: This method does not return a value.
        """
        if experiment_name is None or csv_filename is None:
            raise ValueError(
                "Both 'experiment_name' and 'csv_filename' must be provided together."
            )

    def _load_reference_graph(
        self,
        true_dag_filename: Optional[str]
    ) -> Optional[nx.DiGraph]:
        """
        Load the reference DAG from disk if a path is provided.

        This is used for evaluating discovered graphs against a ground-truth
        structure when a DOT file is available.

        Args:
            true_dag_filename (Optional[str]): Path to the DOT file or None.

        Returns:
            Optional[nx.DiGraph]: The loaded reference graph, or None if missing.
        """
        if true_dag_filename is None:
            return None
        return utils.graph_from_dot_file(true_dag_filename)

    def _load_dataset_metadata(
        self,
        csv_filename: str
    ) -> Tuple[pd.DataFrame, str, List[str]]:
        """
        Load dataset contents, dataset name, and validated column names.

        The CSV data is read, numeric columns are downcast for efficiency, and
        column names are validated for uniqueness and allowed characters.

        Args:
            csv_filename (str): Path to the input CSV file.

        Returns:
            Tuple[pd.DataFrame, str, List[str]]: The loaded data, dataset name,
                and validated list of column names.
        """
        if not os.path.exists(csv_filename):
            raise FileNotFoundError(f"Data file {csv_filename} not found")

        dataset_name = os.path.splitext(os.path.basename(csv_filename))[0]
        data = pd.read_csv(csv_filename)
        data = data.apply(pd.to_numeric, downcast='float')
        data_columns = list(data.columns)
        self._validate_column_names(data_columns)
        return data, dataset_name, data_columns

    @staticmethod
    def _build_split_indices(
        data: pd.DataFrame,
        train_size: float,
        random_state: int
    ) -> Tuple[pd.Index, pd.Index]:
        """
        Build deterministic train/test indices for the dataset.

        This uses a fixed random seed so experiments share the same split and
        remain comparable across different estimators.

        Args:
            data (pd.DataFrame): Dataset to split.
            train_size (float): Fraction of rows for training.
            random_state (int): Random seed for sampling.

        Returns:
            Tuple[pd.Index, pd.Index]: Train and test row indices.
        """
        # Share split indices across experiments to avoid repeated sampling.
        train_idx = data.sample(
            frac=train_size, random_state=random_state).index
        test_idx = data.index[~data.index.isin(train_idx)]
        return train_idx, test_idx

    @staticmethod
    def _validate_column_names(data_columns: List[Any]) -> None:
        """
        Validate dataset column names for uniqueness and allowed characters.

        Column names must be non-empty strings that start with a letter and
        contain only letters and numbers.

        Args:
            data_columns (List[Any]): Column names to validate.

        Returns:
            None: This method does not return a value.
        """
        if not data_columns:
            raise ValueError("Dataset must include at least one column")
        seen = set()
        invalid_columns = []
        duplicate_columns = set()
        for col in data_columns:
            if col in seen:
                duplicate_columns.add(col)
            else:
                seen.add(col)
            if not isinstance(col, str) or not col:
                invalid_columns.append(col)
                continue
            if not re.match(r"^[A-Za-z][A-Za-z0-9]*$", col):
                invalid_columns.append(col)
        if duplicate_columns:
            raise ValueError(
                "Dataset has duplicate column names: "
                + ", ".join(sorted(duplicate_columns))
            )
        if invalid_columns:
            raise ValueError(
                "Invalid column names (must start with a letter and contain only letters/numbers): "
                + ", ".join(map(str, invalid_columns))
            )

    @staticmethod
    def _validate_dag_nodes(
        ref_graph: Optional[nx.DiGraph],
        data_columns: Optional[List[Any]]
    ) -> None:
        """
        Ensure the reference DAG nodes align with dataset columns.

        This check catches mismatches between a provided ground-truth DAG and
        the dataset schema before any training or evaluation.

        Args:
            ref_graph (Optional[nx.DiGraph]): Reference DAG to validate.
            data_columns (Optional[List[Any]]): Dataset column names.

        Returns:
            None: This method does not return a value.
        """
        if ref_graph is None or data_columns is None:
            return
        dag_nodes = {str(node) for node in ref_graph.nodes}
        dataset_columns = {str(col) for col in data_columns}
        if dag_nodes != dataset_columns:
            missing = sorted(dataset_columns - dag_nodes)
            extra = sorted(dag_nodes - dataset_columns)
            details = []
            if missing:
                details.append(f"missing in DAG: {', '.join(missing)}")
            if extra:
                details.append(f"extra in DAG: {', '.join(extra)}")
            raise ValueError(
                "DAG nodes must match dataset columns exactly; "
                + "; ".join(details)
            )

    def _validate_prior_columns(
        self,
        prior: Optional[List[List[str]]]
    ) -> None:
        """
        Validate that the prior column names exist in the dataset.

        This ensures any user-supplied prior constraints reference valid
        dataset columns before the estimator is invoked.

        Args:
            prior (List[List[str]], optional): The prior to validate.

        Raises:
            ValueError: If any prior column names are invalid or not in the dataset.

        Returns:
            None: This method does not return a value.
        """
        if prior is None:
            return
        if self.data_columns is None:
            raise ValueError(
                "Dataset columns are not available; cannot validate prior."
            )
        if not isinstance(prior, list) or any(
            not isinstance(tier, list) for tier in prior
        ):
            raise ValueError("Prior must be a list of lists of column names.")
        invalid_names = []
        prior_names: List[str] = []
        for tier in prior:
            for name in tier:
                if not isinstance(name, str) or not name:
                    invalid_names.append(name)
                    continue
                prior_names.append(name)
        if invalid_names:
            raise ValueError(
                "Prior contains invalid names (must be non-empty strings): "
                + ", ".join(map(str, invalid_names))
            )
        data_columns = {str(col) for col in self.data_columns}
        extra = sorted(
            {name for name in prior_names if name not in data_columns})
        if extra:
            raise ValueError(
                "Prior includes variables not present in dataset columns: "
                + ", ".join(extra)
            )

    def _select_regressors(self) -> List[str]:
        """
        Select regressors based on the estimator type.

        ReX requires multiple regressors for ensembling, while other estimators
        run as a single model type.

        Args:
            None.

        Returns:
            List[str]: The list of regressor names to instantiate.
        """
        if self.estimator == 'rex':
            return DEFAULT_REGRESSORS
        return [self.estimator]

    def create_experiments(self) -> Dict[str, Experiment]:
        """
        Create an Experiment object for each regressor configured on the instance.

        This uses the dataset metadata and train/test indices prepared during
        initialization, and prepares the trainer map without fitting models.

        Args:
            None.

        Returns:
            Dict[str, Experiment]: A dictionary of Experiment objects keyed by
                trainer name.
        """
        if self.csv_filename is None:
            raise AttributeError(
                "CSV filename is required to create experiments.")

        csv_filename = cast(str, self.csv_filename)
        dot_filename = cast(str, self.dot_filename)
        self.trainer: Dict[str, Experiment] = {}
        for model_type in self.regressors:
            trainer_name = f"{self.dataset_name}_{model_type}"
            self.trainer[trainer_name] = Experiment(
                experiment_name=self.dataset_name,
                csv_filename=csv_filename,
                dot_filename=dot_filename,
                data=self.data,
                data_is_processed=True,
                train_idx=self.train_idx,
                test_idx=self.test_idx,
                model_type=model_type,
                input_path=self.dataset_path,
                output_path=self.output_path,
                verbose=False)

        return self.trainer

    def fit_experiments(
        self,
        hpo_iterations: Optional[int] = None,
        bootstrap_iterations: Optional[int] = None,
        prior: Optional[List[List[str]]] = None,
        bootstrap_tolerance: Optional[float] = None,
        quiet: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Fit the Experiment objects prepared by `create_experiments`.

        This configures estimator-specific options (ReX vs. other methods)
        and forwards any additional keyword arguments to `fit_predict`.

        Args:
            hpo_iterations (Optional[int]): Number of HPO trials for ReX.
            bootstrap_iterations (Optional[int]): Number of bootstrap trials
                for ReX.
            prior (Optional[List[List[str]]]): Optional prior constraints.
            bootstrap_tolerance (Optional[float]): Threshold for bootstrapped
                adjacency matrix filtering.
            quiet (bool): Disable verbose output and progress indicators.
            **kwargs: Additional keyword arguments forwarded to `fit_predict`.

        Returns:
            None: This method does not return a value.
        """
        self._validate_prior_columns(prior)
        verbose = False if quiet else self.verbose
        xargs: Dict[str, Any] = {}
        if self.estimator == 'rex':
            xargs = {
                'verbose': verbose,
                'prior': prior,
                'device': self.device,
                'parallel_jobs': self.parallel_jobs,
                'bootstrap_parallel_jobs': self.bootstrap_parallel_jobs
            }
            if self.max_shap_samples is not None:
                xargs['max_shap_samples'] = self.max_shap_samples
            if hpo_iterations is not None:
                xargs['hpo_n_trials'] = hpo_iterations
            if bootstrap_iterations is not None:
                xargs['bootstrap_trials'] = bootstrap_iterations
            if bootstrap_tolerance is not None:
                xargs['bootstrap_tolerance'] = bootstrap_tolerance
            if quiet:
                xargs['prog_bar'] = False
                xargs['silent'] = True
        else:
            xargs = {
                'verbose': verbose
            }

        # Combine the arguments
        xargs.update(kwargs)
        if quiet:
            xargs['verbose'] = False
            if self.estimator == 'rex':
                xargs['prog_bar'] = False
                xargs['silent'] = True

        for trainer_name, experiment in self.trainer.items():
            if not trainer_name.endswith("_rex"):
                experiment.fit_predict(estimator=self.estimator, **xargs)

    def combine_and_evaluate_dags(
        self,
        prior: Optional[List[List[str]]] = None,
        combine_op: str = 'union'
    ) -> Experiment:
        """
        Combine or select DAGs from experiments and compute evaluation metrics.

        For non-ReX estimators this simply selects the single model DAG. For ReX,
        it combines multiple DAGs (currently the first two) using the requested
        union or intersection operation before evaluation.

        Args:
            prior (List[List[str]], optional): The prior to use for ReX.
                Defaults to None.
            combine_op (str, optional): Operation used to combine DAGs in ReX.
                Supported values are: 'union' and 'intersection'.

        Returns:
            Experiment: The experiment object with the final DAG and metrics.
        """
        if self.estimator != 'rex':
            trainer_key = f"{self.dataset_name}_{self.estimator}"
            estimator_obj = getattr(self.trainer[trainer_key], self.estimator)
            self.trainer[trainer_key].dag = estimator_obj.dag
            if self.ref_graph is not None and self.data_columns is not None:
                self.trainer[trainer_key].metrics = evaluate_graph(
                    self.ref_graph, estimator_obj.dag, self.data_columns)
            else:
                self.trainer[trainer_key].metrics = None

            self.dag = self.trainer[trainer_key].dag
            self.metrics = self.trainer[trainer_key].metrics
            return self.trainer[trainer_key]

        # For ReX, we need to combine the DAGs. Hardcoded for now to combine
        # the first and second DAGs
        estimator1 = getattr(self.trainer[list(self.trainer.keys())[0]], 'rex')
        estimator2 = getattr(self.trainer[list(self.trainer.keys())[1]], 'rex')
        union_dag, inter_dag, union_cycles_removed, inter_cycles_removed = utils.combine_dags(
            estimator1.dag, estimator2.dag,
            discrepancies=estimator1.shaps.shap_discrepancies,
            prior=prior
        )
        if combine_op not in {'union', 'intersection'}:
            raise ValueError("combine_op must be 'union' or 'intersection'")
        if combine_op == 'union':
            dag = union_cycles_removed
        else:
            dag = inter_cycles_removed

        # Create a new Experiment object for the combined DAG
        new_trainer = f"{self.dataset_name}_rex"
        if new_trainer in self.trainer:
            del self.trainer[new_trainer]
        self.trainer[new_trainer] = Experiment(
            experiment_name=self.dataset_name,
            model_type='rex',
            data=self.data,
            data_is_processed=True,
            train_idx=self.train_idx,
            test_idx=self.test_idx,
            input_path=self.dataset_path,
            output_path=self.output_path,
            verbose=False)

        # Set the DAG and evaluate it
        self.trainer[new_trainer].ref_graph = self.ref_graph
        self.trainer[new_trainer].dag = dag
        if self.ref_graph is not None and self.data_columns is not None:
            self.trainer[new_trainer].metrics = evaluate_graph(
                self.ref_graph, dag, self.data_columns)
        else:
            self.trainer[new_trainer].metrics = None

        self.dag = self.trainer[new_trainer].dag
        self.metrics = self.trainer[new_trainer].metrics
        return self.trainer[new_trainer]

    def run(
        self,
        hpo_iterations: Optional[int] = None,
        bootstrap_iterations: Optional[int] = None,
        prior: Optional[List[List[str]]] = None,
        bootstrap_tolerance: Optional[float] = None,
        quiet: bool = False,
        combine_op: str = 'union',
        **kwargs: Any
    ) -> None:
        """
        Run the full experiment pipeline from creation to evaluation.

        This is a convenience wrapper that creates experiments, fits them, and
        combines/evaluates the resulting DAGs in one call.

        Args:
            hpo_iterations (int, optional): Number of HPO trials for REX.
                Defaults to None.
            bootstrap_iterations (int, optional): Number of bootstrap trials
                for REX. Defaults to None.
            prior (Optional[List[List[str]]], optional): Optional prior
                constraints to pass to ReX.
            bootstrap_tolerance (float, optional): Threshold to apply to the
                bootstrapped adjacency matrix. Defaults to None.
            quiet (bool, optional): Disable verbose output and progress
                indicators. Defaults to False.
            combine_op (str, optional): Operation used to combine DAGs in ReX.
                Defaults to 'union'.
            **kwargs: Additional keyword arguments forwarded to `fit_experiments`.

        Returns:
            None: This method does not return a value.
        """
        self.create_experiments()
        self.fit_experiments(
            hpo_iterations=hpo_iterations,
            bootstrap_iterations=bootstrap_iterations,
            prior=prior,
            bootstrap_tolerance=bootstrap_tolerance,
            quiet=quiet,
            **kwargs)
        self.combine_and_evaluate_dags(prior=prior, combine_op=combine_op)

    def save(self, full_filename_path: str) -> None:
        """
        Save the current trainer state to disk.

        This is a convenience alias for `save_model`.

        Args:
            full_filename_path (str): Full path to the output pickle file.

        Returns:
            None: This method does not return a value.
        """
        self.save_model(full_filename_path)

    def save_model(self, full_filename_path: str) -> None:
        """
        Save the model as an Experiment object.

        Use this after fitting to persist the trainer state for later reuse
        or analysis.

        Args:
            full_filename_path (str): A full path where to save the model,
                including the filename.

        Returns:
            None: This method does not return a value.
        """
        assert self.trainer, "No trainer to save"
        assert full_filename_path, "No output path specified"

        full_dir_path = os.path.dirname(full_filename_path)
        # Check only if not local dir
        if full_dir_path != "." and full_dir_path != "":
            assert os.path.exists(full_dir_path), \
                f"Output directory {full_dir_path} does not exist"
        else:
            full_dir_path = os.getcwd()

        saved_as = utils.save_experiment(
            os.path.basename(full_filename_path), full_dir_path,
            self.trainer, overwrite=False)
        print(f"Saved model as: {saved_as}", flush=True)

    def load(self, model_path: str) -> Dict[str, Experiment]:
        """
        Load a saved trainer state from disk.

        This is a convenience alias for `load_model`.

        Args:
            model_path (str): Path to the pickle file.

        Returns:
            Dict[str, Experiment]: The loaded trainer dictionary.
        """
        return self.load_model(model_path)

    def load_model(self, model_path: str) -> Dict[str, Experiment]:
        """
        Load the model from a pickle file.

        This restores the trainer dictionary and updates the cached DAG/metrics
        on the current instance.

        Args:
            model_path (str): Path to the pickle file containing the model

        Returns:
            Dict[str, Experiment]: The loaded trainer dictionary.
        """
        with open(model_path, 'rb') as f:
            self.trainer = pickle.load(f)
            print(f"Loaded model from: {model_path}", flush=True)

        # Set the dag and metrics
        self.dag = self.trainer[list(self.trainer.keys())[-1]].dag
        self.metrics = self.trainer[list(self.trainer.keys())[-1]].metrics
        return self.trainer

    def _sampling_summary(self) -> str:
        """
        Generate a summary string of the SHAP adaptive sampling strategy.

        Returns:
            str: A summary of the SHAP adaptive sampling strategy.
        """
        if self.estimator != 'rex':
            return "SHAP adaptive sampling: not used"
        rex_estimator = None
        for trainer in self.trainer.values():
            candidate = getattr(trainer, 'rex', None)
            if candidate is not None:
                rex_estimator = candidate
                break
        if rex_estimator is None:
            return "SHAP adaptive sampling: unavailable"

        adaptive = getattr(rex_estimator, 'adaptive_shap_sampling', True)
        max_shap_samples = getattr(
            rex_estimator, 'max_shap_samples', DEFAULT_MAX_SAMPLES)
        k_max = getattr(rex_estimator, 'K_max', 5)
        if not isinstance(max_shap_samples, int) or max_shap_samples <= 0:
            max_shap_samples = DEFAULT_MAX_SAMPLES
        if not isinstance(k_max, int) or k_max <= 0:
            k_max = 5

        data_ref = self.data
        if data_ref is None:
            for trainer in self.trainer.values():
                candidate_data = getattr(trainer, 'data', None)
                if candidate_data is not None:
                    data_ref = candidate_data
                    break
        if data_ref is None:
            return "SHAP adaptive sampling: unavailable"

        n_rows = len(data_ref)
        if not adaptive:
            mode = "no_sampling"
            n_background = n_rows
            k_target = 1
            return (
                "SHAP adaptive sampling: disabled "
                f"({mode}, K={k_target}, samples={n_background})"
            )
        if n_rows <= max_shap_samples:
            mode = "no_sampling"
            n_background = n_rows
            k_target = 1
        elif n_rows <= 2 * max_shap_samples:
            mode = "single_sample"
            n_background = max_shap_samples
            k_target = 1
        else:
            mode = "multi_sample"
            n_background = min(max_shap_samples, n_rows)
            k_target = min(int(k_max), 5)

        return f"SHAP adaptive sampling: {mode} (K={k_target}, samples={n_background})"

    def printout_results(
        self,
        graph: nx.DiGraph,
        metrics: Optional[Metrics],
        combine_op: str
    ) -> None:
        """
        Print the DAG and metrics to stdout in a readable, hierarchical format.

        This is intended for CLI runs where the user needs a quick textual
        summary of the discovered graph, optional evaluation metrics, and
        the sampling strategy used during estimation.

        Args:
            graph (nx.DiGraph): The DAG to print.
            metrics (Optional[Metrics]): Optional metrics summary to display.
            combine_op (str): The DAG combination operation used for labeling.

        Returns:
            None: This method does not return a value.
        """
        if len(graph.edges()) == 0:
            print("Empty graph")
            print(self._sampling_summary())
            return

        print(self._sampling_summary())
        combination = "Union" if combine_op == 'union' else "Intersection"
        msg = f"Graph from '{self.estimator.upper()}' using {combination} of DAGs:"
        print(f"{msg}\n" + "-" * len(msg))

        def dfs(node: Any, visited: Set[Any], indent: str = "") -> None:
            if node in visited:
                return  # Avoid revisiting nodes
            visited.add(node)

            # Print edges for this node
            for neighbor in graph.successors(node):
                print(f"{indent}{node} -> {neighbor}")
                dfs(neighbor, visited, indent + "  ")

        visited = set()

        # Start traversal from all nodes without predecessors (roots)
        for node in graph.nodes:
            if graph.in_degree(node) == 0:
                dfs(node, visited)

        # Handle disconnected components (not reachable from any "root")
        for node in graph.nodes:
            if node not in visited:
                dfs(node, visited)

        if metrics is not None:
            msg = f"Graph {combination} Metrics:"
            print(f"\n{msg}\n" + "-" * len(msg))
            print(metrics)



    def export(self, output_file: str) -> None:
        """
        Export the current DAG to a DOT file.

        This is a convenience alias for `export_dag`.

        Args:
            output_file (str): Path to the output DOT file.

        Returns:
            None: This method does not return a value.
        """
        self.export_dag(output_file)

    def export_dag(self, output_file: str) -> str:
        """
        Export the most recent DAG to a DOT file.

        This is typically called after training to persist the discovered
        causal graph for external inspection or visualization.

        Args:
            output_file (str): Path to the output DOT file.

        Returns:
            str: The path to the output DOT file.
        """
        model = self.trainer[list(self.trainer.keys())[-1]]
        if model.dag is None:
            raise ValueError(
                "No DAG available to export. Run the experiment first.")
        return utils.graph_to_dot_file(model.dag, output_file)

    def plot(
        self,
        show_metrics: bool = False,
        show_node_fill: bool = True,
        title: Optional[str] = None,
        ax: Optional[Axes] = None,
        figsize: Tuple[int, int] = (5, 5),
        dpi: int = 75,
        save_to_pdf: Optional[str] = None,
        layout: str = 'dot',
        **kwargs: Any
    ) -> None:
        """
        Plot the current DAG using networkx and matplotlib utilities.

        Use this to visualize the discovered graph after training, optionally
        overlaying evaluation metrics or saving the figure to a PDF file.

        Args:
            show_metrics (bool, optional): Whether to show metrics on the plot.
            show_node_fill (bool, optional): Whether to fill nodes with color.
            title (Optional[str], optional): Title for the plot.
            ax (Optional[Axes], optional): Matplotlib axes to draw on.
            figsize (Tuple[int, int], optional): Figure size in inches.
            dpi (int, optional): Figure DPI.
            save_to_pdf (Optional[str], optional): Path to save the plot as PDF.
            layout (str, optional): Layout engine to use ('dot' or 'circular').
            **kwargs: Additional keyword arguments forwarded to `plot.dag`.

        Returns:
            None: This method does not return a value.
        """
        model = self.trainer[list(self.trainer.keys())[-1]]
        if model.dag is None:
            raise ValueError(
                "No DAG available to plot. Run the experiment first.")
        if model.ref_graph is not None:
            ref_graph = model.ref_graph
        else:
            ref_graph = None
        plot.dag(
            graph=model.dag, reference=ref_graph, show_metrics=show_metrics,
            show_node_fill=show_node_fill, title=title or "",
            ax=ax, figsize=figsize, dpi=dpi, save_to_pdf=save_to_pdf,
            layout=layout, **kwargs)

    def plot_interactive(
        self,
        ui_parent: Any,
        show_metrics: bool = False,
        show_node_fill: bool = True,
        title: Optional[str] = None,
        layout: str = "dagre",
        rank_dir: str = "TB",
        width: str = "900px",
        height: str = "500px",
        persist_positions: bool = True,
        on_node_click: Optional[Callable[[str], None]] = None,
        on_edge_click: Optional[Callable[[str, List[str]], None]] = None,
        root_causes: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Dict[str, float]]:
        """
        Render the current DAG in a NiceGUI container using Cytoscape.js.

        Example (within a NiceGUI page):
            >>> from nicegui import ui
            >>> with ui.column() as container:
            ...     discoverer.plot_interactive(container, layout="dagre", rank_dir="LR")

        Args:
            ui_parent (Any): NiceGUI container to attach the visualization.
            show_metrics (bool, optional): Reserved for parity with plot().
            show_node_fill (bool, optional): Whether to apply node fill based on scores.
            title (Optional[str], optional): Title to show above the graph.
            layout (str, optional): "dagre" or "elk".
            rank_dir (str, optional): Layout direction ("LR", "RL", "TB", "BT").
            width (str, optional): CSS width of the graph container.
            height (str, optional): CSS height of the graph container.
            persist_positions (bool, optional): Persist node positions on drag.
            on_node_click (Callable, optional): Callback receiving the node id.
            on_edge_click (Callable, optional): Callback receiving edge id and classes.
            root_causes (Optional[List[str]]): Nodes to emphasize with a thicker border.
            **kwargs: Reserved for future styling options.

        Returns:
            Dict[str, Dict[str, float]]: Persisted node positions keyed by id.
        """
        _ = show_metrics
        _ = kwargs
        model = self.model
        if model.dag is None:
            raise ValueError("No DAG available to plot. Run the experiment first.")

        graph_id = uuid.uuid4().hex
        ref_graph = model.ref_graph
        use_saved = persist_positions and bool(self._cy_positions)
        elements, _ = _build_cytoscape_elements(
            model.dag,
            ref_graph,
            show_node_fill=show_node_fill,
            root_causes=root_causes,
            positions=self._cy_positions if use_saved else None,
        )
        layout_config = _cytoscape_layout_config(layout, rank_dir, use_saved)
        stylesheet = _cytoscape_stylesheet()
        spec: Dict[str, Any] = {
            "elements": elements,
            "style": stylesheet,
            "layout": layout_config,
            "width": width,
            "height": height,
            "asset_base": _CY_ASSET_BASE_URL or "",
        }

        from nicegui import app, ui

        _ensure_cytoscape_assets()

        position_endpoint = None
        click_endpoint = None
        edge_click_endpoint = None

        if persist_positions:
            position_endpoint = f"/_cytoscape/{graph_id}/positions"

            async def _handle_positions(payload: Dict[str, Any]) -> Dict[str, str]:
                positions = payload.get("positions")
                if isinstance(positions, dict):
                    updated: Dict[str, Dict[str, float]] = {}
                    for node_id, pos in positions.items():
                        if isinstance(pos, dict):
                            try:
                                updated[str(node_id)] = {
                                    "x": float(pos.get("x", 0.0)),
                                    "y": float(pos.get("y", 0.0)),
                                }
                            except (TypeError, ValueError):
                                continue
                    self._cy_positions = updated
                return {"status": "ok"}

            if position_endpoint not in _CY_REGISTERED_ROUTES:
                app.add_api_route(
                    position_endpoint, _handle_positions, methods=["POST"]
                )
                _CY_REGISTERED_ROUTES.add(position_endpoint)

        if on_node_click is not None:
            click_endpoint = f"/_cytoscape/{graph_id}/click"

            async def _handle_click(payload: Dict[str, Any]) -> Dict[str, str]:
                node_id = payload.get("node_id")
                if node_id is not None:
                    on_node_click(str(node_id))
                return {"status": "ok"}

            if click_endpoint not in _CY_REGISTERED_ROUTES:
                app.add_api_route(
                    click_endpoint, _handle_click, methods=["POST"]
                )
                _CY_REGISTERED_ROUTES.add(click_endpoint)

        if on_edge_click is not None:
            edge_click_endpoint = f"/_cytoscape/{graph_id}/edge"

            async def _handle_edge_click(payload: Dict[str, Any]) -> Dict[str, str]:
                edge_id = payload.get("edge_id")
                classes = payload.get("classes")
                if edge_id is not None:
                    if isinstance(classes, list):
                        class_list = [str(item) for item in classes]
                    else:
                        class_list = []
                    on_edge_click(str(edge_id), class_list)
                return {"status": "ok"}

            if edge_click_endpoint not in _CY_REGISTERED_ROUTES:
                app.add_api_route(
                    edge_click_endpoint, _handle_edge_click, methods=["POST"]
                )
                _CY_REGISTERED_ROUTES.add(edge_click_endpoint)

        container_id = f"cytoscape-{graph_id}"
        container_html = (
            f'<div id="{container_id}" '
            f'style="width: {width}; height: {height};"></div>'
        )
        script = _cytoscape_init_script(
            container_id, spec, position_endpoint, click_endpoint, edge_click_endpoint
        )

        if ui_parent is None or ui_parent is ui:
            if title:
                ui.label(title)
            ui.html(container_html, sanitize=False)
            ui.run_javascript(script)
        else:
            with ui_parent:
                if title:
                    ui.label(title)
                ui.html(container_html, sanitize=False)
                ui.run_javascript(script)

        return self._cy_positions

    @property
    def model(self) -> Experiment:
        """
        Return the most recent Experiment from the trainer map.

        This is commonly used after training to access the final DAG or metrics.

        Args:
            None.

        Returns:
            Experiment: The most recently added Experiment instance.
        """
        return self.trainer[list(self.trainer.keys())[-1]]


def _cytoscape_sanity_check() -> Dict[str, int]:
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
