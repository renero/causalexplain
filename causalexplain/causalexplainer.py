"""
This module contains the GraphDiscovery class which is responsible for
creating, fitting, and evaluating causal discovery experiments.
"""
import os
import re
import pickle
import pandas as pd
from matplotlib.axes import Axes
import networkx as nx
from typing import Any, Dict, List, Tuple, Optional, cast

from causalexplain.common import (
    DEFAULT_REGRESSORS,
    utils,
)
from causalexplain.common import plot
from causalexplain.common.notebook import Experiment
from causalexplain.metrics.compare_graphs import evaluate_graph


class GraphDiscovery:
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        model_type: str = 'rex',
        csv_filename: Optional[str] = None,
        true_dag_filename: Optional[str] = None,
        verbose: bool = False,
        seed: int = 42
    ) -> None:
        """
        Initializes a new instance of the GraphDiscovery class.

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
        """
        self.trainer: Dict[str, Experiment] = {}
        normalized_experiment = self._normalize_optional_str(experiment_name)
        normalized_csv = self._normalize_optional_str(csv_filename)

        if normalized_experiment is None and normalized_csv is None:
            self._init_empty_state(seed)
            return

        self._validate_experiment_inputs(normalized_experiment, normalized_csv)
        csv_filename = cast(str, normalized_csv)

        self.experiment_name = normalized_experiment
        self.estimator = model_type
        self.csv_filename = csv_filename
        self.dot_filename = true_dag_filename
        self.verbose = verbose
        self.seed = seed
        self.train_size = 0.9
        self.random_state = seed

        self.dataset_path = os.path.dirname(csv_filename)
        self.output_path = os.getcwd()

        self.ref_graph = self._load_reference_graph(true_dag_filename)
        if true_dag_filename is not None and self.ref_graph is None:
            raise ValueError("True DAG could not be loaded from dot file")
        self.data, self.dataset_name, self.data_columns = self._load_dataset_metadata(csv_filename)
        self.train_idx, self.test_idx = self._build_split_indices(
            self.data, self.train_size, self.random_state)
        self._validate_dag_nodes(self.ref_graph, self.data_columns)
        self.regressors = self._select_regressors()

    @staticmethod
    def _normalize_optional_str(value: Optional[str]) -> Optional[str]:
        """Strip whitespace and normalize empty strings to None for optional values."""
        if not isinstance(value, str):
            return value
        value = value.strip()
        return value or None

    def _init_empty_state(self, seed: int) -> None:
        """Initialize the object when no experiment name or CSV is provided."""
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
        self.trainer: Dict[str, Experiment] = {}

    def _validate_experiment_inputs(
        self,
        experiment_name: Optional[str],
        csv_filename: Optional[str]
    ) -> None:
        """Ensure experiment and CSV inputs are provided together."""
        if experiment_name is None or csv_filename is None:
            raise ValueError(
                "Both 'experiment_name' and 'csv_filename' must be provided together."
            )

    def _load_reference_graph(self, true_dag_filename: Optional[str]):
        """Load the reference DAG from disk if a path is provided."""
        if true_dag_filename is None:
            return None
        return utils.graph_from_dot_file(true_dag_filename)

    def _load_dataset_metadata(self, csv_filename: str):
        """Load dataset name and column metadata from the CSV file."""
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
        # Share split indices across experiments to avoid repeated sampling.
        train_idx = data.sample(frac=train_size, random_state=random_state).index
        test_idx = data.index[~data.index.isin(train_idx)]
        return train_idx, test_idx

    @staticmethod
    def _validate_column_names(data_columns: List) -> None:
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
        data_columns: Optional[List]
    ) -> None:
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

    def _select_regressors(self):
        """Select regressors based on the estimator type."""
        if self.estimator == 'rex':
            return DEFAULT_REGRESSORS
        return [self.estimator]

    def create_experiments(self) -> dict:
        """
        Create an Experiment object for each regressor.

        Args:
            dataset_name (str): Name of the dataset
            true_dag (str): Path to the true DAG DOT file
            regressors (list): List of regressor types to create experiments for
            dataset_path (str): Path to the input dataset
            output_path (str): Path for output files

        Returns:
            dict: A dictionary of Experiment objects
        """
        if self.csv_filename is None:
            raise AttributeError("CSV filename is required to create experiments.")

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
        **kwargs
    ) -> None:
        """
        Fit the Experiment objects.

        Args:
            trainer (dict): A dictionary of Experiment objects
            estimator (str): The estimator to use ('rex' or other)
            verbose (bool, optional): Whether to print verbose output.
                Defaults to False.
            hpo_iterations (int, optional): Number of HPO trials for REX.
                Defaults to None.
            bootstrap_iterations (int, optional): Number of bootstrap trials
                for REX. Defaults to None.
            bootstrap_tolerance (float, optional): Threshold to apply to the
                bootstrapped adjacency matrix. Defaults to None.
            quiet (bool, optional): Disable verbose output and progress
                indicators. Defaults to False.
        """
        verbose = False if quiet else self.verbose
        xargs: Dict[str, Any] = {}
        if self.estimator == 'rex':
            xargs = {
                'verbose': verbose,
                # 'prior': prior
            }
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
        Retrieve the DAG from the Experiment objects.

        Args:
            prior (List[List[str]], optional): The prior to use for ReX.
                Defaults to None.
            combine_op (str, optional): Operation used to combine DAGs in ReX.
                Supported values are: 'union' and 'intersection'.

        Returns:
            Experiment: The experiment object with the final DAG
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
            **kwargs):
        """
        Run the experiment.

        Args:
            hpo_iterations (int, optional): Number of HPO trials for REX.
                Defaults to None.
            bootstrap_iterations (int, optional): Number of bootstrap trials
                for REX. Defaults to None.
            bootstrap_tolerance (float, optional): Threshold to apply to the
                bootstrapped adjacency matrix. Defaults to None.
            quiet (bool, optional): Disable verbose output and progress
                indicators. Defaults to False.
            combine_op (str, optional): Operation used to combine DAGs in ReX.
                Defaults to 'union'.
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
        self.save_model(full_filename_path)

    def save_model(self, full_filename_path: str) -> None:
        """
        Save the model as an Experiment object.

        Args:
            full_filename_path (str): A full path where to save the model,
                including the filename.
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
        return self.load_model(model_path)

    def load_model(self, model_path: str) -> Dict[str, Experiment]:
        """
        Load the model from a pickle file.

        Args:
            model_path (str): Path to the pickle file containing the model

        Returns:
            Experiment: The loaded Experiment object
        """
        with open(model_path, 'rb') as f:
            self.trainer = pickle.load(f)
            print(f"Loaded model from: {model_path}", flush=True)

        # Set the dag and metrics
        self.dag = self.trainer[list(self.trainer.keys())[-1]].dag
        self.metrics = self.trainer[list(self.trainer.keys())[-1]].metrics
        return self.trainer

    def printout_results(self, graph, metrics, combine_op: str) -> None:
        """
        This method prints the DAG to stdout in hierarchical order.

        Parameters:
        -----------
        dag : nx.DiGraph
            The DAG to be printed.
        """
        if len(graph.edges()) == 0:
            print("Empty graph")
            return

        combination = "Union" if combine_op == 'union' else "Intersection"
        msg = f"Graph from '{self.estimator.upper()}' using {combination} of DAGs:"
        print(f"{msg}\n" + "-" * len(msg))

        def dfs(node, visited, indent=""):
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
        self.export_dag(output_file)

    def export_dag(self, output_file: str) -> str:
        """
        This method exports the DAG to a DOT file.

        Parameters:
        -----------
        dag : nx.DiGraph
            The DAG to be exported.
        output_file : str
            The path to the output DOT file.

        Returns:
        --------
        str
            The path to the output DOT file.
        """
        model = self.trainer[list(self.trainer.keys())[-1]]
        if model.dag is None:
            raise ValueError("No DAG available to export. Run the experiment first.")
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
        **kwargs
    ):
        """
        This method plots the DAG using networkx and matplotlib.

        Parameters:
        -----------
        show_metrics : bool, optional
            Whether to show the metrics on the plot. Defaults to False.
        show_node_fill : bool, optional
            Whether to fill the nodes with color. Defaults to True.
        title : str, optional
            The title of the plot. Defaults to None.
        ax : Axes, optional
            The matplotlib axes to plot on. Defaults to None.
        figsize : Tuple[int, int], optional
            The size of the plot. Defaults to (5, 5).
        dpi : int, optional
            The DPI of the plot. Defaults to 75.
        save_to_pdf : str, optional
            The path to save the plot as a PDF. Defaults to None.
        layout : str, optional
            The layout to use for the plot. Defaults to 'dot'. Other option
            is 'circular'.
        """
        model = self.trainer[list(self.trainer.keys())[-1]]
        if model.dag is None:
            raise ValueError("No DAG available to plot. Run the experiment first.")
        if model.ref_graph is not None:
            ref_graph = model.ref_graph
        else:
            ref_graph = None
        plot.dag(
            graph=model.dag, reference=ref_graph, show_metrics=show_metrics,
            show_node_fill=show_node_fill, title=title or "",
            ax=ax, figsize=figsize, dpi=dpi, save_to_pdf=save_to_pdf,
            layout=layout, **kwargs)

    @property
    def model(self):
        return self.trainer[list(self.trainer.keys())[-1]]
