"""
Main class for the REX estimator.
(C) J. Renero, 2022, 2023, 2024
"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name, W0221:arguments-differ
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

from multiprocessing import get_context
from functools import partial
import multiprocessing
import os
import time
import warnings
from collections import defaultdict
from copy import copy, deepcopy
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from mlforge.mlforge import Pipeline
from mlforge.progbar import ProgBar
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_random_state

from ...common import (
    utils, DEFAULT_HPO_TRIALS, DEFAULT_BOOTSTRAP_TRIALS,
    DEFAULT_BOOTSTRAP_TOLERANCE, DEFAULT_BOOTSTRAP_SAMPLING_SPLIT
)
from .knowledge import Knowledge
from ...explainability.regression_quality import RegQuality
from ...explainability.shapley import ShapEstimator
from ...metrics.compare_graphs import Metrics, evaluate_graph
from ...models import GBTRegressor, NNRegressor


np.set_printoptions(precision=4, linewidth=120)
warnings.filterwarnings('ignore')


class Rex(BaseEstimator, ClassifierMixin):
    """
    Regression with Explainability (Rex) is a causal inference discovery that
    uses a regression model to predict the outcome of a treatment and uses
    explainability to identify the causal variables.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from causalexplain.estimators.rex import Rex   # doctest: +SKIP
    >>> import numpy as np   # doctest: +SKIP

    >>> dataset_name = 'rex_generated_linear_0'  # doctest: +SKIP
    >>> ref_graph = utils.graph_from_dot_file(f"../data/{dataset_name}.dot")  # doctest: +SKIP
    >>> data = pd.read_csv(f"{input_path}{dataset_name}.csv")  # doctest: +SKIP
    >>> scaler = StandardScaler()  # doctest: +SKIP
    >>> data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)  # doctest: +SKIP
    >>> train = data.sample(frac=0.8, random_state=42)  # doctest: +SKIP
    >>> test = data.drop(train.index)  # doctest: +SKIP

    >>> rex = Rex(   # doctest: +SKIP
        name=dataset_name, tune_model=tune_model,   # doctest: +SKIP
        model_type=model_type, explainer=explainer)   # doctest: +SKIP
    >>> rex.fit_predict(train, test, ref_graph)   # doctest: +SKIP

    """

    shaps = None
    hierarchies = None
    pi = None
    models = None
    dag = None
    indep = None
    name: str = ""
    feature_names: List[str] = []
    root_causes: List[str] = []
    hpo_n_trials: int = 100
    G_final: Optional[nx.DiGraph] = None
    verbose: bool = False
    n_jobs: int = -1
    random_state: Optional[int] = None
    is_fitted_ = False

    def __init__(
            self,
            name: str,
            model_type: str = "nn",
            explainer: str = "gradient",
            tune_model: bool = False,
            correlation_th: Optional[float] = None,
            corr_method: str = 'spearman',
            corr_alpha: float = 0.6,
            corr_clusters: int = 15,
            condlen: int = 1,
            condsize: int = 0,
            mean_pi_percentile: float = 0.8,
            discrepancy_threshold: float = 0.99,
            hpo_n_trials: int = DEFAULT_HPO_TRIALS,
            bootstrap_trials: int = DEFAULT_BOOTSTRAP_TRIALS,
            bootstrap_sampling_split='auto',
            bootstrap_tolerance: Union[float, str] = 'auto',
            bootstrap_parallel_jobs: int = 0,
            parallel_jobs: int = 0,
            verbose: bool = False,
            prog_bar=True,
            silent: bool = False,
            shap_fsize: Tuple[int, int] = (10, 10),
            dpi: int = 75,
            pdf_filename: Optional[str] = None,
            random_state=1234,
            **kwargs):
        """
        Arguments:
        ----------
            model_type (str): The type of model to use. Either "nn" for MLP
                or "gbt" for GradientBoostingRegressor.
            explainer (str): The explainer to use for the shap values. The default
                values is "explainer", which uses the shap.Explainer class. Other
                options are "gradient", which uses the shap.GradientExplainer class,
                and "kernel", which uses the shap.KernelExplainer class.
            tune_model (bool): Whether to tune the model for HPO. Default is False.
            correlation_th (float): The threshold for the correlation. Default is None.
            corr_method (str): The method to use for the correlation.
                Default is "spearman", but it can also be 'pearson', 'kendall or 'mic'.
            corr_alpha (float): The alpha value for the correlation. Default is 0.6.
            corr_clusters (int): The number of clusters to use for the correlation.
                Default is 15.
            condlen (int): The depth of the conditioning sequence. Default is 1.
            condsize (int): The size of the conditioning sequence. Default is 0.
            mean_pi_percentile (float): The percentile for the mean permutation
                importance. Default is 0.8.
            discrepancy_threshold (float): The threshold for the discrepancy.
                Default is 0.99.
            prog_bar (bool): Whether to display a progress bar.
                Default is False.
            verbose (bool): Whether to print the progress of the training. Default
                is False.
            silent (bool): Whether to print anything. Default is False. This overrides
                the verbose argument and the prog_bar argument.
            random_state (int): The seed for the random number generator.
                Default is 1234.

            Additional arguments:
                shap_fsize: The size of the figure for the shap values.
                    Default is (5, 3).
                dpi: The dpi for the figures. Default is 75.
                pdf_filename: The filename for the pdf file where final comparison will
                    be saved. Default is None, producing no pdf file.
        """
        self.name = name
        self.prior = None
        self.hpo_study_name = kwargs.get(
            'hpo_study_name', f"{self.name}_{model_type}")

        self.prog_bar = prog_bar
        self.verbose = verbose
        self.silent = silent
        self.random_state = random_state

        self.model_type = NNRegressor if model_type == "nn" else GBTRegressor
        self.explainer = explainer
        self._check_model_and_explainer(model_type, explainer)

        # Get a copy of kwargs
        self.kwargs = deepcopy(kwargs)

        self.tune_model = tune_model
        self.correlation_th = correlation_th
        self.corr_method = corr_method
        self.corr_alpha = corr_alpha
        self.corr_clusters = corr_clusters

        self.hpo_n_trials = hpo_n_trials
        self.bootstrap_trials = bootstrap_trials
        self.bootstrap_sampling_split = bootstrap_sampling_split
        self.bootstrap_tolerance = bootstrap_tolerance
        self.bootstrap_parallel_jobs = bootstrap_parallel_jobs
        self.parallel_jobs = parallel_jobs

        self.condlen = condlen
        self.condsize = condsize
        self.mean_pi_percentile = mean_pi_percentile
        self.mean_pi_threshold = 0.0
        self.discrepancy_threshold = discrepancy_threshold

        self.is_fitted_ = False
        self.is_iterative_fitted_ = False

        self.shap_fsize = shap_fsize
        self.dpi = dpi
        self.pdf_filename = pdf_filename

        # This is used to decide how to fit the model in the pipeline
        self.fit_step = 'models.tune_fit' if self.tune_model else 'models.fit'

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._fit_desc = "Running Causal Discovery pipeline"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def fit(self, X, y=None, pipeline: Optional[Union[list, str]] = None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self
        """
        self.init_fit_time = time.time()

        self.random_state_state = check_random_state(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.feature_names = utils.get_feature_names(X)
        self.feature_types = utils.get_feature_types(X)

        self.X = copy(X)
        self.y = copy(y) if y is not None else None

        # Create the pipeline for the training stages.
        self._set_fit_pipeline(pipeline)

        n_steps = self.fit_pipeline.len()
        n_steps += (self._steps_from_hpo(self.fit_pipeline) * 2) - 1

        self.fit_pipeline.run(n_steps)
        self.fit_pipeline.close()
        self.is_fitted_ = True

        self.end_fit_time = time.time()
        self.fit_time = self.end_fit_time - self.init_fit_time

        return self

    def _set_fit_pipeline(self, pipeline: list | str | None) -> None:
        """
        Set the pipeline for the training stages.

        Parameters
        ----------
        pipeline : list
            A list of tuples with the steps to add to the pipeline.
        """
        self.fit_pipeline = Pipeline(
            self,  # type: ignore
            description="Fitting models", prog_bar=self.prog_bar,
            verbose=self.verbose, silent=self.silent, subtask=True)
        if pipeline is not None:
            if isinstance(pipeline, list):
                self.fit_pipeline.from_list(pipeline)
            elif isinstance(pipeline, str):
                self.fit_pipeline.from_config(pipeline)
        else:
            # This is the final set of steps in default mode.
            steps = [
                ('models', self.model_type),
                ('models.tune_fit', {'hpo_n_trials': self.hpo_n_trials}),
                ('models.score', {})
            ]
            self.fit_pipeline.from_list(steps)

    def predict(self,
                X: pd.DataFrame,
                ref_graph: Optional[nx.DiGraph] = None,
                prior: Optional[List[List[str]]] = None,
                pipeline: Optional[list | str] = None
                ):
        """
        Predicts the causal graph from the given data.

        Parameters
        ----------
        - X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.
        - ref_graph: nx.DiGraph
            The reference graph, or ground truth.
        - prior: str
            The prior to use for building the DAG. This prior is a list of lists
            of node/feature names, ordered according to a temporal structure so that
            the first list contains the first set of nodes to be considered as
            root causes, the second list contains the set of nodes to be
            considered as potential effects of the first set, and the nodes in this
            second list, and so on. The number of lists in the prior is the depth of
            the conditioning sequence. This prior imposes the rule that the nodes in
            the first list are the only ones that can be root causes, and the nodes
            in the following lists cannot be the cause of the nodes in the previous
            lists. If the prior is not provided, the DAG is built without any prior
            information.

        Returns
        -------
        - G_final : nx.DiGraph
            The final graph, after the correction stage.

        Examples
        --------
        In the following example, where four features are used, the prior is
        defined as [['A', 'B'], ['C', 'D']], which means that the first set of
        features to be considered as root causes are 'A' and 'B', and the second
        set of features to be considered as potential effects of the first set are
        'C' and 'D'.

        The resulting DAG cannot contain any edge from 'C' or 'D' to 'A' or 'B'.

            ```python
            rex.predict(X_test, ref_graph, prior=[['A', 'B'], ['C', 'D']])
            ```
        """
        if not self.is_fitted_:
            raise ValueError("This Rex instance is not fitted yet. \
                Call 'fit' with appropriate arguments before using this estimator.")

        self.init_predict_time = time.time()

        # Check that prior is a list of lists and does not contain repeated elements.
        if prior is not None:
            if not isinstance(prior, list):
                raise ValueError("The prior must be a list of lists.")
            if any([len(p) != len(set(p)) for p in prior]):
                raise ValueError("The prior cannot contain repeated elements.")
            self.prior = prior

        # If reference graph is passed, store it in the object.
        if ref_graph is not None:
            self.ref_graph = ref_graph

        # Create a new pipeline for the prediction stages.
        self.predict_pipeline = Pipeline(
            self,  # type: ignore
            description="Predicting causal graph",
            prog_bar=self.prog_bar,
            verbose=self.verbose,
            silent=self.silent,
            subtask=True)

        # Overwrite values for prog_bar and verbosity with current pipeline
        # values, in case predict is called from a loaded experiment
        if hasattr(self, "shaps") and self.shaps is not None:
            self.shaps.prog_bar = self.prog_bar
            self.shaps.verbose = self.verbose

        # Load a pipeline if specified, or create the default one.
        self._set_predict_pipeline(ref_graph, pipeline)
        n_steps = self._get_steps_predict_pipeline()

        self.predict_pipeline.run(n_steps)

        # Check if "G_final" exists in this object (self)
        if 'G_final' in self.__dict__ and self.G_final is not None:
            if '\\n' in self.G_final.nodes:
                self.G_final.remove_node('\\n')
        self.predict_pipeline.close()

        self.end_predict_time = time.time()
        self.predict_time = self.end_predict_time - self.init_predict_time

        # For compatibility with compared methods, we always set an attribute
        # called 'dag' for the final DAG.
        self.dag = self.G_final

        return self

    def _get_steps_predict_pipeline(self):
        """
        Get the number of steps in the pipeline for the prediction stage.

        Returns
        -------
        n_steps : int
            The number of steps in the pipeline.
        """
        n_steps = self.predict_pipeline.len()
        n_steps += self._steps_from_hpo(self.predict_pipeline)
        n_steps += 1

        return n_steps

    def _set_predict_pipeline(
            self,
            ref_graph: Optional[nx.DiGraph],
            pipeline: list | str | None):
        """
        Set the pipeline for the prediction stage as `self.predict_pipeline`.

        Parameters
        ----------
        ref_graph: nx.DiGraph
            The reference graph, or ground truth.
        pipeline: list | str | None
            The pipeline to use for the prediction stage.
        """
        if pipeline is not None:
            if isinstance(pipeline, list):
                self.predict_pipeline.from_list(pipeline)
            elif isinstance(pipeline, str):
                self.predict_pipeline.from_config(pipeline)
        else:
            steps = [
                ('shaps', ShapEstimator, {
                    'models': self.models,
                    'parallel_jobs': self.parallel_jobs
                }),
                ('G_final', 'bootstrap', {
                    'num_iterations': self.bootstrap_trials,
                    'sampling_split': self.bootstrap_sampling_split,
                    'tolerance': self.bootstrap_tolerance,
                    'parallel_jobs': self.bootstrap_parallel_jobs,
                    'random_state': self.random_state
                })
            ]
            if ref_graph is not None:
                steps.append(('metrics', 'score', {
                    'ref_graph': ref_graph, 'predicted_graph': 'G_final'})
                )

            self.predict_pipeline.from_list(steps)

    def fit_predict(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            ref_graph: nx.DiGraph,
            prior: Optional[List[List[str]]] = None):
        """
        Fit the model according to the given training data and predict
        the outcome of the treatment.

        Parameters
        ----------
        train : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        test : {array-like, sparse matrix}, shape (n_samples, n_features)
            Test vector, where n_samples is the number of samples and
            n_features is the number of features.
        ref_graph : nx.DiGraph
            The reference graph, or ground truth.

        Returns
        -------
        G_final : nx.DiGraph
            The final graph, after the correction stage.
        """
        self.fit(train)
        self.predict(test, ref_graph, prior)
        return self

    @staticmethod
    def _bootstrap_iteration(iter, X, models, sampling_split, feature_names, prior,
                             random_state, verbose=False):
        """
        Process an iteration of the iterative prediction.

        Parameters
        ----------
        iter : int
            The current iteration number.
        X : pd.DataFrame
            The input data.
        models : list
            A list of models to use for the prediction.
        sampling_split : float
            The fraction of the data to use for bootstrapping.
        feature_names : list
            The names of the features.
        prior : list
            The prior knowledge on the graph to use for bootstrapping.
        random_state : int
            The random state to use for bootstrapping.
        verbose : bool
            Whether to print verbose messages.
        """
        data_sample = X.sample(frac=sampling_split,
                               random_state=iter * random_state)
        shaps_instance = ShapEstimator(
            models=models, parallel_jobs=0, prog_bar=False)
        shaps_instance.fit(data_sample)
        dag = shaps_instance.predict(data_sample, prior=prior)
        adjacency_matrix = utils.graph_to_adjacency(dag, feature_names)
        if verbose:
            print("· Iteration", iter + 1, "done.")
        return adjacency_matrix

    def _build_bootstrapped_adjacency_matrix(
        self,
        X: pd.DataFrame,
        num_iterations: int = DEFAULT_BOOTSTRAP_TRIALS,
        sampling_split: float = 0.5,
        prior: Optional[list] = None,
        parallel_jobs: int = 0,
        random_state: int = 1234,
    ) -> np.ndarray:
        """
        Performs iterative prediction on the given directed acyclic graph (DAG)
        and adjacency matrix.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        num_iterations : int
            The number of iterations to perform.
        sampling_split : float
            The fraction of the data to use for bootstrapping.
        prior : list
            The prior knowledge on the graph to use for bootstrapping.
        parallel_jobs : int
            The number of parallel jobs to use for bootstrapping.
        random_state : int
            The random state to use for bootstrapping.

        Returns
        -------
        adjacency_matrix : np.ndarray
        """
        # Assert that 'shaps' exists and has been fit
        if not self.is_fitted_:
            raise ValueError("This Rex instance is not fitted yet. \
                Call 'fit' with appropriate arguments before using this estimator.")

        if self.verbose:
            print(
                f"Building iterative adjacency matrix with {num_iterations} "
                f"iterations, {sampling_split:.2f} split.")

        iter_adjacency_matrix = np.zeros(
            (self.n_features_in_, self.n_features_in_))

        if self.prog_bar and not self.verbose:
            pbar = ProgBar().start_subtask("Bootstrap", num_iterations)
        else:
            pbar = None

        results = []
        if parallel_jobs != 0 and parallel_jobs != 1:
            # Prepare the partial function with fixed arguments
            partial_process_iteration = partial(
                Rex._bootstrap_iteration, X=X, models=self.models,
                sampling_split=sampling_split, feature_names=self.feature_names,
                prior=prior, random_state=random_state, verbose=self.verbose)

            # Determine the nr of processes to pass to Pool()
            if parallel_jobs == -1:
                nr_processes = min(multiprocessing.cpu_count(), num_iterations)
            else:
                nr_processes = min(parallel_jobs, multiprocessing.cpu_count())

            # Use multiprocessing Pool
            with get_context('spawn').Pool(processes=nr_processes) as pool:
                for result in pool.imap_unordered(
                        partial_process_iteration, range(num_iterations)):
                    results.append(result)
                    if pbar:
                        pbar.update_subtask("Bootstrap", len(results))
                if pbar:
                    pbar.remove("Bootstrap")
                    pbar = None
        else:
            # Sequential processing
            for iter in range(num_iterations):
                result = Rex._bootstrap_iteration(
                    iter, X, self.models, sampling_split,
                    self.feature_names, prior, random_state, self.verbose)
                results.append(result)
                if self.prog_bar and not self.verbose and pbar is not None:
                    pbar.update_subtask("Bootstrap", iter)
            if self.prog_bar and not self.verbose and pbar is not None:
                pbar.remove("Bootstrap")

        for result in results:
            iter_adjacency_matrix += result
        iter_adjacency_matrix = iter_adjacency_matrix / num_iterations

        return iter_adjacency_matrix

    def iterative_predict(self, X, ref_graph=None, **kwargs):
        return self.bootstrap(X, ref_graph, **kwargs)

    def bootstrap(
            self,
            X: pd.DataFrame,
            ref_graph: Optional[nx.DiGraph] = None,
            num_iterations: int = DEFAULT_BOOTSTRAP_TRIALS,
            sampling_split: float = DEFAULT_BOOTSTRAP_SAMPLING_SPLIT,
            prior: Optional[list] = None,
            random_state: int = 1234,
            tolerance: Union[float, str] = DEFAULT_BOOTSTRAP_TOLERANCE,
            key_metric: str = 'f1',
            direction: str = 'maximize',
            parallel_jobs: int = 0) -> nx.DiGraph:
        """
        Finds the best tolerance value for the iterative predict method by iterating
        over different tolerance values and selecting the one that gives the best
        `key_metric` with respect to the reference graph.

        Parameters
        ----------
        ref_graph : nx.DiGraph
            The reference graph to evaluate the F1 score against.
        target : str, optional
            The target DAG to evaluate. Defaults to 'shap'.
            Possible values: 'shap', 'rho', 'adjusted', 'perm_imp', 'indep', and 'final'
        key_metric : str, optional
            The key metric to evaluate. Defaults to 'f1'.
            Possible values: 'f1', 'precision', 'recall', 'shd', sid', 'aupr',
            'Tp', 'Tn', 'Fp', 'Fn'    '
        direction : str, optional
            The direction of the key metric. Defaults to 'maximize'.
            Possible values: 'maximize' or 'minimize'
        parallel_jobs: int, optional
            Number of processes to run the iterations in parallel. Defaults to 0.

        Returns
        -------
        nx.DiGraph : The best DAG found by the iterative predict method.
        """
        if ref_graph is None and tolerance == 'auto':
            print(f"Setting tolerance to {DEFAULT_BOOTSTRAP_TOLERANCE}, as no "
                  f"true_graph was provided.")
            tolerance = DEFAULT_BOOTSTRAP_TOLERANCE
        if direction != 'maximize' and direction != 'minimize':
            raise ValueError("direction must be 'maximize' or 'minimize'")

        if sampling_split == 'auto':
            sampling_split = self._set_sampling_split()

        if self.verbose:
            print(f"Iterative prediction with {num_iterations} iterations, and "
                  f"{sampling_split:.2f} sampling split.")

        iter_adjacency_matrix = self._build_bootstrapped_adjacency_matrix(
            X, num_iterations, sampling_split, prior, parallel_jobs, random_state)

        if self.shaps is not None:
            self.shaps.fit(X)
            self.shaps.predict(X)

        if tolerance == 'auto':
            self.tolerance = self._find_best_tolerance(
                ref_graph, key_metric, direction, iter_adjacency_matrix)
        else:
            self.tolerance = tolerance
            try:
                self.tolerance = float(self.tolerance)
            except ValueError:
                raise ValueError("tolerance must be a number")

        # Now, predict with selected tolerance
        return self._dag_from_bootstrap_adj_matrix(
            iter_adjacency_matrix, tolerance=self.tolerance)

    def _find_best_tolerance(
            self,
            ref_graph,
            key_metric,
            direction,
            iter_adjacency_matrix) -> float:
        """
        Finds the best tolerance value for the iterative predict method by iterating
        over different tolerance values and selecting the one that gives the best
        `key_metric` with respect to the reference graph.

        Parameters
        ----------
        ref_graph : nx.DiGraph
            The reference graph to evaluate the F1 score against.
        key_metric : str
            The key metric to evaluate. Defaults to 'f1'.
            Possible values: 'f1', 'precision', 'recall', 'shd', sid', 'aupr',
            'Tp', 'Tn', 'Fp', 'Fn'
        direction : str
            The direction of the key metric. Defaults to 'maximize'.
            Possible values: 'maximize' or 'minimize'
        iter_adjacency_matrix : np.ndarray
            The adjacency matrix obtained from the iterative prediction.

        Returns
        -------
        float : The best tolerance value.
        """
        if self.verbose:
            print("Finding best tolerance value for iterative prediction...")

            # This lambda expression is used to compare values, depending on the direction
        _is_better_value = {
            'maximize': lambda x, y: x >= y,
            'minimize': lambda x, y: x < y
        }[direction]

        reference_key_metric = -100000.0 if direction == 'maximize' else +100000.0
        self.iterative_metrics = []
        best_tolerance = 0.0
        for tol in np.arange(0.1, 1.0, 0.05):
            dag = self._dag_from_bootstrap_adj_matrix(
                iter_adjacency_matrix, tolerance=tol)

            metric = evaluate_graph(ref_graph, dag)
            self.iterative_metrics.append(metric)
            value_obtained = getattr(metric, key_metric)
            if _is_better_value(value_obtained, reference_key_metric):
                reference_key_metric = value_obtained
                best_tolerance = tol
                if self.verbose:
                    print(f"· · Better tolerance found: {best_tolerance:.2f}, "
                          f"{key_metric}: {reference_key_metric:.4f}")

        if self.verbose:
            print(f"Best tolerance: {best_tolerance:.2f}, "
                  f"{key_metric}: {reference_key_metric:.4f}")

        return best_tolerance

    def _dag_from_bootstrap_adj_matrix(
            self,
            iter_adjacency_matrix: np.ndarray,
            tolerance: float = 0.3) -> nx.DiGraph:
        """
        Performs iterative prediction on the given directed acyclic graph (DAG)
        and adjacency matrix. Prediction is based on the adjacency matrices previously
        computed, which are here normalized and filtered, so values below tolerance
        are set to zero. This way, only those edges present in more than "tolerance"
        percent of the iterations are kept.

        Parameters:
            dag (dict): The input DAG.
            num_iterations (int): The number of iterations to perform. Defaults to 10.
            tolerance (float): The tolerance value for filtering the adjacency matrix.
                Defaults to 0.3.
            inplace (bool): Whether to store the predicted DAG in the object.
                Defaults to True. If False, the predicted DAG is returned.

        Returns:
            dict: The predicted DAG.
        """
        assert isinstance(iter_adjacency_matrix, np.ndarray), \
            "Adjacency must be a 2D numpy array"
        assert isinstance(tolerance, (int, float)
                          ), "Tolerance must be a number"
        assert (tolerance <= 1.0) and (tolerance >= 0.0), \
            "Tolerance must be range between 0.0 and 1.0"
        assert hasattr(self, "shaps"), "ShapEstimator is None"

        filtered_matrix = self._filter_adjacency_matrix(
            iter_adjacency_matrix, tolerance)

        dag = utils.graph_from_adjacency(filtered_matrix, self.feature_names)
        dag = utils.break_cycles_if_present(
            dag, self.shaps.shap_discrepancies,                 # type: ignore
            self.prior, verbose=self.verbose)

        return dag

    def score(                                                  # type: ignore
            self,
            ref_graph: nx.DiGraph,
            predicted_graph: Union[str, nx.DiGraph] = 'final'
    ) -> Optional[Metrics]:
        """
        Obtains the score of the predicted graph against the reference graph.
        The score contains different metrics, such as the precision, recall,
        F1-score, SHD or SID.

        Parameters:
        -----------
            ref_graph (nx.DiGraph): The reference graph, or ground truth.
            predicted_graph (str): The name of the graph to use for the score.
                Default is 'final', but other possible intermediate graphs are
                'shap' and 'indep', for those stages of the pipeline corresponding
                to the graph constructed by interpreting only the SHAP values and
                the graph constructed after the FCI algorithm, respectively.
        """
        if isinstance(predicted_graph, str):
            if predicted_graph == 'final':
                pred_graph = self.G_final
            elif predicted_graph == 'shap':
                pred_graph = self.G_shap                     # type: ignore
            elif predicted_graph == 'pi':
                pred_graph = self.G_pi                       # type: ignore
            elif predicted_graph == 'indep':
                pred_graph = self.G_indep                    # type: ignore
            else:
                raise ValueError(
                    "Predicted graph must be one of 'final', 'shap' or 'indep'.")
        elif isinstance(predicted_graph, nx.DiGraph):
            pred_graph = predicted_graph

        if ref_graph is None:
            return None

        if pred_graph is None:
            return None

        return evaluate_graph(ref_graph, pred_graph, self.feature_names)

    def compute_regression_quality(self) -> set:
        """
        Compute the regression quality for each feature in the dataset.

        Returns:
            set: A set of features that are considered as root causes.
        """
        assert self.models is not None, "Models is None"
        root_causes = RegQuality.predict(self.models.scoring)
        root_causes = set([self.feature_names[i] for i in root_causes])
        return root_causes

    def summarize_knowledge(self, ref_graph: nx.DiGraph) -> pd.DataFrame:
        """
        Returns a dataframe with the knowledge about each edge in the graph
        The dataframe is obtained from the Knowledge class.

        Parameters:
        -----------
            ref_graph (nx.DiGraph): The reference graph, or ground truth.

        Returns:
            pd.DataFrame: A dataframe with the knowledge about each edge in
            the reference graph and the predicted graph
        """
        if ref_graph is None:
            return None

        self.knowledge = Knowledge(self, ref_graph)
        self.learnings = self.knowledge.info()

        return self.learnings

    def break_cycles(self, dag: nx.DiGraph):
        """ Break a cycle in the given DAG.

        Parameters:
        -----------
            dag (nx.DiGraph): The DAG to break the cycle from.
        """
        assert self.shaps is not None, "ShapEstimator is None"
        return utils.break_cycles_if_present(
            dag, self.shaps.shap_discrepancies, self.prior, verbose=self.verbose)

    def get_prior_from_ref_graph(self, input_path):
        return self._get_prior_from_ref_graph(input_path)

    def _get_prior_from_ref_graph(
        self,
        input_path: str
    ) -> Optional[List[List[str]]]:
        """
        Get the prior from a reference graph.

        Returns
        -------
        List[List[str]]
            A list of two lists of nodes. The first list contains the root nodes,
            and the second list contains the rest of the nodes. The root nodes are
            the nodes with no incoming edges, and the rest of the nodes are the
            ones with at least one incoming edge.

        Raises
        ------
        ValueError
            If the reference graph is not found.

        Notes
        -----
        The reference graph is obtained from the name of the model.

        The root nodes are obtained from the reference graph, and the rest of the
        nodes are obtained from the difference between the nodes of the reference
        graph and the root nodes.

        Returns
        -------
        List[List[str]]
            A list of two lists of nodes. The first list contains the root nodes,
            and the second list contains the rest of the nodes.
        """
        ref_graph_file = self.name.replace(f"_{self.model_type}", "")
        ref_graph_file = f"{ref_graph_file}.dot"
        ref_graph = utils.graph_from_dot_file(os.path.join(
            input_path, ref_graph_file))

        if ref_graph is None:
            print(
                f"WARNING: The reference graph '{ref_graph_file}' was not found.")
            return None

        root_nodes = [node for node,
                      degree in ref_graph.in_degree() if degree == 0]  # type: ignore
        return [root_nodes, [node for node in ref_graph.nodes if node not in root_nodes]]

    def _set_sampling_split(self):
        r"""
        Set the sampling splits for the bootstrap.

        .. math::

            \tau = \frac{s}{m} \ge 1 - p^{\frac{1}{r}} \ge 1-e^{\frac{\ln p}{r}}

        We take \( p=0.01 \), so \( \ln p = -4.605170185988091 \)

        Returns
        -------
        float
        """
        return 1.0 - np.e**(-4.605170185988091 / self.bootstrap_trials)

    def _steps_from_hpo(self, fit_steps) -> int:
        """
        Update the number of trials for the HPO.

        Parameters
        ----------
        fit_steps: Pipeline
            The pipeline where looking for the number of trials.

        Returns
        -------
        int
        """
        if fit_steps.contains_method('tune_fit', exact_match=False):
            if 'hpo_n_trials' in self.kwargs:
                return self.kwargs['hpo_n_trials']
            else:
                if fit_steps.contains_argument('hpo_n_trials'):
                    return fit_steps.get_argument_value('hpo_n_trials')
                else:
                    return DEFAULT_HPO_TRIALS

        return 0

    def _steps_from_bootstrap(self, fit_steps) -> int:
        """
        Check if the pipeline contains a stage called iterative_fit and
        retrieve the number of iterations.

        Parameters
        ----------
        fit_steps: Pipeline
            The pipeline where looking for the number of trials.

        Returns
        -------
        int
        """
        num_iterations = 0
        num_iterative_steps = fit_steps.contains_method(
            'bootstrap', exact_match=False)
        if num_iterative_steps > 0:
            if fit_steps.contains_argument('num_iterations'):
                all_iterations = fit_steps.all_argument_values(
                    'num_iterations')
                if all_iterations != []:
                    return sum(all_iterations)
            else:
                num_iterations += DEFAULT_BOOTSTRAP_TRIALS

        return num_iterations

    def _filter_adjacency_matrix(
            self,
            adjacency_matrix: np.ndarray,
            tolerance: float) -> np.ndarray:
        """
        Given an adjacency matrix, return a filtered version of it, where
        all weights with absolute value less than the tolerance are
        set to zero.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            The adjacency matrix.
        tolerance : float
            The tolerance value.

        Returns
        -------
        np.ndarray
            The filtered adjacency matrix.
        """
        filtered_adjacency = adjacency_matrix.copy()
        filtered_adjacency[np.abs(filtered_adjacency) < tolerance] = 0
        return filtered_adjacency

    def _check_model_and_explainer(self, model_type, explainer):
        """ Check that the explainer is supported for the model type. """
        if (model_type == "nn" and explainer != "gradient"):
            if self.verbose:
                print(
                    f"WARNING: SHAP '{explainer}' not supported for model "
                    f"'{model_type}'. Using 'gradient' instead.")
            self.explainer = "gradient"
        if (model_type == "gbt" and explainer != "explainer"):
            if self.verbose:
                print(
                    f"WARNING: SHAP '{explainer}' not supported for model "
                    f"'{model_type}'. Using 'explainer' instead.")
            self.explainer = "explainer"

    def _more_tags(self):
        return {
            'multioutput_only': True,
            "non_deterministic": True,
            "no_validation": True,
            "poor_score": True,
            "_xfail_checks": {
                "check_methods_sample_order_invariance": "This test shouldn't be running at all!",
                "check_methods_subset_invariance": "This test shouldn't be running at all!",
            }
        }

    def __str__(self):
        return utils.stringfy_object(self)

    def _unused_adjust_discrepancy(self, dag: nx.DiGraph):
        """
        Adjusts the discrepancy in the directed acyclic graph (DAG) by adding new
        edges based on the goodness-of-fit (GOF) R2 values calculated from the
        learning data.

        Args:
            dag (nx.DiGraph): The original DAG.

        Returns:
            nx.DiGraph: The adjusted DAG with new edges added based on GOF values.
        """
        assert self.shaps is not None, "ShapEstimator is None"
        G_adj = dag.copy()
        gof = np.zeros((len(self.feature_names), len(self.feature_names)))

        # Loop through all pairs of nodes where the edge is not present in the graph.
        for origin in self.feature_names:
            for target in self.feature_names:
                if origin == target:
                    continue
                if not G_adj.has_edge(origin, target) and not G_adj.has_edge(target, origin):
                    i = self.feature_names.index(origin)
                    j = self.feature_names.index(target)
                    gof[i, j] = self.shaps.shap_discrepancies[target][origin].shap_gof

        new_edges = set()
        # Loop through the i, j positions in the matrix `gof` that are
        # greater than zero.
        for i, j in zip(*np.where(gof > 0)):
            # If the edge (i, j) is not present in the graph, then add it,
            # but only if position (i, j) is greater than position (j, i).
            if not G_adj.has_edge(self.feature_names[i], self.feature_names[j]) and \
                not G_adj.has_edge(self.feature_names[j], self.feature_names[i]) \
                    and gof[i, j] > 0.0 and gof[j, i] > 0.0:
                if gof[j, i] < gof[i, j]:
                    new_edges.add(
                        (self.feature_names[i], self.feature_names[j]))
        # Add the new edges to the graph `G_adj`, if any.
        if new_edges:
            G_adj.add_edges_from(new_edges)

        G_adj = self.break_cycles(G_adj)

        return G_adj

    def _unused_dag_from_discrepancy(
            self,
            discrepancy_upper_threshold: float = 0.99,
            verbose: bool = False) -> nx.DiGraph:
        """
        Build a directed acyclic graph (DAG) from the discrepancies in the SHAP values.
        The discrepancies are calculated as 1.0 - GoodnessOfFit, so that a low
        discrepancy means that the GoodnessOfFit is close to 1.0, which means that
        the SHAP values are similar.

        Parameters:
        -----------
            discrepancy_upper_threshold (float): The threshold for the discrepancy.
                Default is 0.99, which means that the GoodnessOfFit must be
                at least 0.01.

        Returns:
        --------
            nx.DiGraph: The directed acyclic graph (DAG) built from the discrepancies.
        """
        assert self.shaps is not None, "ShapEstimator is None"

        if verbose:
            print("-----\ndag_from_discrepancies()")

        # Find out what pairs of features have low discrepancy, and add them as edges.
        # A low discrepancy means that 1.0 - GoodnesOfFit is lower than the threshold.
        low_discrepancy_edges = defaultdict(list)
        if verbose:
            print('    ' + ' '.join([f"{f:^5s}" for f in self.feature_names]))
        for child in self.feature_names:
            if verbose:
                print(f"{child}: ", end="")
            for parent in self.feature_names:
                if child == parent:
                    if verbose:
                        print("  X  ", end=" ")
                    continue
                discrepancy = 1. - \
                    self.shaps.shap_discrepancies[child][parent].shap_gof
                if verbose:
                    print(f"{discrepancy:+.2f}", end=" ")
                if discrepancy < discrepancy_upper_threshold:
                    if low_discrepancy_edges[child]:
                        low_discrepancy_edges[child].append(parent)
                    else:
                        low_discrepancy_edges[child] = [parent]
            if verbose:
                print()

        # Build a DAG from the connected features.
        self.G_rho = utils.digraph_from_connected_features(
            self.X, self.feature_names, self.models, low_discrepancy_edges,
            root_causes=self.root_causes, prior=self.prior, verbose=verbose)

        self.G_rho = self.break_cycles(self.G_rho)

        return self.G_rho

    def _unused_custom_pipeline(self, steps):
        """
        Execute a pipeline formed by a list of custom steps previously defined.

        Parameters
        ----------
        steps : list
            A list of tuples with the steps to add to the pipeline.

        Returns
        -------
        self : object
            Returns self.
        """
        # Create the pipeline for the training stages.
        pipeline = Pipeline(
            self,                                           # type: ignore
            description="Custom pipeline",
            prog_bar=self.prog_bar,
            verbose=self.verbose,
            silent=self.silent,
            subtask=True)
        pipeline.from_list(steps)
        pipeline.run()
        pipeline.close()

        return self


def main(dataset_name,
         input_path="/Users/renero/phd/data/RC4/",
         output_path="/Users/renero/phd/output/RC4/",
         load_model: bool = False,
         fit_model: bool = True,
         predict_model: bool = True,
         scale_data: bool = False,
         tune_model: bool = False,
         model_type="nn",
         explainer="gradient",
         save=False):
    """
    Custom main function to run the pipeline with the given dataset.
    Specially useful for testing and debugging.
    """

    ref_graph = utils.graph_from_dot_file(f"{input_path}{dataset_name}.dot")
    data = pd.read_csv(f"{input_path}{dataset_name}.csv")
    if scale_data:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    if load_model:
        rex: Rex = utils.load_experiment(
            f"{dataset_name}_{model_type}", output_path)
    else:
        rex = Rex(
            name=dataset_name, tune_model=tune_model,
            model_type=model_type, explainer=explainer, hpo_n_trials=1)

    if fit_model:
        rex.fit(data)  # , pipeline=".fast_fit_pipeline.yaml")

    if predict_model:
        prior = rex._get_prior_from_ref_graph(input_path)
        rex.predict(data, ref_graph, prior=prior)  # ,
        # pipeline=".fast_predict_pipeline.yaml")

    if save:
        where_to = utils.save_experiment(rex.name, output_path, rex)
        print(f"Saved '{rex.name}' to '{where_to}'")


if __name__ == "__main__":
    main('toy_dataset',
         input_path="/Users/renero/phd/data/",
         model_type="nn",
         explainer="gradient",
         load_model=False,
         fit_model=True,
         predict_model=True,
         scale_data=False,
         tune_model=True,
         save=False)
