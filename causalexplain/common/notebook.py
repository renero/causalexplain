"""
A module to run experiments with the causalexplain package, and simplify the
process of loading and saving experiments in notebooks.

Example:
    >> from causalexplain.common.notebook import Experiment
    >> experiment = Experiment("linear", csv_filename="linear.csv")
    >> rex = experiment.load()

(C) 2023, 2024 J. Renero
"""

import os
import time
import warnings
from os import path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import utils
from ..estimators.cam.cam import CAM
from ..estimators.fci.fci import FCI
from ..estimators.ges.ges import GES
from ..estimators.lingam.lingam import DirectLiNGAM as LiNGAM
from ..estimators.pc.pc import PC
from ..estimators.notears.notears import NOTEARS
from ..estimators.rex.rex import Rex

warnings.filterwarnings('ignore')


# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches


estimators = {
    'rex': Rex,
    'fci': FCI,
    'pc': PC,
    'lingam': LiNGAM,
    'ges': GES,
    'cam': CAM,
    'notears': NOTEARS
}
method_names = ['pc', 'fci', 'ges', 'lingam', 'cam', 'notears', 'rex']
metric_columns = ['method', 'data_type', 'f1', 'precision',
                  'recall', 'aupr', 'Tp', 'Tn', 'Fp', 'Fn', 'shd', 'sid',
                  'n_edges', 'ref_n_edges', 'diff_edges', 'name']
RAW_DAG_NAMES = ['G_shap', 'G_prior', 'G_iter', 'G_iter_prior']
COMBINED_DAG_NAMES = ['un_G_shap', 'in_G_shap',
                      'un_G_prior', 'in_G_prior',
                      'un_G_iter', 'in_G_iter',
                      'un_G_iter_prior', 'in_G_iter_prior']


class BaseExperiment:
    """
    Base class for experiments.

    Args:
    input_path (str): The path to the input data.
    output_path (str): The path to save the experiment output.
    train_anyway (bool, optional): Whether to train the model even if the
        experiment exists. Defaults to False.
    save_anyway (bool, optional): Whether to save the experiment even
        if it exists. Defaults to False.
    train_size (float, optional): The proportion of data to use for training.
        Defaults to 0.9.
    random_state (int, optional): The random state for reproducibility.
        Defaults to 42.
    verbose (bool, optional): Whether to display verbose output.
        Defaults to False.
    """
    model_type: str | None = None

    def __init__(
            self,
            input_path: str,
            output_path: str,
            train_anyway: bool = False,
            save_anyway: bool = False,
            scale: bool = False,
            train_size: float = 0.9,
            random_state: int = 42,
            verbose: bool = False):

        self.input_path = input_path
        self.output_path = output_path
        self.train_anyway = train_anyway
        self.save_anyway = save_anyway
        self.scale = scale
        self.train_size = train_size
        self.random_state = random_state
        self.verbose = verbose
        self._data_ref: pd.DataFrame | None = None
        self._train_idx: pd.Index | None = None
        self._test_idx: pd.Index | None = None
        self._train_data: pd.DataFrame | None = None
        self._test_data: pd.DataFrame | None = None

        # Display Options
        np.set_printoptions(precision=4, linewidth=100)
        pd.set_option('display.precision', 4)
        pd.set_option('display.float_format', '{:.4f}'.format)

    @property
    def data(self) -> pd.DataFrame | None:
        return self._data_ref

    @data.setter
    def data(self, value: pd.DataFrame | None) -> None:
        self._data_ref = value
        # Reset cached splits when the data reference changes.
        self._train_data = None
        self._test_data = None

    @property
    def train_data(self) -> pd.DataFrame:
        if self._train_data is None:
            if self._data_ref is None or self._train_idx is None:
                raise ValueError("Training data is not initialized")
            # Materialize the split lazily to avoid keeping multiple full copies.
            self._train_data = self._data_ref.loc[self._train_idx]
        return self._train_data

    @train_data.setter
    def train_data(self, value: pd.DataFrame) -> None:
        self._train_data = value

    @property
    def test_data(self) -> pd.DataFrame:
        if self._test_data is None:
            if self._data_ref is None or self._test_idx is None:
                raise ValueError("Test data is not initialized")
            self._test_data = self._data_ref.loc[self._test_idx]
        return self._test_data

    @test_data.setter
    def test_data(self, value: pd.DataFrame) -> None:
        self._test_data = value

    def prepare_experiment_input(
            self,
            experiment_filename,
            csv_filename=None,
            dot_filename=None,
            data: pd.DataFrame | None = None,
            data_is_processed: bool = False,
            train_idx: pd.Index | None = None,
            test_idx: pd.Index | None = None):
        """
        - Loads the data and
        - splits it into train and test,
        - scales it
        - loads the reference graph from the dot file, which has to be named
          as the experiment file, with the .dot extension
        """
        self.experiment_name = path.splitext(
            path.basename(experiment_filename))[0]
        if csv_filename is None:
            csv_filename = f"{path.join(self.input_path, self.experiment_name)}.csv"
        if dot_filename is None:
            dot_filename = f"{path.join(self.input_path, self.experiment_name)}.dot"

        if data is None:
            data_ref = pd.read_csv(csv_filename)
            data_ref = data_ref.apply(pd.to_numeric, downcast='float')
        else:
            data_ref = data if data_is_processed else data.apply(
                pd.to_numeric, downcast='float')

        if self.scale:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data_ref)
            data_ref = pd.DataFrame(
                scaled, columns=data_ref.columns, index=data_ref.index)

        # Keep a shared reference to the dataset to avoid unnecessary copies.
        self.data = data_ref

        if train_idx is None or test_idx is None:
            train_idx = data_ref.sample(
                frac=self.train_size, random_state=self.random_state).index
            test_idx = data_ref.index[~data_ref.index.isin(train_idx)]
        self._train_idx = train_idx
        self._test_idx = test_idx

        self.ref_graph = utils.graph_from_dot_file(dot_filename)

    def experiment_exists(self, name):
        """Checks whether the experiment exists in the output path"""
        return os.path.exists(
            os.path.join(self.output_path, f"{os.path.basename(name)}.pickle"))

    def create_estimator(self, estimator_name: str, name: str, **kwargs):
        """
        Dynamically creates an instance of a class based on the estimator name.

        Args:
        estimator_name (str): The name of the estimator (key in the 'estimators'
            dictionary).
        name (str): The name of the estimator instance.
        *args: Variable length argument list to be passed to the class constructor.
        **kwargs: Arbitrary keyword arguments to be passed to the class constructor.

        Returns:
        An instance of the specified class, or None if the class does not exist.
        """
        estimator_class = estimators.get(estimator_name)
        if estimator_class is None:
            print(f"Estimator '{estimator_name}' not found.")
            return None

        # Special case: when estimator is ReX, model_type needs also to be
        # passed to the constructor
        if estimator_name == 'rex':
            kwargs['model_type'] = self.model_type

        return estimator_class(name=name, **kwargs)


class Experiment(BaseExperiment):
    """
    Represents an experiment for causal graph analysis.

    Methods:
        load: Loads the experiment data.
        fit: Fits the experiment data.
        save: Saves the experiment data.
    """

    estimator_name = None
    rex: Rex | None = None
    pc: PC | None = None
    lingam: LiNGAM | None = None
    ges: GES | None = None
    fci: FCI | None = None
    cam: CAM | None = None
    notears: NOTEARS | None = None
    estimator: Any | None = None

    def __init__(
        self,
        experiment_name,
        csv_filename: str|None = None,
        dot_filename: str|None = None,
        data: pd.DataFrame | None = None,
        data_is_processed: bool = False,
        train_idx: pd.Index | None = None,
        test_idx: pd.Index | None = None,
        model_type: str = 'nn',
        input_path="/Users/renero/phd/data/RC4/",
        output_path="/Users/renero/phd/output/RC4/",
        train_size: float = 0.9,
        random_state: int = 42,
        verbose=False
    ):
        """
        Initializes a new instance of the Experiment class.

        Args:
            experiment_name (str): The name of the experiment.
            csv_filename (str, optional): The filename of the CSV file containing
                the data. Defaults to None.
            dot_filename (str, optional): The filename of the DOT file containing
                the causal graph. Defaults to None.
            data (pd.DataFrame, optional): Dataframe to use instead of reading the CSV.
            data_is_processed (bool, optional): Whether data is already numeric and ready.
            train_idx (pd.Index, optional): Precomputed train indices.
            test_idx (pd.Index, optional): Precomputed test indices.
            model_type (str, optional): The type of model to use. Defaults to 'nn'.
                Other options are: 'gbt', 'nn', 'cam', 'pc', 'fci', 'notears',
                'ges' and 'lingam'.
            input_path (str, optional): The path to the input data.
                Defaults to "/Users/renero/phd/data/RC4/".
            output_path (str, optional): The path to save the output.
                Defaults to "/Users/renero/phd/output/RC4/".
            train_size (float, optional): The proportion of data to use for training.
                Defaults to 0.9.
            random_state (int, optional): The random seed for reproducibility.
                Defaults to 42.
            verbose (bool, optional): Whether to print verbose output.
                Defaults to False.
        """

        super().__init__(
            input_path, output_path, train_size=train_size,
            random_state=random_state, verbose=verbose)
        self.model_type = self._check_model_type(model_type)
        self.is_fitted = False
        self.verbose = verbose
        self.dag: Optional[nx.DiGraph] = None
        self.metrics: Any = None
        self.ref_graph: Optional[nx.DiGraph] = None

        # Prepare the input
        self.prepare_experiment_input(
            experiment_name, csv_filename, dot_filename, data=data,
            data_is_processed=data_is_processed,
            train_idx=train_idx, test_idx=test_idx)

    def _check_model_type(self, model_type) -> str:
        """
        Checks if the model type is valid.
        """
        model_type = model_type.lower()
        if model_type in ['dnn', 'nn']:
            model_type = 'nn'
        elif model_type == 'gbt':
            model_type = 'gbt'
        elif model_type in method_names:
            model_type = model_type
        else:
            raise ValueError(
                f"Model type '{model_type}' not supported. "
                f"Supported options are: "
                f"'nn', 'gbt', 'pc', 'fci', 'cam', 'notears', 'ges' and "
                f"'lingam'.")

        return model_type

    def _set_estimator_attr(self, estimator_name: str|None, estimator: Any) -> None:
        """
        Assigns the estimator to a stable, typed attribute.
        """
        assert estimator_name is not None, "estimator_name cannot be None"

        if estimator_name == 'rex':
            self.rex = estimator
        elif estimator_name == 'pc':
            self.pc = estimator
        elif estimator_name == 'lingam':
            self.lingam = estimator
        elif estimator_name == 'ges':
            self.ges = estimator
        elif estimator_name == 'fci':
            self.fci = estimator
        elif estimator_name == 'cam':
            self.cam = estimator
        elif estimator_name == 'notears':
            self.notears = estimator

    def fit(self, estimator_name='rex', **kwargs):
        """
        Fits the experiment data.

        Args:
            **kwargs: Additional keyword arguments to pass to the Rex constructor.

        Returns:
            Rex: The fitted experiment data.
        """
        self.estimator_name = estimator_name
        kwargs['model_type'] = self.model_type

        estimator_object = self.create_estimator(
            estimator_name, name=self.experiment_name, **kwargs)

        if estimator_object is None:
            raise ValueError(f"Estimator '{estimator_name}' not found.")

        pipeline = kwargs.pop('pipeline') if 'pipeline' in kwargs else None

        estimator_object.fit(
            self.train_data, y=self.test_data, pipeline=pipeline)

        setattr(self, estimator_name, estimator_object)
        self.is_fitted = True

        return self

    def predict(self, estimator='rex', **kwargs):
        """
        Predicts with the experiment data.

        Args:
            **kwargs: Additional keyword arguments to pass to the `predict()` method

        Returns:
            Rex: The fitted experiment data.
        """
        if self.estimator_name is None:
            self.estimator_name = 'rex'
        estimator = getattr(self, self.estimator_name)
        estimator.predict(self.train_data, **kwargs)

        return self

    def fit_predict(self, estimator='rex', **kwargs):
        """
        Fits and predicts with the experiment data.

        Args:
            **kwargs: Additional keyword arguments to pass to the Rex constructor.

        Returns:
            Rex: The fitted experiment data.
        """
        start_time = time.time()
        self.estimator_name = estimator

        # Extract from the kwargs the parameter 'prior'
        prior = kwargs.pop('prior') if 'prior' in kwargs else None

        estimator_object = self.create_estimator(
            estimator, name=self.experiment_name, **kwargs)
        fit_predict_args = [self.train_data, self.test_data, self.ref_graph]
        if estimator == 'rex':
            fit_predict_args.append(prior)
        if estimator_object is not None:
            estimator_object.fit_predict(*fit_predict_args)
        else:
            raise ValueError(f"Estimator '{estimator}' not found.")

        setattr(self, estimator, estimator_object)
        end_time = time.time()
        self.fit_predict_time = end_time - start_time

        return self

    def _set_estimator_name(self, exp_object: Any, exp_name: str) -> None:
        # A priori, I don't know which estimator was used to train the experiment
        # so I have to check the type of the object
        if isinstance(exp_object, Rex):
            self.estimator_name = 'rex'
        elif isinstance(exp_object, PC):
            self.estimator_name = 'pc'
        elif isinstance(exp_object, LiNGAM):
            self.estimator_name = 'lingam'
        elif isinstance(exp_object, GES):
            self.estimator_name = 'ges'
        elif isinstance(exp_object, FCI):
            self.estimator_name = 'fci'
        elif isinstance(exp_object, CAM):
            self.estimator_name = 'cam'
        elif isinstance(exp_object, NOTEARS):
            self.estimator_name = 'notears'
        else:
            raise ValueError(
                f"Estimator '{exp_name}' not recognized.")

    def load(self, exp_name=None) -> "Experiment":
        """
        Loads the experiment data.

        Args:
            exp_name (str, optional): The name of the experiment to load.
            If None, loads the current experiment. Defaults to None.

        Returns:
            Rex: The loaded experiment data.
        """

        if exp_name is None:
            exp_name = self.experiment_name

        if self.model_type:
            exp_object = utils.load_experiment(
                f"{exp_name}_{self.model_type}", self.output_path)
        else:
            exp_object = utils.load_experiment(exp_name, self.output_path)

        self._set_estimator_name(exp_object, exp_name)

        # Assign explicitly to keep types stable for static analyzers.
        self._set_estimator_attr(self.estimator_name, exp_object)
        self.estimator = exp_object

        if self.model_type is None:
            self.model_type = 'rex'  # Default model type
        if self.verbose:
            print(f"Loaded '{exp_name}' ({self.model_type.upper()}) "
                  f"from '{self.output_path}'")
            if isinstance(exp_object, Rex):
                fit_time = utils.format_time(exp_object.fit_time)
                predict_time = utils.format_time(exp_object.predict_time)
                print(
                    f"This model took {fit_time[0]:.1f}{fit_time[1]}. to fit, and "
                    f"{predict_time[0]:.1f}{predict_time[1]}. to build predicted DAGs"
                )

        return self

    def save(self, exp_name=None, overwrite: bool = False):
        """
        Saves the experiment data.

        Args:
        -----
        - exp_name (str, optional): The name to save the experiment as.
            If None, uses the experiment name. Defaults to None.
        - overwrite (bool, optional): Whether to overwrite an existing
            experiment with the same name. Defaults to False.
        """
        save_as = exp_name if exp_name is not None else self.experiment_name
        if self.estimator_name is None:
            self.estimator_name = 'rex'
        where_to = utils.save_experiment(
            f"{save_as}_{self.model_type}",
            self.output_path, getattr(self, self.estimator_name),
            overwrite)

        if self.verbose:
            print(f"Saved '{self.experiment_name}' to '{where_to}'")

        return where_to


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=150)
    warnings.filterwarnings('ignore')
    extra_args = {
        'rex': {
            'prog_bar': True,
            'verbose': False,
            'hpo_n_trials': 1,
            'bootstrap_trials': 10,
            'bootstrap_parallel_jobs': -1,
            'parallel_jobs': -1
        },
        'pc': {},
        'ges': {},
        'lingam': {},
        'fci': {},
        'cam': {
            'pruning': True,
            'pruneMethodPars': {"cutOffPVal": 0.05, "numBasisFcts": 10}
        },
        'notears': {}
    }

    input_path = os.path.expanduser("~/phd/data/")
    output_path = os.path.expanduser("~/phd/output/")

    method_name = "rex"
    dataset_name = "toy_dataset"
    # dataset_name =  "generated_10vars_linear_0"

    exp = Experiment(
        experiment_name=dataset_name,
        csv_filename=os.path.join(input_path, f"{dataset_name}.csv"),
        dot_filename=os.path.join(input_path, f"{dataset_name}.dot"),
        model_type="gbt",
        input_path=input_path,
        output_path=output_path)

    exp = exp.fit_predict(method_name, **extra_args[method_name])
    method = getattr(exp, method_name)
    print(method.dag.edges())
    print(method.metrics)
    t, u = utils.format_time(exp.fit_predict_time)
    print(f"Elapsed time: {t:.1f}{u}")
