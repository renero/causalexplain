"""
A class to train DFF networks for all variables in data. Each network will be trained to
predict one of the variables in the data, using the rest as predictors plus one
source of random noise.

(C) 2022,2023,2024,2025 J. Renero
"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches
# pylint: disable=W0102:dangerous-default-value

import inspect
import warnings
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from mlforge.progbar import ProgBar   # type: ignore

from ..common import DEFAULT_HPO_TRIALS,  utils
from ._columnar import ColumnsDataset
from ._models import MLPModel
from ._optuna_storage import (
    _ensure_writable_optuna_storage,
    _fallback_optuna_storage,
    _is_readonly_storage_error,
)

warnings.filterwarnings("ignore")


class NNRegressor(BaseEstimator):
    """
    A class to train DFF networks for all variables in data. Each network will be
    trained to predict one of the variables in the data, using the rest as predictors
    plus one source of random noise.

    Attributes:
    -----------
        hidden_dim (int): The dimension(s) of the hidden layer(s). This value
            can be a single integer for DFF or an array with the dimension of
            each hidden layer for the MLP case.
        activation (str): The activation function to use, either 'relu' or 'selu'.
            Default is 'relu'.
        learning_rate (float): The learning rate for the optimizer.
        dropout (float): The dropout rate for the dropout layer.
        batch_size (int): The batch size for the optimizer.
        num_epochs (int): The number of epochs for the optimizer.
        loss_fn (str): The loss function to use. Default is "mmd".
        device (str): The device to use. Either "cpu", "cuda", or "mps". Default
            is "auto".
        test_size (float): The proportion of the data to use for testing. Default
            is 0.1.
        seed (int): The seed for the random number generator. Default is 1234.
        early_stop (bool): Whether to use early stopping. Default is True.
        patience (int): The patience for early stopping. Default is 10.
        min_delta (float): The minimum delta for early stopping. Default is 0.001.
        prog_bar (bool): Whether to enable the progress bar. Default
            is False.

    """

    def __init__(
            self,
            hidden_dim: Union[int, List[int]] = [75, 17],
            activation: str = 'relu',
            learning_rate: float = 0.0046,
            dropout: float = 0.001,
            batch_size: int = 44,
            num_epochs: int = 40,
            loss_fn: str = 'mse',
            device: Union[int, str] = "cpu",
            test_size: float = 0.1,
            early_stop: bool = False,
            patience: int = 10,
            min_delta: float = 0.001,
            correlation_th: float|None = None,
            random_state: int = 1234,
            verbose: bool = False,
            prog_bar: bool = True,
            silent: bool = False,
            optuna_prog_bar: bool = False,
            parallel_jobs: int = 0):
        """
        Train DFF networks for all variables in data. Each network will be trained to
        predict one of the variables in the data, using the rest as predictors plus one
        source of random noise.

        Args:
            data (pandas.DataFrame): The dataframe with the continuous variables.
            model_type (str): The type of model to use. Either 'dff' or 'mlp'.
            hidden_dim (int): The dimension(s) of the hidden layer(s). This value
                can be a single integer for DFF or an array with the dimension of
                each hidden layer for the MLP case.
            activation (str): The activation function to use, either 'relu' or 'selu'.
                Default is 'relu'.
            learning_rate (float): The learning rate for the optimizer.
            dropout (float): The dropout rate for the dropout layer.
            batch_size (int): The batch size for the optimizer.
            num_epochs (int): The number of epochs for the optimizer.
            loss_fn (str): The loss function to use. Default is "mmd".
            device (str): The device to use. Either "cpu", "cuda", or "mps". Default
                is "auto".
            test_size (float): The proportion of the data to use for testing. Default
                is 0.1.
            seed (int): The seed for the random number generator. Default is 1234.
            early_stop (bool): Whether to use early stopping. Default is True.
            patience (int): The patience for early stopping. Default is 10.
            min_delta (float): The minimum delta for early stopping. Default is 0.001.
            prog_bar (bool): Whether to enable the progress bar. Default
                is False.
            parallel_jobs (int): Number of parallel jobs to use for CPU training.
                Default is 0 (sequential).

        Returns:
            dict: A dictionary with the trained DFF networks, using the name of the
                variables as the key.
        """
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss_fn = loss_fn
        self.device = device
        self.test_size = test_size
        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        self.correlation_th = correlation_th
        self.random_state = random_state
        self.verbose = verbose
        self.prog_bar = prog_bar
        self.silent = silent
        self.optuna_prog_bar = optuna_prog_bar
        self.parallel_jobs = parallel_jobs

        self.regressor = None
        self._fit_desc = "Training NNs"

        if self.verbose:
            self.prog_bar = False

    def fit(self, X):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        # X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.feature_names = utils.get_feature_names(X)
        self.feature_types = utils.get_feature_types(X)
        self.regressor = {}

        # Who is calling me?
        try:
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            caller_name = calframe[1][3]
            if caller_name == "__call__":
                caller_name = "HPO"
        except Exception:                                # pylint: disable=broad-except
            caller_name = "unknown"

        X_original = X

        loss_fn_by_target = {}
        for target_name in self.feature_names:
            if self.feature_types[target_name] == 'categorical':
                loss_fn_by_target[target_name] = 'crossentropy'
            elif self.feature_types[target_name] == 'binary':
                loss_fn_by_target[target_name] = 'binary_crossentropy'
            else:
                loss_fn_by_target[target_name] = self.loss_fn

        pbar = None
        pbar_name = ""
        if self.prog_bar and not self.verbose:
            pbar_name = f"({caller_name}) DNN_fit"
            pbar = ProgBar().start_subtask(pbar_name, len(self.feature_names))
        else:
            pbar = None

        def _fit_target(target_name: str) -> Tuple[str, MLPModel]:
            X_target = X_original

            model = MLPModel(
                target=target_name,
                input_size=X_target.shape[1],
                activation=self.activation,
                hidden_dim=self.hidden_dim,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                loss_fn=loss_fn_by_target[target_name],
                dropout=self.dropout,
                num_epochs=self.num_epochs,
                dataframe=X_target,
                test_size=self.test_size,
                device=self.device,
                seed=self.random_state,
                early_stop=self.early_stop,
                patience=self.patience,
                min_delta=self.min_delta,
                prog_bar=self.prog_bar)
            model.train()
            return target_name, model

        use_parallel = self.parallel_jobs and self.parallel_jobs > 1
        use_parallel = use_parallel and str(self.device).lower() == "cpu"

        if use_parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            results: Dict[str, MLPModel] = {}
            with ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
                futures = [
                    executor.submit(_fit_target, name)
                    for name in self.feature_names
                ]
                for idx, future in enumerate(as_completed(futures)):
                    target_name, model = future.result()
                    results[target_name] = model
                    pbar.update_subtask(pbar_name, idx + 1) if pbar else None
            for target_name in self.feature_names:
                self.regressor[target_name] = results[target_name]
        else:
            for target_idx, target_name in enumerate(self.feature_names):
                target_name, model = _fit_target(target_name)
                self.regressor[target_name] = model
                pbar.update_subtask(pbar_name, target_idx+1) if pbar else None

        pbar.remove(pbar_name) if pbar else None
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the values for each target variable.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to make predictions on.

        Returns
        -------
        np.ndarray
            The predictions for each target variable.
        """
        assert X.shape[1] == self.n_features_in_, \
            f"X has {X.shape[1]} features, expected {self.n_features_in_}"

        # X = check_array(X, accept_sparse=True)
        if not self.is_fitted_:
            raise ValueError("This Rex instance is not fitted yet. \
                Call 'fit' with appropriate arguments before using this estimator.")

        loaders = {
            target: DataLoader(
                ColumnsDataset(target, X),
                batch_size=self.batch_size,
                shuffle=False)
            for target in self.feature_names
        }

        prediction = pd.DataFrame(columns=self.feature_names)
        for target in self.feature_names:
            # Creat a data loader for the target variable. The ColumnsDataset will
            # return the target variable as the second element of the tuple, and
            # will drop the target from the features.
            if loaders is not None:
                loader = loaders[target]
            else:
                loader = DataLoader(
                    ColumnsDataset(target, X), batch_size=self.batch_size,
                    shuffle=False)
            model = self.regressor[target].model

            # Obtain the predictions for the target variable
            preds = np.empty((0,), dtype=np.float32)
            for (tensor_X, _) in loader:
                tensor_X = tensor_X.to(self.device).float()
                model = model.to(tensor_X.device).float()
                y_hat = model.forward(tensor_X)
                preds = np.append(
                    preds,
                    y_hat.detach().cpu().numpy().astype(
                        np.float32, copy=False).flatten(),
                )
            prediction[target] = preds

        # Concatenate the numpy array for all the batchs
        np_preds = prediction.values
        final_preds = []
        if np_preds.ndim > 1 and np_preds.shape[0] > 1:
            for i in range(len(self.feature_names)):
                column = np_preds[:, i]
                if column.ndim == 1:
                    final_preds.append(column)
                else:
                    final_preds.append(np.concatenate(column))
            final_preds = np.array(final_preds)
        else:
            final_preds = np_preds

        # If final_preds is still 1D, reshape it to 2D
        if final_preds.ndim == 1:
            final_preds = final_preds.reshape(1, -1)

        if len(final_preds) == 0:
            final_preds = np_preds

        return np.asarray(final_preds, dtype=np.float32)

    def score(self, X):
        """
        Scores the model using the loss function. It returns the list of losses
        for each target variable.
        """
        assert X.shape[1] == self.n_features_in_, \
            f"X has {X.shape[1]} features, expected {self.n_features_in_}"
        if not self.is_fitted_:
            raise ValueError("This Rex instance is not fitted yet. \
                Call 'fit' with appropriate arguments before using this estimator.")

        # Call the class method to predict the values for each target variable
        y_hat = self.predict(X)

        # Handle the case where the prediction returned by the model is not a
        # numpy array but a numpy object type
        if isinstance(y_hat, np.ndarray) and y_hat.dtype == np.object_:
            y_hat = np.vstack(y_hat[:, :].flatten()).astype(np.float32)
            y_hat = torch.as_tensor(y_hat, dtype=torch.float32)
        scores = []
        for i, target in enumerate(self.feature_names):
            y_preds = torch.as_tensor(y_hat[i], dtype=torch.float32)
            y = torch.as_tensor(X[target].values, dtype=torch.float32)
            scores.append(self.regressor[target].model.loss_fn(y_preds, y))

        self.scoring = np.array(scores)
        return self.scoring

    def __repr__(self):
        """
        Return a readable snapshot of user-facing attributes.

        Args:
            None.

        Returns:
            str: A formatted summary of non-callable attributes.
        """
        attrs = [
            "hidden_dim",
            "activation",
            "learning_rate",
            "dropout",
            "batch_size",
            "num_epochs",
            "loss_fn",
            "device",
            "test_size",
            "early_stop",
            "patience",
            "min_delta",
            "correlation_th",
            "random_state",
            "verbose",
            "prog_bar",
            "silent",
            "optuna_prog_bar",
            "parallel_jobs",
        ]
        parts = []
        for attr in attrs:
            if hasattr(self, attr):
                parts.append(f"{attr}={getattr(self, attr)!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

        # forbidden_attrs = [
        #     'fit', 'predict', 'score', 'get_params', 'set_params']
        # ret = f"REX object attributes\n"
        # ret += f"{'-'*80}\n"
        # for attr in dir(self):
        #     if attr.startswith('_') or \
        #         attr in forbidden_attrs or \
        #             type(getattr(self, attr)) == types.MethodType:
        #         continue
        #     elif attr == "X" or attr == "y":
        #         if isinstance(getattr(self, attr), pd.DataFrame):
        #             ret += f"{attr:25} {getattr(self, attr).shape}\n"
        #             continue
        #     elif isinstance(getattr(self, attr), pd.DataFrame):
        #         ret += f"{attr:25} DataFrame {getattr(self, attr).shape}\n"
        #     else:
        #         ret += f"{attr:25} {getattr(self, attr)}\n"

        # return ret

    def tune(
            self,
            training_data: pd.DataFrame,
            test_data: pd.DataFrame,
            study_name: str|None = None,
            min_loss: float = 0.05,
            storage: str = 'sqlite:///rex_tuning.db',
            load_if_exists: bool = True,
            n_trials: int = DEFAULT_HPO_TRIALS) -> Dict[str, Any]:
        """
        Tune the hyperparameters of the model using Optuna.
        """
        class Objective:
            """
            A class to define the objective function for the hyperparameter optimization
            Some of the parameters for NNRegressor have been taken to default values to
            reduce the number of hyperparameters to optimize.

            Include this class in the hyperparameter optimization as follows:

            >>> study = optuna.create_study(direction='minimize',
            >>>                             study_name='study_name_here',
            >>>                             storage='sqlite:///db.sqlite3',
            >>>                             load_if_exists=True)
            >>> study.optimize(Objective(train_data, test_data), n_trials=100)

            The only dependency is you need to pass the train and test data to the class
            constructor. Tha class will build the data loaders for them from the
            dataframes.
            """

            def __init__(
                    self,
                    train_data,
                    test_data,
                    device='cpu',
                    prog_bar=True,
                    verbose=False):
                """
                Initialize the Optuna objective with training data and settings.

                Args:
                    train_data (pd.DataFrame): Training dataset.
                    test_data (pd.DataFrame): Validation dataset.
                    device (str): Device to run the model on.
                    prog_bar (bool): Whether to show a progress bar.
                    verbose (bool): Whether to enable verbose logging.

                Returns:
                    None: This method does not return a value.
                """
                self.train_data = train_data
                self.test_data = test_data
                self.device = device

                self.n_layers = None
                self.activation = None
                self.learning_rate = None
                self.dropout = None
                self.batch_size = None
                self.num_epochs = None
                self.models = None
                self.prog_bar = prog_bar
                self.verbose = verbose

            def __call__(self, trial):
                """
                This method is called by Optuna to evaluate the objective function.
                """
                self.n_layers = trial.suggest_int('n_layers', 1, 6)
                self.layers = []
                for i in range(self.n_layers):
                    self.layers.append(
                        trial.suggest_int(f'n_units_l{i}', 1, 182))
                self.activation = trial.suggest_categorical(
                    'activation', ['relu', 'selu', 'linear'])
                self.learning_rate = trial.suggest_loguniform(
                    'learning_rate', 1e-5, 1e-1)
                self.dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
                self.batch_size = trial.suggest_int('batch_size', 8, 128)
                self.num_epochs = trial.suggest_int('num_epochs', 10, 80)

                self.models = NNRegressor(
                    hidden_dim=self.layers,
                    activation=self.activation,
                    learning_rate=self.learning_rate,
                    dropout=self.dropout,
                    batch_size=self.batch_size,
                    num_epochs=self.num_epochs,
                    loss_fn='mse',
                    device=self.device,
                    random_state=42,
                    verbose=self.verbose,
                    prog_bar=True & (not self.verbose) & (self.prog_bar),
                    silent=True)

                self.models.fit(self.train_data)

                # Now, measure the performance of the model with the test data.
                loss = []
                for target in list(self.train_data.columns):
                    model = self.models.regressor[target].model
                    loader = DataLoader(
                        ColumnsDataset(target, self.test_data),
                        batch_size=self.batch_size,
                        shuffle=False)
                    avg_loss, _, _ = self.compute_loss(model, loader)
                    loss.append(avg_loss)

                return np.median(loss)

            def compute_loss(
                    self,
                    model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    n_repeats: int = 10) -> tuple[float, float, np.ndarray[Any] | np.ndarray[Any]]:
                """
                Computes the average MSE loss for a given model and dataloader.

                Parameters:
                -----------
                model: torch.nn.Module
                    The model to compute the loss for.
                dataloader: torch.utils.data.DataLoader
                    The dataloader to use for computing the loss.
                shuffle: int
                    If > 0, the column of the input data to shuffle.

                Returns:
                --------
                avg_loss: float
                    The average MSE loss.
                std_loss: float
                    The standard deviation of the MSE loss.
                losses: np.ndarray
                    The MSE loss for each batch.
                """
                mse = np.array([])
                num_batches = 0
                model = model.to(self.device)
                for _ in range(n_repeats):
                    loss = []
                    for _, (X, y) in enumerate(dataloader):
                        X = X.to(self.device)
                        y = y.to(self.device)
                        yhat = model.forward(X)
                        loss.append(model.loss_fn(yhat, y).item())
                        num_batches += 1
                    if len(mse) == 0:
                        mse = np.array(loss)
                    else:
                        mse = np.vstack((mse, [loss]))

                return np.mean(mse), np.std(mse), mse

        # Callback to stop the study if the MSE is below a threshold.
        def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            """
            Stop the Optuna study when the loss threshold is reached.

            Args:
                study (optuna.study.Study): The active Optuna study.
                trial (optuna.trial.FrozenTrial): The completed trial.

            Returns:
                None: This method does not return a value.
            """
            if trial.value < min_loss or study.best_value < min_loss:
                study.stop()

        if self.verbose is False:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create and run the HPO study.
        resolved_storage = _ensure_writable_optuna_storage(storage, study_name)
        fallback_storage = _fallback_optuna_storage(storage, study_name)
        try:
            study = optuna.create_study(
                direction='minimize', study_name=study_name,
                storage=resolved_storage, load_if_exists=load_if_exists)
            study.optimize(
                Objective(
                    training_data, test_data, device=self.device,
                    prog_bar=self.prog_bar, verbose=self.verbose),
                n_trials=n_trials,
                show_progress_bar=(self.optuna_prog_bar & (
                    not self.silent) & (not self.verbose)),
                callbacks=[callback]
            )
        except Exception as exc:  # pylint: disable=broad-except
            if not _is_readonly_storage_error(exc):
                raise
            if fallback_storage == resolved_storage:
                raise
            if self.verbose and not self.silent:
                print(
                    "Optuna storage is read-only; retrying with "
                    f"storage={fallback_storage}")
            study = optuna.create_study(
                direction='minimize', study_name=study_name,
                storage=fallback_storage, load_if_exists=load_if_exists)
            study.optimize(
                Objective(
                    training_data, test_data, device=self.device,
                    prog_bar=self.prog_bar, verbose=self.verbose),
                n_trials=n_trials,
                show_progress_bar=(self.optuna_prog_bar & (
                    not self.silent) & (not self.verbose)),
                callbacks=[callback]
            )

        # Capture the best parameters and the minimum loss.
        best_trials = sorted(study.best_trials, key=lambda x: x.values[0])
        self.best_params = best_trials[0].params
        self.min_tunned_loss = best_trials[0].values[0]

        if self.verbose and not self.silent:
            print(
                f"          > Best params (min loss:{self.min_tunned_loss:.6f}):")
            for k, v in self.best_params.items():
                print(f"            > {k:<15s}: {v}")

        regressor_args = {
            'hidden_dim': [self.best_params[f'n_units_l{i}']
                           for i in range(self.best_params['n_layers'])],
            'activation': self.best_params['activation'],
            'learning_rate': self.best_params['learning_rate'],
            'dropout': self.best_params['dropout'],
            'batch_size': self.best_params['batch_size'],
            'num_epochs': self.best_params['num_epochs']}

        return regressor_args

    def tune_fit(
            self,
            X: pd.DataFrame,
            hpo_study_name: str = None,
            hpo_min_loss: float = 0.05,
            hpo_storage: str = 'sqlite:///rex_tuning.db',
            hpo_load_if_exists: bool = True,
            hpo_n_trials: int = DEFAULT_HPO_TRIALS):
        """
        Tune the hyperparameters of the model using Optuna, and the fit the model
        with the best parameters.
        """
        # split X into train and test
        train_data = X.sample(frac=0.9, random_state=self.random_state)
        test_data = X.drop(train_data.index)

        # tune the model
        regressor_args = self.tune(
            train_data, test_data, n_trials=hpo_n_trials, study_name=hpo_study_name,
            min_loss=hpo_min_loss, storage=hpo_storage,
            load_if_exists=hpo_load_if_exists)

        if self.verbose and not self.silent:
            print(
                f"          > Best params (min loss:{self.min_tunned_loss:.6f}):")
            for k, v in regressor_args.items():
                print(f"            > {k:<15s}: {v}")

        # Set the object parameters to the best parameters found.
        for k, v in regressor_args.items():
            setattr(self, k, v)

        # Fit the model with the best parameters.
        self.fit(X)


#
# Main function
#

def custom_main(score: bool = False, tune: bool = False):
    """
    Run a small local workflow for scoring or tuning.

    Args:
        score (bool): Whether to load and score an existing model.
        tune (bool): Whether to run hyperparameter tuning.

    Returns:
        None: This function does not return a value.
    """
    import os
    from causalexplain.common import utils
    path = "/Users/renero/phd/data/RC4/risks"
    output_path = "/Users/renero/phd/output/RC4/"
    experiment_name = 'transformed_data'

    # ref_graph = utils.graph_from_dot_file(f"{path}{experiment_name}.dot")

    data = pd.read_csv(f"{os.path.join(path, experiment_name)}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Split the dataframe into train and test
    train = data.sample(frac=0.9, random_state=42)
    test = data.drop(train.index)

    if score:
        rex = utils.load_experiment(f"{experiment_name}", output_path)
        rex.is_fitted_ = True
        print(f"Loaded experiment {experiment_name}")
        rex.models.score(test)
    elif tune:
        nn = NNRegressor()
        nn.tune_fit(data, hpo_n_trials=10)


if __name__ == "__main__":
    custom_main(tune=True)
