import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

from causalexplain.causalexplainer import GraphDiscovery
from causalexplain.common import DEFAULT_REGRESSORS


@pytest.fixture
def sample_csv(tmp_path):
    # Create a temporary CSV file
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_dot(tmp_path):
    # Create a temporary DOT file
    dot_content = """
    digraph G {
        A -> B;
        B -> C;
    }
    """
    dot_path = tmp_path / "test_graph.dot"
    with open(dot_path, 'w') as f:
        f.write(dot_content)
    return str(dot_path)


@pytest.fixture
def mock_experiment(monkeypatch):
    """Mock the Experiment class without relying on pytest-mock"""
    mock = MagicMock()
    monkeypatch.setattr('causalexplain.causalexplainer.Experiment', mock)
    return mock


def test_default_initialization():
    """Test initialization with default parameters"""
    gd = GraphDiscovery()
    assert gd.experiment_name is None
    assert gd.estimator == 'rex'
    assert gd.csv_filename is None
    assert gd.dot_filename is None
    assert gd.verbose is False
    assert gd.seed == 42


def test_valid_initialization(sample_csv, sample_dot):
    """Test initialization with valid parameters"""
    gd = GraphDiscovery(
        experiment_name="test_exp",
        model_type="rex",
        csv_filename=sample_csv,
        true_dag_filename=sample_dot,
        verbose=True,
        seed=123
    )
    assert gd.experiment_name == "test_exp"
    assert gd.estimator == "rex"
    assert gd.csv_filename == sample_csv
    assert gd.dot_filename == sample_dot
    assert gd.verbose is True
    assert gd.seed == 123
    assert gd.dataset_name == "test_data"
    assert gd.data_columns == ['A', 'B', 'C']
    assert gd.regressors == DEFAULT_REGRESSORS


def test_initialization_with_non_rex_model(sample_csv, sample_dot):
    """Test initialization with a non-rex model type"""
    gd = GraphDiscovery(
        experiment_name="test_exp",
        model_type="random_forest",
        csv_filename=sample_csv,
        true_dag_filename=sample_dot
    )
    assert gd.regressors == ["random_forest"]


def test_invalid_csv_path(tmp_path):
    """Test initialization with non-existent CSV file"""
    dot_path = tmp_path / "graph.dot"
    dot_path.write_text("digraph G { A -> B; }")
    with pytest.raises(FileNotFoundError):
        GraphDiscovery(
            experiment_name="test_exp",
            model_type="rex",
            csv_filename="non_existent.csv",
            true_dag_filename=str(dot_path)
        )


def test_partial_initialization():
    """Test initialization with partial parameters"""
    with pytest.raises(ValueError) as exc_info:
        GraphDiscovery(experiment_name="test_exp")
    assert "'experiment_name' and 'csv_filename' must be provided" in str(
        exc_info.value)


def test_mismatched_experiment_name_and_csv():
    """Test that providing only experiment_name or only csv_filename 
    raises ValueError"""
    # Test providing only experiment_name
    with pytest.raises(ValueError) as exc_info:
        GraphDiscovery(experiment_name="test_exp", csv_filename=None)
    assert "'experiment_name' and 'csv_filename' must be provided" in str(
        exc_info.value)

    # Test providing only csv_filename
    with pytest.raises(ValueError) as exc_info:
        GraphDiscovery(experiment_name=None, csv_filename="data.csv")
    assert "'experiment_name' and 'csv_filename' must be provided" in str(
        exc_info.value)


def test_none_values():
    """Test initialization with explicit None values"""
    gd = GraphDiscovery(
        experiment_name=None,
        model_type=None,
        csv_filename=None,
        true_dag_filename=None
    )
    assert gd.estimator == 'rex'
    assert gd.csv_filename is None
    assert gd.dot_filename is None
    assert gd.experiment_name is None


def test_empty_string_values():
    """Test initialization with empty string values"""
    gd = GraphDiscovery(experiment_name="", csv_filename="")
    assert gd.experiment_name is None
    assert gd.csv_filename is None


def test_whitespace_string_values():
    """Test initialization with whitespace string values"""
    gd = GraphDiscovery(experiment_name="   ", csv_filename="  ")
    assert gd.experiment_name is None
    assert gd.csv_filename is None


def test_create_experiments_rex(mock_experiment, sample_csv, sample_dot):
    """Test create_experiments with rex model type"""
    gd = GraphDiscovery(
        experiment_name="test_exp",
        model_type="rex",
        csv_filename=sample_csv,
        true_dag_filename=sample_dot
    )
    trainers = gd.create_experiments()

    # For rex, we should have multiple trainers (one for each DEFAULT_REGRESSOR)
    assert len(trainers) == len(DEFAULT_REGRESSORS)

    # Check that Experiment was called correctly for each regressor
    for model_type in DEFAULT_REGRESSORS:
        trainer_name = f"test_data_{model_type}"
        assert trainer_name in trainers
        mock_experiment.assert_any_call(
            experiment_name="test_data",
            csv_filename=sample_csv,
            dot_filename=sample_dot,
            data=gd.data,
            data_is_processed=True,
            train_idx=gd.train_idx,
            test_idx=gd.test_idx,
            model_type=model_type,
            input_path=os.path.dirname(sample_csv),
            output_path=os.getcwd(),
            verbose=False
        )


def test_create_experiments_non_rex(mock_experiment, sample_csv, sample_dot):
    """Test create_experiments with non-rex model type"""
    gd = GraphDiscovery(
        experiment_name="test_exp",
        model_type="lingam",
        csv_filename=sample_csv,
        true_dag_filename=sample_dot
    )
    trainers = gd.create_experiments()

    # For non-rex, we should have only one trainer
    assert len(trainers) == 1
    trainer_name = "test_data_lingam"
    assert trainer_name in trainers

    # Check that Experiment was called correctly
    mock_experiment.assert_called_once_with(
        experiment_name="test_data",
        csv_filename=sample_csv,
        dot_filename=sample_dot,
        data=gd.data,
        data_is_processed=True,
        train_idx=gd.train_idx,
        test_idx=gd.test_idx,
        model_type="lingam",
        input_path=os.path.dirname(sample_csv),
        output_path=os.getcwd(),
        verbose=False
    )


def test_create_experiments_without_initialization():
    """Test create_experiments without proper initialization"""
    gd = GraphDiscovery()
    with pytest.raises(AttributeError):
        gd.create_experiments()


def test_create_experiments_empty_regressors(mock_experiment, sample_csv, sample_dot):
    """Test create_experiments with empty regressors list"""
    gd = GraphDiscovery(
        experiment_name="test_exp",
        model_type="rex",
        csv_filename=sample_csv,
        true_dag_filename=sample_dot
    )
    gd.regressors = []  # Empty the regressors list
    trainers = gd.create_experiments()

    assert len(trainers) == 0
    mock_experiment.assert_not_called()
