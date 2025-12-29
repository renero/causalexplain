import argparse
import json
import os
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import networkx as nx
import pandas as pd
import pytest

import causalexplain.causalexplainer as gd_module

sys.modules.setdefault("causalexplainer", gd_module)

from causalexplain import __main__ as main_mod  # noqa: E402
from causalexplain.causalexplainer import GraphDiscovery  # noqa: E402


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({'X': [1, 2], 'Y': [3, 4], 'Z': [5, 6]})
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def args_factory():
    def _factory(**overrides):
        base = argparse.Namespace(
            dataset=None,
            method='rex',
            true_dag=None,
            load_model=None,
            no_train=False,
            threshold=None,
            combine=None,
            iterations=None,
            bootstrap=None,
            prior=None,
            seed=None,
            quiet=False,
            verbose=False,
            save_model=None,
            output=None
        )
        for key, value in overrides.items():
            setattr(base, key, value)
        return base
    return _factory


def test_parse_args_combine_option(monkeypatch):
    argv = [
        "prog",
        "-d", "data.csv",
        "-c", "intersection",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = main_mod.parse_args()
    assert args.combine == "intersection"
    assert args.dataset == "data.csv"


def test_check_args_requires_dataset_or_model(args_factory):
    args = args_factory()
    with pytest.raises(ValueError):
        main_mod.check_args_validity(args)


def test_check_args_with_dataset_and_save_defaults(
        sample_csv, args_factory, monkeypatch):
    monkeypatch.setattr(
        main_mod.utils, "valid_output_name",
        lambda filename, path: os.path.join(path, "unique_name.pickle"))
    args = args_factory(dataset=sample_csv, save_model='', seed=7)
    run_values = main_mod.check_args_validity(args)
    assert run_values['dataset_name'] == "sample"
    assert run_values['dataset_filepath'] == sample_csv
    assert run_values['seed'] == 7
    assert run_values['save_model'] == os.path.basename(
        sample_csv).replace('.csv', '') + "_rex.pickle"
    assert run_values['output_path'] == os.getcwd()
    assert run_values['bootstrap_iterations'] == main_mod.DEFAULT_BOOTSTRAP_TRIALS


def test_check_args_load_model_without_dataset(tmp_path, args_factory, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"data")
    args = args_factory(load_model="model.pkl", no_train=True)
    run_values = main_mod.check_args_validity(args)
    assert run_values['dataset_name'] is None
    assert run_values['load_model'] == str(model_path)


def test_check_args_fails_when_load_model_missing(tmp_path, args_factory):
    args = args_factory(load_model=str(
        tmp_path / "missing.pkl"), no_train=True)
    with pytest.raises(FileNotFoundError):
        main_mod.check_args_validity(args)


def test_check_args_validates_method(sample_csv, args_factory):
    args = args_factory(dataset=sample_csv, method='invalid')
    with pytest.raises(AssertionError):
        main_mod.check_args_validity(args)


def test_check_args_handles_true_dag_and_prior(
        sample_csv, args_factory, tmp_path, monkeypatch):
    dot_file = tmp_path / "truth.dot"
    dot_file.write_text("digraph G { A -> B; }")
    prior_file = tmp_path / "prior.json"
    prior_data = [["A", "B"]]
    prior_file.write_text(json.dumps({"prior": prior_data}))
    args = args_factory(
        dataset=sample_csv,
        method='pc',
        true_dag=str(dot_file),
        prior=str(prior_file))
    run_values = main_mod.check_args_validity(args)
    assert run_values['true_dag'] == str(dot_file)
    assert run_values['prior'] == prior_data


def test_header_prints_banner(capsys):
    main_mod.header_()
    output = capsys.readouterr().out
    assert main_mod.HEADER_ASCII.strip() in output


def test_show_run_values_outputs_dataframe_shape(capsys):
    run_values = {'data': pd.DataFrame({'a': [1], 'b': [2]}), 'value': 5}
    main_mod.show_run_values(run_values)
    captured = capsys.readouterr().out
    assert "1x2 DataFrame" in captured
    assert "- value: 5" in captured


def test_main_trains_and_saves(monkeypatch, tmp_path):
    class DummyDiscovery:
        def __init__(self, **kwargs):
            self.init_args = kwargs
            self.trainer = {'initial': SimpleNamespace(
                dag="start", metrics=None)}
            self.saved = None
            self.printed = None

        def create_experiments(self):
            self.created = True

        def fit_experiments(self, *args, **kwargs):
            self.fit_args = (args, kwargs)

        def combine_and_evaluate_dags(self, prior, combine_op='union'):
            self.combined_prior = prior
            self.combined_op = combine_op
            return SimpleNamespace(dag="final_dag", metrics="final_metrics")

        def save_model(self, path):
            self.saved = path

        def printout_results(self, dag, metrics, combine_op='union'):
            self.printed = (dag, metrics, combine_op)

    run_values = {
        'dataset_name': 'sample',
        'estimator': 'rex',
        'dataset_filepath': 'data.csv',
        'true_dag': None,
        'verbose': False,
        'seed': 7,
        'load_model': None,
        'no_train': False,
        'hpo_iterations': 3,
        'bootstrap_iterations': 4,
        'prior': [['A', 'B']],
        'combine_op': 'union',
        'output_path': str(tmp_path),
        'model_filename': str(tmp_path / "saved.pkl"),
        'output_dag_file': str(tmp_path / "dag.dot"),
    }
    times = iter([100.0, 101.0])
    monkeypatch.setattr(main_mod.time, "time", lambda: next(times))
    monkeypatch.setattr(main_mod.utils, "format_time",
                        lambda delta: (delta, "seconds"))
    saved_paths = []
    monkeypatch.setattr(main_mod.utils, "graph_to_dot_file",
                        lambda dag, path: saved_paths.append((dag, path)))
    instances = []

    def factory(**kwargs):
        inst = DummyDiscovery(**kwargs)
        instances.append(inst)
        return inst

    monkeypatch.setattr(main_mod, "GraphDiscovery", factory)
    monkeypatch.setattr(main_mod, "parse_args", lambda: SimpleNamespace())
    monkeypatch.setattr(main_mod, "check_args_validity", lambda _: run_values)
    main_mod.main()
    dummy = instances[0]
    assert dummy.saved == run_values['model_filename']
    assert saved_paths[0][1] == run_values['output_dag_file']


def test_main_loads_existing_model(monkeypatch):
    class DummyDiscovery:
        def __init__(self, **kwargs):
            self.trainer = {'one': SimpleNamespace(
                dag='loaded', metrics='metrics')}

        def load_model(self, path):
            self.loaded = path

        def fit_experiments(self, *args):
            pass

        def combine_and_evaluate_dags(self, prior, combine_op='union'):
            return SimpleNamespace(dag="combined", metrics="metrics")

        def printout_results(self, dag, metrics, combine_op='union'):
            self.printed = (dag, metrics, combine_op)

        def save_model(self, path):
            self.saved = path

    run_values = {
        'dataset_name': 'sample',
        'estimator': 'rex',
        'dataset_filepath': None,
        'true_dag': None,
        'verbose': False,
        'seed': 0,
        'load_model': 'model.pkl',
        'no_train': True,
        'hpo_iterations': 0,
        'bootstrap_iterations': 0,
        'prior': None,
        'combine_op': 'union',
        'output_path': None,
        'model_filename': None,
        'output_dag_file': None,
    }
    instances = []

    def factory(**kwargs):
        inst = DummyDiscovery(**kwargs)
        instances.append(inst)
        return inst

    monkeypatch.setattr(main_mod, "GraphDiscovery", factory)
    monkeypatch.setattr(main_mod, "parse_args", lambda: SimpleNamespace())
    monkeypatch.setattr(main_mod, "check_args_validity", lambda _: run_values)
    monkeypatch.setattr(main_mod.time, "time", lambda: 0.0)
    monkeypatch.setattr(main_mod.utils, "format_time",
                        lambda delta: (delta, "seconds"))
    main_mod.main()
    assert instances[0].loaded == 'model.pkl'


def make_graph_discovery(sample_csv, tmp_path, model_type='rex'):
    return GraphDiscovery(
        experiment_name="exp",
        model_type=model_type,
        csv_filename=sample_csv,
        true_dag_filename=None,
        verbose=False,
        seed=1
    )


def test_fit_experiments_non_rex_calls_fit(sample_csv, tmp_path):
    gd = make_graph_discovery(sample_csv, tmp_path, model_type='pc')
    trainer = MagicMock()
    gd.trainer = {f"{gd.dataset_name}_pc": trainer}
    gd.fit_experiments(hpo_iterations=5, bootstrap_iterations=6, extra="value")
    trainer.fit_predict.assert_called_once_with(
        estimator='pc', verbose=False, extra="value")


def test_fit_experiments_rex_skips_rex_named_entries(sample_csv, tmp_path):
    gd = make_graph_discovery(sample_csv, tmp_path)
    rex_trainer = MagicMock()
    other_trainer = MagicMock()
    gd.trainer = {"exp_rex": rex_trainer, "exp_alt": other_trainer}
    gd.fit_experiments(hpo_iterations=2, bootstrap_iterations=3)
    other_trainer.fit_predict.assert_called_once()
    rex_trainer.fit_predict.assert_not_called()


def test_combine_and_evaluate_non_rex_sets_metrics(sample_csv, tmp_path, monkeypatch):
    gd = make_graph_discovery(sample_csv, tmp_path, model_type='pc')
    gd.ref_graph = nx.DiGraph()
    gd.data_columns = ['X', 'Y']
    trainer = SimpleNamespace(pc=SimpleNamespace(
        dag="dag"), dag=None, metrics=None)
    gd.trainer = {f"{gd.dataset_name}_pc": trainer}
    monkeypatch.setattr(
        "causalexplain.causalexplainer.evaluate_graph",
        lambda ref, dag, cols: {"sid": 0})
    result = gd.combine_and_evaluate_dags()
    assert result.dag == "dag"
    assert gd.metrics == {"sid": 0}


def test_combine_and_evaluate_rex_combines(sample_csv, tmp_path, monkeypatch):
    gd = make_graph_discovery(sample_csv, tmp_path)

    class Estimator:
        def __init__(self, label):
            self.dag = f"dag_{label}"
            self.shaps = SimpleNamespace(shap_discrepancies=f"disc_{label}")
    gd.trainer = {
        "exp_a": SimpleNamespace(rex=Estimator('a')),
        "exp_b": SimpleNamespace(rex=Estimator('b')),
    }
    monkeypatch.setattr(
        "causalexplain.causalexplainer.utils.combine_dags",
        lambda *args, **kwargs: (None, None, "combined", None))
    result = gd.combine_and_evaluate_dags(prior=[['A', 'B']])
    assert result.dag == "combined"
    assert gd.dag == "combined"


def test_run_invokes_sequence(sample_csv, tmp_path):
    gd = make_graph_discovery(sample_csv, tmp_path)
    gd.create_experiments = MagicMock()
    gd.fit_experiments = MagicMock()
    gd.combine_and_evaluate_dags = MagicMock()
    gd.run(5, 6, prior=[['A', 'B']], option=True)
    gd.create_experiments.assert_called_once()
    gd.fit_experiments.assert_called_once()
    gd.combine_and_evaluate_dags.assert_called_once_with(
        prior=[['A', 'B']], combine_op='union')


def test_save_validates_state(sample_csv, tmp_path):
    gd = make_graph_discovery(sample_csv, tmp_path)
    gd.trainer = {}
    with pytest.raises(AssertionError):
        gd.save_model(str(tmp_path / "model.pkl"))


def test_save_writes_model(sample_csv, tmp_path, monkeypatch):
    gd = make_graph_discovery(sample_csv, tmp_path)
    gd.trainer = {"t": SimpleNamespace()}
    saved = {}
    monkeypatch.setattr(
        "causalexplain.causalexplainer.utils.save_experiment",
        lambda name, path, trainer, overwrite: saved.setdefault("path", os.path.join(path, name)))
    gd.save_model(str(tmp_path / "model.pkl"))
    assert saved["path"].endswith("model.pkl")


def test_load_sets_properties(tmp_path):
    gd = GraphDiscovery()
    trainer_data = {'a': SimpleNamespace(dag="dag", metrics="metrics")}
    model_path = tmp_path / "trainer.pkl"
    with open(model_path, 'wb') as handle:
        pickle.dump(trainer_data, handle)
    loaded = gd.load_model(str(model_path))
    assert gd.dag == "dag"
    assert loaded == trainer_data


def test_printout_results_handles_empty_graph(capsys):
    gd = GraphDiscovery()
    graph = nx.DiGraph()
    gd.printout_results(graph, None, 'union')
    assert "Empty graph" in capsys.readouterr().out


def test_printout_results_lists_edges(capsys):
    gd = GraphDiscovery()
    graph = nx.DiGraph()
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    gd.printout_results(graph, "metrics", 'union')
    output = capsys.readouterr().out
    assert "A -> B" in output and "Graph Union Metrics" in output


def test_export_delegates_to_utils(sample_csv, tmp_path, monkeypatch):
    gd = make_graph_discovery(sample_csv, tmp_path, model_type='pc')
    gd.trainer = {'a': SimpleNamespace(dag="dag")}
    exported = {}
    monkeypatch.setattr(
        "causalexplain.causalexplainer.utils.graph_to_dot_file",
        lambda dag, path: exported.setdefault("path", path))
    result = gd.export_dag("file.dot")
    assert result == "file.dot"
    assert exported["path"] == "file.dot"


def test_plot_calls_plot_module(sample_csv, tmp_path, monkeypatch):
    gd = make_graph_discovery(sample_csv, tmp_path, model_type='pc')
    gd.trainer = {'a': SimpleNamespace(dag="dag", ref_graph="ref")}
    called = {}
    monkeypatch.setattr(
        "causalexplain.causalexplainer.plot.dag",
        lambda **kwargs: called.setdefault("kwargs", kwargs))
    gd.plot(show_metrics=True, layout='circular')
    assert called["kwargs"]["graph"] == "dag"
    assert called["kwargs"]["layout"] == 'circular'


def test_model_property_returns_last_trainer(sample_csv, tmp_path):
    gd = make_graph_discovery(sample_csv, tmp_path, model_type='pc')
    gd.trainer = {'first': SimpleNamespace(
    ), 'second': SimpleNamespace(marker=True)}
    assert gd.model.marker is True
