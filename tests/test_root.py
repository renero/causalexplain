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
            command='run',
            dataset=None,
            generate_dataset=False,
            mechanism=None,
            variables=None,
            samples=None,
            generate_output=None,
            timeout=30,
            max_retries=50,
            min_edges=0,
            max_edges=30,
            max_parents=3,
            rescale=True,
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
            output=None,
            adaptive_shap_sampling=True,
            cuda=False,
            mps=False,
            parallel_jobs=0,
            bootstrap_parallel_jobs=0,
            gui=False,
            bootstrap_shap_cache=True,
            precompute_target_matrices=False,
            hpo_optimization=False,
            hpo_optimization_limit=None,
            shap_budget=None,
            max_shap_samples=None,
            export_diagnostics=None,
            compat_warning=None,
        )
        for key, value in overrides.items():
            setattr(base, key, value)
        return base
    return _factory


def test_parse_args_combine_option(monkeypatch):
    argv = [
        "prog",
        "run",
        "-d", "data.csv",
        "-c", "intersection",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = main_mod.parse_args()
    assert args.command == "run"
    assert args.combine == "intersection"
    assert args.dataset == "data.csv"


def test_parse_args_adaptive_shap_sampling(monkeypatch):
    argv = [
        "prog",
        "run",
        "-d", "data.csv",
        "--no-shap-sampling",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = main_mod.parse_args()
    assert args.adaptive_shap_sampling is False


def test_parse_args_cuda_flag(monkeypatch):
    argv = [
        "prog",
        "run",
        "-d", "data.csv",
        "--cuda",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = main_mod.parse_args()
    assert args.cuda is True
    assert args.mps is False


def test_parse_args_mps_flag(monkeypatch):
    argv = [
        "prog",
        "run",
        "-d", "data.csv",
        "--mps",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = main_mod.parse_args()
    assert args.mps is True
    assert args.cuda is False


def test_parse_args_parallel_jobs(monkeypatch):
    argv = [
        "prog",
        "run",
        "-d", "data.csv",
        "--parallel-jobs", "3",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = main_mod.parse_args()
    assert args.parallel_jobs == 3


def test_parse_args_export_diagnostics(monkeypatch):
    argv = [
        "prog",
        "run",
        "-d", "data.csv",
        "--export-diagnostics", "diagnostics/report",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = main_mod.parse_args()
    assert args.export_diagnostics == "diagnostics/report"


def test_parse_args_bootstrap_parallel_jobs(monkeypatch):
    argv = [
        "prog",
        "run",
        "-d", "data.csv",
        "--bootstrap-parallel-jobs", "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = main_mod.parse_args()
    assert args.bootstrap_parallel_jobs == 2


def test_parse_args_no_subcommand(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = main_mod.parse_args()
    assert args.command is None
    assert args.compat_warning is None


def test_parse_args_generate_dataset(monkeypatch):
    argv = [
        "prog",
        "generate",
        "--mechanism", "linear",
        "--variables", "5",
        "--samples", "20",
        "--output", "generated/toy_dataset",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = main_mod.parse_args()
    assert args.command == "generate"
    assert args.mechanism == "linear"
    assert args.variables == 5
    assert args.samples == 20
    assert args.output == "generated/toy_dataset"
    assert args.compat_warning is None


def test_parse_args_gui_subcommand(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "gui"])
    args = main_mod.parse_args()
    assert args.command == "gui"
    assert args.compat_warning is None


def test_parse_args_legacy_run_warns(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "-d", "data.csv"])
    args = main_mod.parse_args()
    assert args.command == "run"
    assert args.dataset == "data.csv"
    assert "deprecated" in args.compat_warning.lower()


def test_parse_args_legacy_generate_warns(monkeypatch):
    argv = [
        "prog",
        "--generate-dataset",
        "--mechanism", "linear",
        "--variables", "5",
        "--samples", "20",
        "--generate-output", "generated/toy_dataset",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = main_mod.parse_args()
    assert args.command == "generate"
    assert args.output == "generated/toy_dataset"
    assert "deprecated" in args.compat_warning.lower()


def test_parse_args_legacy_gui_warns(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--gui"])
    args = main_mod.parse_args()
    assert args.command == "gui"
    assert "deprecated" in args.compat_warning.lower()


def test_check_args_requires_dataset_or_model(args_factory):
    args = args_factory()
    with pytest.raises(ValueError):
        main_mod.check_args_validity(args)


def test_check_args_generate_dataset_mode(args_factory):
    args = args_factory(
        command="generate",
        mechanism="linear",
        variables=5,
        samples=20,
        output="outputs/toy_dataset.csv",
        verbose=True,
    )
    run_values = main_mod.check_args_validity(args)
    assert run_values["mode"] == "generate"
    assert run_values["mechanism"] == "linear"
    assert run_values["nodes"] == 5
    assert run_values["samples"] == 20
    assert run_values["output_base"] == "outputs/toy_dataset"
    assert run_values["output_csv_file"].endswith("outputs/toy_dataset.csv")
    assert run_values["output_dot_file"].endswith("outputs/toy_dataset.dot")
    assert run_values["verbose"] is True


def test_check_args_generate_dataset_requires_mandatory_fields(args_factory):
    args = args_factory(command="generate", mechanism="linear")
    with pytest.raises(ValueError, match="`generate` requires"):
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
    assert run_values['device'] == "cpu"
    assert run_values['parallel_jobs'] == 0
    assert run_values['bootstrap_parallel_jobs'] == 0


def test_check_args_normalizes_export_diagnostics(sample_csv, args_factory):
    args = args_factory(dataset=sample_csv, export_diagnostics="reports/diag")
    run_values = main_mod.check_args_validity(args)
    assert run_values["export_diagnostics"] == "reports/diag.xlsx"


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


def test_check_args_cuda_available(sample_csv, args_factory, monkeypatch):
    monkeypatch.setattr(
        main_mod.utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        main_mod.utils.torch.backends.cuda, "is_built", lambda: True)
    args = args_factory(dataset=sample_csv, cuda=True)
    run_values = main_mod.check_args_validity(args)
    assert run_values['device'] == "cuda"


def test_check_args_cuda_unavailable(sample_csv, args_factory, monkeypatch):
    monkeypatch.setattr(
        main_mod.utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        main_mod.utils.torch.backends.cuda, "is_built", lambda: False)
    args = args_factory(dataset=sample_csv, cuda=True)
    with pytest.raises(ValueError, match="CUDA requested"):
        main_mod.check_args_validity(args)


def test_check_args_mps_available(sample_csv, args_factory, monkeypatch):
    monkeypatch.setattr(
        main_mod.utils.torch.backends.mps, "is_available",
        lambda: True, raising=False)
    monkeypatch.setattr(
        main_mod.utils.torch.backends.mps, "is_built",
        lambda: True, raising=False)
    args = args_factory(dataset=sample_csv, mps=True)
    run_values = main_mod.check_args_validity(args)
    assert run_values['device'] == "mps"


def test_check_args_mps_unavailable(sample_csv, args_factory, monkeypatch):
    monkeypatch.setattr(
        main_mod.utils.torch.backends.mps, "is_available",
        lambda: False, raising=False)
    monkeypatch.setattr(
        main_mod.utils.torch.backends.mps, "is_built",
        lambda: False, raising=False)
    args = args_factory(dataset=sample_csv, mps=True)
    with pytest.raises(ValueError, match="MPS requested"):
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
        'adaptive_shap_sampling': False,
        'device': 'cpu',
        'parallel_jobs': 0,
        'bootstrap_parallel_jobs': 0,
    }
    monkeypatch.setattr(main_mod.time, "time", lambda: 100.0)
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
    monkeypatch.setattr(
        main_mod,
        "parse_args",
        lambda: SimpleNamespace(command="run", compat_warning=None),
    )
    monkeypatch.setattr(main_mod, "check_args_validity", lambda _: run_values)
    main_mod.main()
    dummy = instances[0]
    assert dummy.fit_args[1]["adaptive_shap_sampling"] is False
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
        'device': 'cpu',
        'parallel_jobs': 0,
        'bootstrap_parallel_jobs': 0,
    }
    instances = []

    def factory(**kwargs):
        inst = DummyDiscovery(**kwargs)
        instances.append(inst)
        return inst

    monkeypatch.setattr(main_mod, "GraphDiscovery", factory)
    monkeypatch.setattr(
        main_mod,
        "parse_args",
        lambda: SimpleNamespace(command="run", compat_warning=None),
    )
    monkeypatch.setattr(main_mod, "check_args_validity", lambda _: run_values)
    monkeypatch.setattr(main_mod.time, "time", lambda: 0.0)
    monkeypatch.setattr(main_mod.utils, "format_time",
                        lambda delta: (delta, "seconds"))
    main_mod.main()
    assert instances[0].loaded == 'model.pkl'


def test_main_exports_diagnostics_when_requested(monkeypatch, tmp_path):
    class DummyRex:
        def __init__(self):
            self.calls = []

        def export_diagnostics(self, path, include_knowledge=False, ref_graph=None):
            self.calls.append((path, include_knowledge, ref_graph))
            return path

    class DummyDiscovery:
        def __init__(self, **kwargs):
            self.ref_graph = nx.DiGraph([("A", "B")])
            self.trainer = {"sample_nn": SimpleNamespace(rex=DummyRex())}

        def create_experiments(self):
            self.created = True

        def fit_experiments(self, *args, **kwargs):
            self.fit_args = (args, kwargs)

        def combine_and_evaluate_dags(self, prior, combine_op='union'):
            return SimpleNamespace(dag="final_dag", metrics="final_metrics")

        def printout_results(self, dag, metrics, combine_op='union'):
            self.printed = (dag, metrics, combine_op)

    export_path = str(tmp_path / "diag.xlsx")
    run_values = {
        'dataset_name': 'sample',
        'estimator': 'rex',
        'dataset_filepath': 'data.csv',
        'true_dag': 'truth.dot',
        'verbose': False,
        'seed': 7,
        'load_model': None,
        'no_train': False,
        'hpo_iterations': 3,
        'bootstrap_iterations': 4,
        'prior': None,
        'combine_op': 'union',
        'output_path': None,
        'model_filename': None,
        'output_dag_file': None,
        'adaptive_shap_sampling': True,
        'device': 'cpu',
        'parallel_jobs': 0,
        'bootstrap_parallel_jobs': 0,
        'export_diagnostics': export_path,
    }
    instances = []

    def factory(**kwargs):
        inst = DummyDiscovery(**kwargs)
        instances.append(inst)
        return inst

    monkeypatch.setattr(main_mod, "GraphDiscovery", factory)
    monkeypatch.setattr(
        main_mod,
        "parse_args",
        lambda: SimpleNamespace(command="run", compat_warning=None),
    )
    monkeypatch.setattr(main_mod, "check_args_validity", lambda _: run_values)
    monkeypatch.setattr(main_mod.time, "time", lambda: 0.0)
    monkeypatch.setattr(main_mod.utils, "format_time",
                        lambda delta: (delta, "seconds"))

    main_mod.main()

    rex = instances[0].trainer["sample_nn"].rex
    assert rex.calls == [(export_path, True, instances[0].ref_graph)]


def test_main_warns_when_adaptive_disabled_large_dataset(monkeypatch, capsys):
    class DummyDiscovery:
        def __init__(self, **kwargs):
            self.data = pd.DataFrame({"a": range(2001)})
            self.trainer = {'one': SimpleNamespace(
                dag='loaded', metrics=None)}

        def load_model(self, path):
            self.loaded = path

        def printout_results(self, dag, metrics, combine_op='union'):
            self.printed = (dag, metrics, combine_op)

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
        'adaptive_shap_sampling': False,
        'device': 'cpu',
        'parallel_jobs': 0,
        'bootstrap_parallel_jobs': 0,
    }
    monkeypatch.setattr(main_mod, "GraphDiscovery", DummyDiscovery)
    monkeypatch.setattr(
        main_mod,
        "parse_args",
        lambda: SimpleNamespace(command="run", compat_warning=None),
    )
    monkeypatch.setattr(main_mod, "check_args_validity", lambda _: run_values)
    monkeypatch.setattr(main_mod.time, "time", lambda: 0.0)
    monkeypatch.setattr(main_mod.utils, "format_time",
                        lambda delta: (delta, "seconds"))
    main_mod.main()
    captured = capsys.readouterr()
    assert "Adaptive SHAP sampling is disabled" in captured.err


def test_main_generates_dataset_and_writes_outputs(monkeypatch, tmp_path):
    output_base = tmp_path / "generated" / "toy_dataset"
    run_values = {
        "mode": "generate",
        "mechanism": "linear",
        "nodes": 5,
        "samples": 20,
        "max_parents": 3,
        "seed": 7,
        "rescale": True,
        "timeout_s": 30.0,
        "max_retries": 5,
        "min_edges": 0,
        "max_edges": 30,
        "output_base": str(output_base),
        "output_csv_file": str(output_base) + ".csv",
        "output_dot_file": str(output_base) + ".dot",
        "verbose": False,
    }
    monkeypatch.setattr(main_mod.time, "time", lambda: 100.0)
    monkeypatch.setattr(main_mod.utils, "format_time",
                        lambda delta: (delta, "seconds"))
    monkeypatch.setattr(
        main_mod,
        "parse_args",
        lambda: SimpleNamespace(command="generate", compat_warning=None),
    )
    monkeypatch.setattr(main_mod, "check_args_validity", lambda _: run_values)

    main_mod.main()

    assert os.path.isfile(run_values["output_csv_file"])
    assert os.path.isfile(run_values["output_dot_file"])
    generated = pd.read_csv(run_values["output_csv_file"])
    assert generated.shape == (20, 5)


def test_main_shows_help_with_no_subcommand(monkeypatch, capsys):
    parser = main_mod._build_parser()
    monkeypatch.setattr(
        main_mod,
        "parse_args",
        lambda: SimpleNamespace(command=None, compat_warning=None, _parser=parser),
    )
    main_mod.main()
    captured = capsys.readouterr()
    assert "usage: " in captured.out
    assert "{run,generate,gui}" in captured.out


def test_main_emits_legacy_compat_warning(monkeypatch, capsys, tmp_path):
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
        'prior': None,
        'combine_op': 'union',
        'output_path': None,
        'model_filename': None,
        'output_dag_file': None,
        'adaptive_shap_sampling': True,
        'device': 'cpu',
        'parallel_jobs': 0,
        'bootstrap_parallel_jobs': 0,
    }

    class DummyDiscovery:
        def __init__(self, **kwargs):
            self.trainer = {'initial': SimpleNamespace(dag="start", metrics=None)}

        def create_experiments(self):
            pass

        def fit_experiments(self, *args, **kwargs):
            pass

        def combine_and_evaluate_dags(self, prior, combine_op='union'):
            return SimpleNamespace(dag="final_dag", metrics="final_metrics")

        def printout_results(self, dag, metrics, combine_op='union'):
            pass

    monkeypatch.setattr(main_mod, "GraphDiscovery", DummyDiscovery)
    monkeypatch.setattr(
        main_mod,
        "parse_args",
        lambda: SimpleNamespace(
            command="run",
            compat_warning="DEPRECATION: legacy flat run CLI is deprecated; use `causalexplain run ...`.",
        ),
    )
    monkeypatch.setattr(main_mod, "check_args_validity", lambda _: run_values)
    monkeypatch.setattr(main_mod.time, "time", lambda: 0.0)
    monkeypatch.setattr(main_mod.utils, "format_time",
                        lambda delta: (delta, "seconds"))

    main_mod.main()
    captured = capsys.readouterr()
    assert "legacy flat run CLI is deprecated" in captured.err


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
    gd.fit_experiments(
        hpo_iterations=5,
        bootstrap_iterations=6,
        extra="value",
        adaptive_shap_sampling=False)
    trainer.fit_predict.assert_called_once_with(
        estimator='pc', verbose=False, extra="value",
        adaptive_shap_sampling=False)


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
