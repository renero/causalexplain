#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# causalexplain/__main__.py
#
# (C) 2024, 2025 J. Renero
#
# This file is part of causalexplain
#
# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init
# pylint: disable=C0103:invalid-name, C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, W0511:fixme
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches
#
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from causalexplain.common import (DEFAULT_BOOTSTRAP_TOLERANCE,
                                  DEFAULT_BOOTSTRAP_TRIALS, DEFAULT_HPO_TRIALS,
                                  DEFAULT_SEED, DEFAULT_MAX_CSV_LINES,
                                  HEADER_ASCII, SUPPORTED_METHODS, utils)
from causalexplain.gui.graph_utils import dag_is_valid
from causalexplain.gui.settings import default_generate_settings

if TYPE_CHECKING:
    import networkx as nx

    from .causalexplainer import GraphDiscovery as GraphDiscoveryType
    from causalexplain.common.notebook import Experiment


GraphDiscovery = None
CLI_SUBCOMMANDS = ("run", "generate", "gui")
SUPPORTED_GENERATION_MECHANISMS = (
    "linear",
    "polynomial",
    "sigmoid_add",
    "sigmoid_mix",
    "gp_add",
    "gp_mix",
)
LEGACY_GENERATE_FLAG_MAP = {
    "--generate-output": "--output",
    "--generate-timeout": "--timeout",
    "--generate-max-retries": "--max-retries",
    "--generate-min-edges": "--min-edges",
    "--generate-max-edges": "--max-edges",
    "--generate-max-parents": "--max-parents",
    "--generate-rescale": "--rescale",
    "--no-generate-rescale": "--no-rescale",
}


def _resolve_graph_discovery() -> type["GraphDiscoveryType"]:
    """Return GraphDiscovery, honoring monkeypatched module globals."""
    global GraphDiscovery
    if GraphDiscovery is None:
        from .causalexplainer import GraphDiscovery as _GraphDiscovery
        GraphDiscovery = _GraphDiscovery
    return GraphDiscovery


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach training/evaluation arguments to the run subcommand."""
    device_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        '-b', '--bootstrap', type=int, required=False,
        help='Bootstrap iterations. Default is 20.')
    parser.add_argument(
        '-B', '--bootstrap-parallel-jobs', type=int, required=False, default=0,
        help='Number of parallel jobs for bootstrap iterations (0 = sequential).')
    parser.add_argument(
        '--bootstrap-shap-cache',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Cache SHAP values once during bootstrap to speed up iterations. '
             'Use --no-bootstrap-shap-cache to disable.')
    parser.add_argument(
        '--precompute-target-matrices',
        action=argparse.BooleanOptionalAction,
        dest='precompute_target_matrices',
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS)
    parser.add_argument(
        '-gbt-optimization', '--gbt-optimization',
        action=argparse.BooleanOptionalAction,
        dest='precompute_target_matrices',
        default=False,
        help='Enable the GBT optimization that caches per-target feature '
             'matrices. Use --no-gbt-optimization to disable (default).')
    parser.add_argument(
        '-c', '--combine', type=str, required=False,
        choices=['union', 'intersection'],
        help='Combine ReX DAGs using the specified operation: union or intersection.')
    device_group.add_argument(
        '-C', '--cuda', action='store_true', required=False,
        help='Run on CUDA (requires a CUDA-enabled build).')
    parser.add_argument(
        '-d', '--dataset', type=str, required=False,
        help='Dataset name. Must be CSV file with headers and comma separated columns')
    parser.add_argument(
        '--export-diagnostics', type=str, required=False,
        help='Optional .xlsx path for exporting Rex diagnostics after prediction.')
    parser.add_argument(
        '-i', '--iterations', type=int, required=False,
        help='Hyper-parameter tuning max. iterations. Default is 20.')
    parser.add_argument(
        '--hpo-optimization',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable HPO optimization (pruning + downsampled objective). '
             'Use --no-hpo-optimization to disable (default).')
    parser.add_argument(
        '--hpo-optimization-limit', type=int, required=False,
        help='Row cap for the HPO objective when optimization is enabled. '
             'Omit to use an internal default.')
    parser.add_argument(
        '-l', '--load_model', type=str, required=False,
        help='Model name (pickle) to load. If not specified, the model will be trained and avealuated.')
    parser.add_argument(
        '--shap-budget', type=int,
        dest='shap_budget', required=False,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS)
    parser.add_argument(
        '-H', '--shap-optimization-limit', type=int,
        dest='shap_budget', required=False,
        help='SHAP optimization limit (background + explained rows). '
             'Omit to disable (default).')
    parser.add_argument(
        '--max-shap-samples', '--max_shap_samples', type=int, required=False,
        help='Deprecated; use --shap-optimization-limit.')
    parser.add_argument(
        '-m', '--method', type=str, required=False,
        choices=['rex', 'pc', 'fci', 'ges', 'lingam', 'cam', 'notears'],
        help="Method to used. If not specified, the method will be ReX.\n" +
        "Other options are: 'pc', 'fci', 'ges', 'lingam', 'cam', 'notears'. "
        "Note: 'pc' and 'cam' are currently unsupported public interfaces.")
    device_group.add_argument(
        '-M', '--mps', action='store_true', required=False,
        help='Run on Apple Silicon MPS (requires MPS support).')
    parser.add_argument(
        '-n', '--no-train', action='store_true', required=False,
        help='Do not train the model, just evaluate it. If not specified, the model will be trained.')
    parser.add_argument(
        '-o', '--output', type=str, required=False,
        help='Output file where saving the final DAG (dot format). If not specified, ' +
        'the final DAG will be printed to stdout.')
    parser.add_argument(
        '-P', '--parallel-jobs', type=int, required=False, default=0,
        help='Number of parallel jobs for CPU training (0 = sequential).')
    parser.add_argument(
        '-p', '--prior', type=str, required=False,
        help='Prior file (JSON format) to use in the model')
    parser.add_argument(
        '-q', '--quiet', action='store_true', required=False, help='Quiet mode.')
    parser.add_argument(
        '-s', '--save_model', type=str, required=False, nargs='?', const='',
        help='Save model as specified name. If not specified the model will be saved' +
        ' with the same name as the dataset, but with pickle extension.')
    parser.add_argument(
        '-S', '--seed', type=int, required=False, help='Random seed')
    parser.add_argument(
        '-a', '--shap-sampling',
        action=argparse.BooleanOptionalAction,
        dest='adaptive_shap_sampling',
        default=True,
        help='Enable adaptive SHAP background sampling and stability checks. '
             'Use --no-shap-sampling to disable. Default is enabled.')
    parser.add_argument(
        '-T', '--threshold', type=float, required=False,
        help='Threshold (0 .. 1) to apply to the bootstrapped adjacency matrix. Default is 0.3')
    parser.add_argument(
        '-t', '--true_dag', type=str, required=False,
        help='True DAG file name. The file must be in .dot format')
    parser.add_argument(
        '-v', '--verbose', action='store_true', required=False,
        help='Verbose mode, instead of progress bar.')


def _add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach synthetic-data generation arguments to the generate subcommand."""
    generate_defaults = default_generate_settings()
    parser.add_argument(
        '--mechanism', type=str, required=False,
        choices=SUPPORTED_GENERATION_MECHANISMS,
        help='Synthetic data mechanism to use.')
    parser.add_argument(
        '--variables', type=int, required=False,
        help='Number of variables to generate.')
    parser.add_argument(
        '--samples', type=int, required=False,
        help='Number of rows to generate.')
    parser.add_argument(
        '-o', '--output', type=str, required=False,
        help='Output base path. The CLI writes both <path>.csv and <path>.dot.')
    parser.add_argument(
        '--timeout', type=float, required=False,
        default=generate_defaults["timeout_s"],
        help='Maximum time in seconds to search for a valid generated DAG. Default is 30.')
    parser.add_argument(
        '--max-retries', type=int, required=False,
        default=generate_defaults["max_retries"],
        help='Maximum number of retries when searching for a valid generated DAG. Default is 50.')
    parser.add_argument(
        '--min-edges', type=int, required=False,
        default=generate_defaults["min_edges"],
        help='Minimum number of edges allowed in the generated DAG. Default is 0.')
    parser.add_argument(
        '--max-edges', type=int, required=False,
        default=generate_defaults["max_edges"],
        help='Maximum number of edges allowed in the generated DAG. Default is 30.')
    parser.add_argument(
        '--max-parents', type=int, required=False,
        default=generate_defaults["max_parents"],
        help='Upper bound on parents per node in the generated DAG. Default is 3.')
    parser.add_argument(
        '--rescale',
        action=argparse.BooleanOptionalAction,
        default=generate_defaults["rescale"],
        help='Rescale generated columns to zero mean and unit variance. '
             'Use --no-rescale to disable. Default is enabled.')
    parser.add_argument(
        '-S', '--seed', type=int, required=False,
        help='Random seed. Default is 1234.')
    parser.add_argument(
        '-v', '--verbose', action='store_true', required=False,
        help='Verbose mode for dataset generation.')


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Causal Graph Learning with ReX and other compared methods.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{run,generate,gui}")
    run_parser = subparsers.add_parser(
        "run",
        help="Train, evaluate, or load causal-discovery models.",
    )
    _add_run_arguments(run_parser)
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate a synthetic dataset and its ground-truth DAG.",
    )
    _add_generate_arguments(generate_parser)
    subparsers.add_parser(
        "gui",
        help="Launch the local NiceGUI interface.",
    )
    return parser


def _normalize_legacy_argv(argv: list[str]) -> tuple[list[str], Optional[str]]:
    """Rewrite legacy flat CLI forms to subcommand-based invocations."""
    if not argv or argv[0] in CLI_SUBCOMMANDS:
        return argv, None
    if any(arg in {"-h", "--help"} for arg in argv):
        return argv, None
    if "--gui" in argv:
        rewritten = [arg for arg in argv if arg != "--gui"]
        return (
            ["gui", *rewritten],
            "DEPRECATION: top-level `--gui` is deprecated; use `causalexplain gui`.",
        )
    if "--generate-dataset" in argv:
        rewritten = []
        for arg in argv:
            if arg == "--generate-dataset":
                continue
            rewritten.append(LEGACY_GENERATE_FLAG_MAP.get(arg, arg))
        return (
            ["generate", *rewritten],
            "DEPRECATION: legacy flat generation CLI is deprecated; "
            "use `causalexplain generate ...`.",
        )
    return (
        ["run", *argv],
        "DEPRECATION: legacy flat run CLI is deprecated; "
        "use `causalexplain run ...`.",
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments for the causal discovery runner.

    Args:
        argv (Optional[list[str]]): Optional argv override without program name.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    normalized_argv, compat_warning = _normalize_legacy_argv(raw_argv)
    args = parser.parse_args(normalized_argv)
    args._parser = parser
    args.raw_argv = raw_argv
    args.normalized_argv = normalized_argv
    args.compat_warning = compat_warning
    return args


def _normalize_generate_output_path(raw_output: str) -> str:
    """Normalize the base output path for dataset generation."""
    output_path = raw_output.strip()
    if not output_path:
        raise ValueError("--output is required.")
    if output_path.endswith(os.sep) or os.path.isdir(output_path):
        raise ValueError(
            "--output must include a file base name, not only a directory."
        )
    base, ext = os.path.splitext(output_path)
    if ext.lower() in {".csv", ".dot"}:
        output_path = base
    elif ext:
        raise ValueError(
            "--output must omit the extension or end in .csv/.dot."
        )
    if not output_path:
        raise ValueError("Invalid --output value.")
    return output_path


def _check_generate_args_validity(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate synthetic dataset generation arguments."""
    missing = []
    if args.mechanism is None:
        missing.append("--mechanism")
    if args.variables is None:
        missing.append("--variables")
    if args.samples is None:
        missing.append("--samples")
    if args.output is None:
        missing.append("--output")
    if missing:
        raise ValueError(
            "`generate` requires " + ", ".join(missing) + "."
        )

    nodes = int(args.variables)
    samples = int(args.samples)
    if nodes < 2:
        raise ValueError("--variables must be at least 2.")
    if samples <= 0:
        raise ValueError("--samples must be a positive integer.")

    timeout_s = float(args.timeout)
    max_retries = int(args.max_retries)
    min_edges = int(args.min_edges)
    max_edges = int(args.max_edges)
    max_parents = int(args.max_parents)
    if timeout_s <= 0:
        raise ValueError("--timeout must be greater than 0.")
    if max_retries < 0:
        raise ValueError("--max-retries must be 0 or greater.")
    if min_edges < 0 or max_edges < 0:
        raise ValueError("Edge limits for generation must be 0 or greater.")
    if min_edges > max_edges:
        raise ValueError(
            "--min-edges must be less than or equal to --max-edges."
        )
    if max_parents < 0:
        raise ValueError("--max-parents must be 0 or greater.")

    output_base = _normalize_generate_output_path(args.output)
    return {
        "mode": "generate",
        "mechanism": args.mechanism,
        "nodes": nodes,
        "samples": samples,
        "max_parents": max_parents,
        "seed": args.seed if args.seed is not None else DEFAULT_SEED,
        "rescale": bool(args.rescale),
        "timeout_s": timeout_s,
        "max_retries": max_retries,
        "min_edges": min_edges,
        "max_edges": max_edges,
        "output_base": output_base,
        "output_csv_file": f"{output_base}.csv",
        "output_dot_file": f"{output_base}.dot",
        "verbose": True if args.verbose else False,
    }


def check_args_validity(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Validate CLI arguments and derive runtime configuration values.

    This performs file existence checks and computes defaults that drive
    the end-to-end experiment run.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Dict[str, Any]: A dictionary of validated run values.
    """
    run_values = {}
    if getattr(args, "command", None) == "generate":
        return _check_generate_args_validity(args)
    if getattr(args, "command", None) == "gui":
        return {"mode": "gui"}
    run_values['mode'] = 'run'

    # Set model type (estimator)
    if args.method is None:
        args.method = 'rex'
        run_values['estimator'] = 'rex'
    else:
        assert args.method in SUPPORTED_METHODS, \
            "Method must be one of: rex, pc, fci, ges, lingam, cam, notears"
        run_values['estimator'] = str(args.method)

    # Check that either dataset is provided or both load_model and no_train
    # are specified
    if args.dataset is None:
        if not (args.load_model and args.no_train):
            raise ValueError(
                "When dataset is not specified, both load_model (-l) and "
                "no_train (-n) options must be provided")
        else:
            run_values['dataset_name'] = None
            run_values['dataset_filepath'] = None
    else:
        # If dataset is provided, check that it exists
        assert os.path.isfile(args.dataset), \
            f"Dataset file '{args.dataset}' does not exist"
        run_values['dataset_filepath'] = args.dataset

        # Extract the path from where the dataset is, dataset basename
        run_values['dataset_path'] = os.path.dirname(args.dataset)
        dataset_name = os.path.basename(args.dataset)
        run_values['dataset_name'] = dataset_name.replace('.csv', '')

    # Load true DAG, if specified (true_dag)
    run_values['true_dag'] = None
    if args.true_dag is not None:
        assert '.dot' in args.true_dag, "True DAG must be in .dot format"
        assert os.path.isfile(args.true_dag), "True DAG file does not exist"
        run_values['true_dag'] = args.true_dag

    # Check if 'args.load_mode' does not contain path information. In that case
    # assume it is located in the current directory
    if args.load_model is not None and not os.path.isabs(args.load_model):
        args.load_model = os.path.join(os.getcwd(), args.load_model)
    if args.load_model and not os.path.isfile(args.load_model):
        raise FileNotFoundError("Model file does not exist")
    run_values['load_model'] = args.load_model

    if args.no_train:
        run_values['no_train'] = True
    else:
        run_values['no_train'] = False

    # Load prior, if specified
    run_values['prior'] = None
    if args.prior is not None:
        assert '.json' in args.prior, "Prior file must be in JSON format"
        assert os.path.isfile(args.prior), "Prior file does not exist"
        # Load the JSON file int a List of List of str in run_values['prior']
        run_values['prior'] = utils.read_json_file(args.prior)

    # Determine where to save the model pickle.
    if args.save_model == '':
        save_model = f"{args.dataset.replace('.csv', '')}"
        save_model = f"{save_model}_{args.method}.pickle"
        run_values['save_model'] = os.path.basename(save_model)
        # Output_path is the current directory
        run_values['output_path'] = os.getcwd()
        run_values['model_filename'] = utils.valid_output_name(
            filename=save_model, path=run_values['output_path'])
    elif args.save_model is not None:
        run_values['save_model'] = args.save_model
        run_values['output_path'] = os.path.dirname(args.save_model)
        run_values['model_filename'] = args.save_model
    else:
        run_values['save_model'] = None
        run_values['output_path'] = None

    run_values['seed'] = args.seed if args.seed is not None else DEFAULT_SEED
    run_values['quiet'] = True if args.quiet else False
    run_values['hpo_iterations'] = args.iterations \
        if args.iterations is not None else DEFAULT_HPO_TRIALS
    run_values['hpo_optimization'] = getattr(args, "hpo_optimization", False)
    hpo_optimization_limit = getattr(args, "hpo_optimization_limit", None)
    if hpo_optimization_limit is not None and hpo_optimization_limit <= 0:
        hpo_optimization_limit = None
    run_values['hpo_optimization_limit'] = hpo_optimization_limit
    run_values['bootstrap_iterations'] = args.bootstrap \
        if args.bootstrap is not None else DEFAULT_BOOTSTRAP_TRIALS
    run_values['bootstrap_tolerance'] = args.threshold \
        if args.threshold is not None else DEFAULT_BOOTSTRAP_TOLERANCE
    run_values['bootstrap_shap_cache'] = getattr(
        args, "bootstrap_shap_cache", True)
    if any(arg.startswith("--precompute-target-matrices")
           for arg in sys.argv) or \
            any(arg.startswith("--no-precompute-target-matrices")
                for arg in sys.argv):
        print("WARNING: --precompute-target-matrices is deprecated; "
              "use --gbt-optimization instead.")
    run_values['precompute_target_matrices'] = getattr(
        args, "precompute_target_matrices", False)

    run_values['combine_op'] = args.combine if args.combine is not None else 'union'
    run_values['verbose'] = True if args.verbose else False
    run_values['output_dag_file'] = args.output
    export_diagnostics = getattr(args, "export_diagnostics", None)
    if export_diagnostics is not None:
        _, export_ext = os.path.splitext(export_diagnostics)
        if not export_ext:
            export_diagnostics = f"{export_diagnostics}.xlsx"
        elif export_ext.lower() != ".xlsx":
            raise ValueError("--export-diagnostics must omit the extension or end in .xlsx.")
    run_values['export_diagnostics'] = export_diagnostics
    run_values['adaptive_shap_sampling'] = (
        args.adaptive_shap_sampling
        if hasattr(args, 'adaptive_shap_sampling') else True)
    if any(arg.startswith("--shap-budget") for arg in sys.argv):
        print("WARNING: --shap-budget is deprecated; "
              "use --shap-optimization-limit instead.")
    shap_budget = getattr(args, "shap_budget", None)
    legacy_max_shap = getattr(args, "max_shap_samples", None)
    if shap_budget is None or shap_budget <= 0:
        if legacy_max_shap is not None and legacy_max_shap > 0:
            print("WARNING: --max-shap-samples is deprecated; "
                  "use --shap-optimization-limit instead.")
            shap_budget = legacy_max_shap
        else:
            shap_budget = None
    run_values['shap_budget'] = shap_budget
    run_values['max_shap_samples'] = shap_budget
    run_values['parallel_jobs'] = getattr(args, "parallel_jobs", 0)
    run_values['bootstrap_parallel_jobs'] = getattr(
        args, "bootstrap_parallel_jobs", 0)
    if getattr(args, "cuda", False):
        requested_device = "cuda"
    elif getattr(args, "mps", False):
        requested_device = "mps"
    else:
        requested_device = "cpu"
    run_values['device'] = utils.resolve_device(requested_device)

    # return a dictionary with all the new variables created
    return run_values


def header_() -> None:
    """
    Print the ASCII header banner for CLI output.

    The banner was created with the "Ogre" font from
    https://patorjk.com/software/taag/.

    Args:
        None.

    Returns:
        None: This method does not return a value.
    """
    print(HEADER_ASCII)


def show_run_values(run_values: Dict[str, Any]) -> None:
    """
    Print resolved run values for debugging or transparency.

    Args:
        run_values (Dict[str, Any]): A dictionary of run values.

    Returns:
        None: This method does not return a value.
    """
    print("-----")
    print("Run values:")
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a runtime dependency
        pd = None

    for k, v in run_values.items():
        if pd is not None and isinstance(v, pd.DataFrame):
            print(f"- {k}: {v.shape[0]}x{v.shape[1]} DataFrame")
            continue
        print(f"- {k}: {v}")

    print("-----")


def _init_discoverer(run_values: Dict[str, Any]) -> GraphDiscoveryType:
    """
    Initialize the GraphDiscovery object with run-time configuration.
    Check also for CSV size warnings related to SHAP sampling.

    Args:
        run_values (Dict[str, Any]): A dictionary of run values.

    Returns:
        GraphDiscovery: The initialized GraphDiscovery object.
    """
    discoverer_cls = _resolve_graph_discovery()
    discoverer = discoverer_cls(
        experiment_name=run_values['dataset_name'],
        model_type=run_values['estimator'],
        csv_filename=run_values['dataset_filepath'],
        true_dag_filename=run_values['true_dag'],
        verbose=run_values['verbose'],
        seed=run_values['seed'],
        parallel_jobs=run_values['parallel_jobs'],
        bootstrap_parallel_jobs=run_values['bootstrap_parallel_jobs'],
        device=run_values['device'],
        max_shap_samples=run_values.get('shap_budget')
    )
    _check_csv_size_warning(discoverer, run_values)

    return discoverer


def _load_or_prepare(
    discoverer: GraphDiscoveryType,
    run_values: Dict[str, Any]
) -> Optional[Experiment]:
    """
    Load a model if specified, otherwise prepare experiments.

    Args:
        discoverer (GraphDiscovery): The GraphDiscovery object.
        run_values (Dict[str, Any]): A dictionary of run values.

    Returns:
        Optional[Experiment]: The loaded experiment or None if created.
    """
    if run_values['load_model'] is not None:
        discoverer.load_model(run_values['load_model'])
        # REX stores multiple entries; others store a single one
        return next(reversed(discoverer.trainer.values()))

    discoverer.create_experiments()
    return None


def _train_if_needed(
    discoverer: GraphDiscoveryType,
    run_values: Dict[str, Any],
    result: Optional[Experiment]
) -> Optional[Experiment]:
    """
    Train the model if requested, otherwise return the existing result.

    Args:
        discoverer (GraphDiscovery): The GraphDiscovery object.
        run_values (Dict[str, Any]): A dictionary of run values.
        result (Optional[Experiment]): The loaded experiment or None.

    Returns:
        Optional[Experiment]: The trained experiment or the loaded one.
    """
    if run_values['no_train']:
        return result

    discoverer.fit_experiments(
        hpo_iterations=run_values['hpo_iterations'],
        bootstrap_iterations=run_values['bootstrap_iterations'],
        prior=run_values['prior'],
        bootstrap_tolerance=run_values.get('bootstrap_tolerance'),
        quiet=run_values.get('quiet', False),
        adaptive_shap_sampling=run_values.get('adaptive_shap_sampling', True),
        shap_budget=run_values.get('shap_budget'),
        bootstrap_shap_cache=run_values.get('bootstrap_shap_cache', True),
        precompute_target_matrices=run_values.get(
            'precompute_target_matrices', False),
        hpo_optimization=run_values.get('hpo_optimization', False),
        hpo_optimization_limit=run_values.get('hpo_optimization_limit'),
    )
    return discoverer.combine_and_evaluate_dags(
        run_values['prior'], combine_op=run_values['combine_op'])


def _ensure_result(result: Optional[Experiment]) -> Experiment:
    """
    Ensure that the experiment result is available.

    Args:
        result (Optional[Experiment]): The loaded experiment or None.

    Returns:
        Experiment: The loaded or trained experiment.
    """
    if result is None:
        raise RuntimeError(
            "No experiment result available. Provide a model with "
            "--load_model or train one by removing the --no-train flag."
        )
    return result


def _ensure_dag(result: Experiment) -> "nx.DiGraph":
    """
    Ensure that the result contains a DAG.

    Args:
        result (Experiment): The loaded or trained experiment.

    Returns:
        nx.DiGraph: The resulting DAG.
    """
    dag = result.dag
    if dag is None:
        raise RuntimeError(
            "No DAG available from the experiment. Ensure training or model "
            "loading produced a graph."
        )
    return dag


def _resolve_rex_for_diagnostics(
    discoverer: GraphDiscoveryType,
    result: Experiment
):
    """Return the fitted Rex estimator that owns diagnostics artifacts."""
    rex_estimator = getattr(result, "rex", None)
    if rex_estimator is not None:
        return rex_estimator

    rex_candidates = []
    for trainer_name, trainer in discoverer.trainer.items():
        candidate = getattr(trainer, "rex", None)
        if candidate is not None:
            rex_candidates.append((trainer_name, candidate))

    if not rex_candidates:
        raise RuntimeError(
            "No Rex estimator is available for diagnostics export."
        )

    if len(rex_candidates) > 1:
        trainer_name, candidate = rex_candidates[0]
        print(
            "Multiple Rex estimators are available; exporting diagnostics from "
            f"the first fitted trainer '{trainer_name}'."
        )
        return candidate

    return rex_candidates[0][1]


def _export_diagnostics_if_requested(
    discoverer: GraphDiscoveryType,
    result: Experiment,
    run_values: Dict[str, Any],
) -> None:
    """Export Rex diagnostics only when explicitly requested via the CLI."""
    output_file = run_values.get("export_diagnostics")
    if output_file is None:
        return
    if run_values.get("estimator") != "rex":
        raise ValueError("--export-diagnostics is only supported for method 'rex'.")

    rex_estimator = _resolve_rex_for_diagnostics(discoverer, result)
    include_knowledge = run_values.get("true_dag") is not None
    exported_as = rex_estimator.export_diagnostics(
        output_file,
        include_knowledge=include_knowledge,
        ref_graph=getattr(discoverer, "ref_graph", None),
    )
    print(f"Saved diagnostics to {exported_as}")


def _check_csv_size_warning(
    discoverer: GraphDiscoveryType,
    run_values: dict
):
    """
    Check if the CSV size exceeds the maximum allowed lines and warn the user.
    Args:
        discoverer (GraphDiscovery): The GraphDiscovery object
        run_values (dict): A dictionary of run values
    """
    # SAFETY/RUNTIME NOTE:
    # Why we warn when adaptive sampling is disabled:
    # Without adaptive sampling, SHAP can scale poorly and appear to hang on
    # large tables, especially with Kernel-based explainers.
    # What dataset size threshold is used and why:
    # We warn when m > DEFAULT_MAX_CSV_LINES to be conservative about runtime/memory blowups.
    # How users can mitigate (enable adaptive sampling, cap explain set, etc):
    # Enable adaptive_shap_sampling or reduce rows via --shap-optimization-limit,
    # max_explain_samples, or external subsampling.
    if not run_values.get('adaptive_shap_sampling', True):
        data = getattr(discoverer, "data", None)
        if data is not None and len(data) > DEFAULT_MAX_CSV_LINES:
            warning_text = (
                "Adaptive SHAP sampling is disabled and the dataset has "
                f"{len(data)} rows (>2000). SHAP computation may take a very "
                "long time, use excessive memory, or fail to halt. Consider "
                "enabling adaptive_shap_sampling=True or reducing rows via "
                "--shap-optimization-limit, max_explain_samples, or subsampling.")
            print(warning_text, file=sys.stderr)


def _generate_dataset(run_values: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a synthetic dataset using the CLI generation settings."""
    import numpy as np

    from causalexplain.generators.generators import AcyclicGraphGenerator

    np.random.seed(run_values["seed"])
    start_time = time.monotonic()
    max_attempts = run_values["max_retries"] + 1
    attempt = 0
    while attempt < max_attempts:
        if (time.monotonic() - start_time) >= run_values["timeout_s"]:
            break
        attempt += 1
        generator = AcyclicGraphGenerator(
            run_values["mechanism"],
            points=run_values["samples"],
            nodes=run_values["nodes"],
            parents_max=run_values["max_parents"],
            verbose=run_values["verbose"],
        )
        graph, data = generator.generate(rescale=run_values["rescale"])
        if not dag_is_valid(graph, run_values["min_edges"], run_values["max_edges"]):
            continue
        return {"graph": graph, "data": data}

    elapsed = time.monotonic() - start_time
    if elapsed >= run_values["timeout_s"]:
        raise TimeoutError("Timeout reached before a valid DAG was found.")
    raise ValueError("No valid DAG found within the retry limit.")


def _save_generated_dataset(run_values: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """Persist generated data and its ground-truth DAG."""
    graph = payload["graph"]
    data = payload["data"]
    output_dir = os.path.dirname(run_values["output_base"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    data.to_csv(run_values["output_csv_file"], index=False)
    utils.graph_to_dot_file(graph, run_values["output_dot_file"])


def main() -> None:
    """
    Run the CLI entry point for causal discovery experiments.

    This orchestrates argument parsing, model loading or training, evaluation,
    and optional persistence of outputs.

    Args:
        None.

    Returns:
        None: This method does not return a value.
    """
    args = parse_args()
    compat_warning = getattr(args, "compat_warning", None)
    if compat_warning:
        print(compat_warning, file=sys.stderr)
    if getattr(args, "command", None) is None:
        parser = getattr(args, "_parser", None)
        if parser is not None:
            parser.print_help()
        return
    if args.command == "gui":
        from causalexplain.gui import run_gui
        run_gui()
        return
    header_()
    run_values = check_args_validity(args)
    start_time = time.time()
    if run_values.get("mode") == "generate":
        payload = _generate_dataset(run_values)
        _save_generated_dataset(run_values, payload)
        elapsed_time, units = utils.format_time(time.time() - start_time)
        graph = payload["graph"]
        print(f"Elapsed time: {elapsed_time:.1f}{units}")
        print(f"Generated dataset saved to {run_values['output_csv_file']}")
        print(f"Ground-truth DAG saved to {run_values['output_dot_file']}")
        print(
            f"Generated graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        return

    discoverer = _init_discoverer(run_values)
    result = _load_or_prepare(discoverer, run_values)
    result = _train_if_needed(discoverer, run_values, result)
    result = _ensure_result(result)
    dag = _ensure_dag(result)

    elapsed_time, units = utils.format_time(time.time() - start_time)
    print(f"Elapsed time: {elapsed_time:.1f}{units}")
    discoverer.printout_results(dag, result.metrics, run_values['combine_op'])

    if run_values['output_path'] is not None:
        discoverer.save_model(run_values['model_filename'])

    if run_values['output_dag_file'] is not None:
        utils.graph_to_dot_file(dag, run_values['output_dag_file'])
        print(f"Saved DAG to {run_values['output_dag_file']}")
    _export_diagnostics_if_requested(discoverer, result, run_values)


# TODO
# [ ] Ensure that the method works with discrete variables (alternative to RMSE and regressors)
# [ ] Update / Check the Notebook API
# [ ] Update the debug / logging to use a proper logging service.
# [ ] Re-check the saving of models. Experiments duplicate data in all the pickles.
# [X] Add option to save the regressors' errors to a CSV file
# [X] Add option to save the bootstrapped adjacency matrix to a CSV file
# [X] Add option to save the SHAP values to a CSV file
# [X] Add option to save the meta-information stored for each edge in `knowledge`
# [X] Add options to run the 'generators' from the CLI
# [X] Make a single progress bar for the entire training process, instead of one per model and stage
# [?] Get rid of the mlforge pipeline dependency in causalexplain
# [?] Get rid of the ProgBar dependency in causalexplain
# [X] Ensure that the prior is used in all methods that support it and it works correctly
# [X] Fix the length of the messages printed by the tqdm progress bars
# [X] Analyze whether to move to GPU the DNN training for ReX
# [X] Remove the logic for 'correlation' cases all over the codebase (it doesn't work)
# [X] Cast everything to 'float32' where possible to reduce memory consumption
# [X] Study how to use GPU acceleration for SHAP computations


if __name__ == "__main__":
    main()
