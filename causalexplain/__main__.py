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

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

import networkx as nx

import pandas as pd
from .causalexplainer import GraphDiscovery
from causalexplain.common.notebook import Experiment

from causalexplain.common import (DEFAULT_BOOTSTRAP_TOLERANCE,
                                  DEFAULT_BOOTSTRAP_TRIALS, DEFAULT_HPO_TRIALS,
                                  DEFAULT_SEED, DEFAULT_MAX_CSV_LINES,
                                  DEFAULT_MAX_SAMPLES,
                                  HEADER_ASCII, SUPPORTED_METHODS, utils)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the causal discovery runner.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Causal Graph Learning with ReX and other compared methods.",
    )
    device_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        '-b', '--bootstrap', type=int, required=False,
        help='Bootstrap iterations. Default is 20.')
    parser.add_argument(
        '-B', '--bootstrap-parallel-jobs', type=int, required=False, default=0,
        help='Number of parallel jobs for bootstrap iterations (0 = sequential).')
    parser.add_argument(
        '-c', '--combine', type=str, required=False, choices=['union', 'intersection'],
        help='Combine ReX DAGs using the specified operation: union or intersection.')
    device_group.add_argument(
        '-C', '--cuda', action='store_true', required=False,
        help='Run on CUDA (requires a CUDA-enabled build).')
    parser.add_argument(
        '-d', '--dataset', type=str, required=False,
        help='Dataset name. Must be CSV file with headers and comma separated columns')
    parser.add_argument(
        '-i', '--iterations', type=int, required=False,
        help='Hyper-parameter tuning max. iterations. Default is 20.')
    parser.add_argument(
        '-l', '--load_model', type=str, required=False,
        help='Model name (pickle) to load. If not specified, the model will be trained and avealuated.')
    parser.add_argument(
        '-H', '--max-shap-samples', '--max_shap_samples', type=int, required=False,
        help='Max background samples for adaptive SHAP. '
             f'Default is {DEFAULT_MAX_SAMPLES}.')
    parser.add_argument(
        '-m', '--method', type=str, required=False,
        choices=['rex', 'pc', 'fci', 'ges', 'lingam', 'cam', 'notears'],
        help="Method to used. If not specified, the method will be ReX.\n" +
        "Other options are: 'pc', 'fci', 'ges', 'lingam', 'cam', 'notears'.")
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
        '-v', '--verbose', action='store_true', required=False, help='Verbose mode, instead of progress bar.')
    parser.add_argument(
        '--gui', action='store_true', required=False,
        help='Launch the local NiceGUI interface.')

    args = parser.parse_args()
    return args


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
    run_values['bootstrap_iterations'] = args.bootstrap \
        if args.bootstrap is not None else DEFAULT_BOOTSTRAP_TRIALS
    run_values['bootstrap_tolerance'] = args.threshold \
        if args.threshold is not None else DEFAULT_BOOTSTRAP_TOLERANCE

    run_values['combine_op'] = args.combine if args.combine is not None else 'union'
    run_values['verbose'] = True if args.verbose else False
    run_values['output_dag_file'] = args.output
    run_values['adaptive_shap_sampling'] = (
        args.adaptive_shap_sampling
        if hasattr(args, 'adaptive_shap_sampling') else True)
    max_shap_samples = getattr(args, "max_shap_samples", None)
    if max_shap_samples is None or max_shap_samples <= 0:
        run_values['max_shap_samples'] = DEFAULT_MAX_SAMPLES
    else:
        run_values['max_shap_samples'] = max_shap_samples
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
    for k, v in run_values.items():
        if isinstance(v, pd.DataFrame):
            print(f"- {k}: {v.shape[0]}x{v.shape[1]} DataFrame")
            continue
        print(f"- {k}: {v}")

    print("-----")


def _init_discoverer(run_values: Dict[str, Any]) -> GraphDiscovery:
    """
    Initialize the GraphDiscovery object with run-time configuration.
    Check also for CSV size warnings related to SHAP sampling.

    Args:
        run_values (Dict[str, Any]): A dictionary of run values.

    Returns:
        GraphDiscovery: The initialized GraphDiscovery object.
    """
    discoverer = GraphDiscovery(
        experiment_name=run_values['dataset_name'],
        model_type=run_values['estimator'],
        csv_filename=run_values['dataset_filepath'],
        true_dag_filename=run_values['true_dag'],
        verbose=run_values['verbose'],
        seed=run_values['seed'],
        parallel_jobs=run_values['parallel_jobs'],
        bootstrap_parallel_jobs=run_values['bootstrap_parallel_jobs'],
        device=run_values['device'],
        max_shap_samples=run_values.get('max_shap_samples')
    )
    _check_csv_size_warning(discoverer, run_values)

    return discoverer


def _load_or_prepare(
    discoverer: GraphDiscovery,
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
    discoverer: GraphDiscovery,
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
        max_shap_samples=run_values.get('max_shap_samples')
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


def _ensure_dag(result: Experiment) -> nx.DiGraph:
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


def _check_csv_size_warning(
    discoverer: GraphDiscovery,
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
    # Enable adaptive_shap_sampling or reduce rows via max_shap_samples,
    # max_explain_samples, or external subsampling.
    if not run_values.get('adaptive_shap_sampling', True):
        data = getattr(discoverer, "data", None)
        if data is not None and len(data) > DEFAULT_MAX_CSV_LINES:
            warning_text = (
                "Adaptive SHAP sampling is disabled and the dataset has "
                f"{len(data)} rows (>2000). SHAP computation may take a very "
                "long time, use excessive memory, or fail to halt. Consider "
                "enabling adaptive_shap_sampling=True or reducing rows via "
                "max_shap_samples, max_explain_samples, or subsampling.")
            print(warning_text, file=sys.stderr)


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
    if getattr(args, "gui", False):
        from causalexplain.gui import run_gui
        run_gui()
        return
    header_()
    run_values = check_args_validity(args)
    start_time = time.time()

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


# TODO
# [ ] Add options to run the 'generators' from the CLI
# [ ] Make a single progress bar for the entire training process, instead of one per model and stage
# [ ] Add option to save the regressors' errors to a CSV file
# [ ] Add option to save the bootstrapped adjacency matrix to a CSV file
# [ ] Add option to save the SHAP values to a CSV file
# [ ] Get rid of the mlforge pipeline dependency in causalexplain
# [ ] Get rid of the ProgBar dependency in causalexplain
# [X] Ensure that the prior is used in all methods that support it and it works correctly
# [X] Fix the length of the messages printed by the tqdm progress bars
# [X] Analyze whether to move to GPU the DNN training for ReX
# [X] Remove the logic for 'correlation' cases all over the codebase (it doesn't work)
# [X] Cast everything to 'float32' where possible to reduce memory consumption
# [X] Study how to use GPU acceleration for SHAP computations


if __name__ == "__main__":
    main()
