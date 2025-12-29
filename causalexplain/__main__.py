#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# causalexplain/__main__.py
#
# (C) 2024 J. Renero
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
import time
from typing import Optional

import networkx as nx

import pandas as pd
from .causalexplainer import GraphDiscovery
from causalexplain.common.notebook import Experiment

from causalexplain.common import (DEFAULT_BOOTSTRAP_TOLERANCE,
                                  DEFAULT_BOOTSTRAP_TRIALS, DEFAULT_HPO_TRIALS,
                                  DEFAULT_SEED,
                                  HEADER_ASCII, SUPPORTED_METHODS, utils)


def parse_args():
    class SplitArgs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            """Split a comma-separated list of values."""
            if values is None:
                setattr(namespace, self.dest, [])
            elif isinstance(values, str):
                setattr(namespace, self.dest, values.split(','))
            else:
                setattr(namespace, self.dest, list(values))

    parser = argparse.ArgumentParser(
        description="Causal Graph Learning with ReX and other compared methods.",
    )
    parser.add_argument(
        '-b', '--bootstrap', type=int, required=False,
        help='Bootstrap iterations. Default is 20.')
    parser.add_argument(
        '-c', '--combine', type=str, required=False, choices=['union', 'intersection'],
        help='Combine ReX DAGs using the specified operation: union or intersection.')
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
        '-m', '--method', type=str, required=False,
        choices=['rex', 'pc', 'fci', 'ges', 'lingam', 'cam', 'notears'],
        help="Method to used. If not specified, the method will be ReX.\n" +
        "Other options are: 'pc', 'fci', 'ges', 'lingam', 'cam', 'notears'.")
    parser.add_argument(
        '-n', '--no-train', action='store_true', required=False,
        help='Do not train the model, just evaluate it. If not specified, the model will be trained.')
    parser.add_argument(
        '-o', '--output', type=str, required=False,
        help='Output file where saving the final DAG (dot format). If not specified, ' +
        'the final DAG will be printed to stdout.')
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
        '-T', '--threshold', type=float, required=False,
        help='Threshold (0 .. 1) to apply to the bootstrapped adjacency matrix. Default is 0.3')
    parser.add_argument(
        '-t', '--true_dag', type=str, required=False,
        help='True DAG file name. The file must be in .dot format')
    parser.add_argument(
        '-v', '--verbose', action='store_true', required=False, help='Verbose mode, instead of progress bar.')

    args = parser.parse_args()
    return args


def check_args_validity(args):
    """
    Check the validity of the arguments.

    Returns:
        dict: A dictionary of run values
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

    # return a dictionary with all the new variables created
    return run_values


def header_():
    """
    Done with "Ogre" font from https://patorjk.com/software/taag/
    """
    print(HEADER_ASCII)


def show_run_values(run_values):
    """
    Print the run values.

    Args:
        run_values (dict): A dictionary of run values
    """
    print("-----")
    print("Run values:")
    for k, v in run_values.items():
        if isinstance(v, pd.DataFrame):
            print(f"- {k}: {v.shape[0]}x{v.shape[1]} DataFrame")
            continue
        print(f"- {k}: {v}")

    print("-----")


def _init_discoverer(run_values: dict) -> GraphDiscovery:
    """
    Initialize the GraphDiscovery object.
    Args:
        run_values (dict): A dictionary of run values
    Returns:
        GraphDiscovery: The initialized GraphDiscovery object
    """
    return GraphDiscovery(
        experiment_name=run_values['dataset_name'],
        model_type=run_values['estimator'],
        csv_filename=run_values['dataset_filepath'],
        true_dag_filename=run_values['true_dag'],
        verbose=run_values['verbose'],
        seed=run_values['seed']
    )


def _load_or_prepare(discoverer: GraphDiscovery, run_values: dict) -> Optional[Experiment]:
    """
    Load a model if specified, otherwise prepare experiments.
    Args:
        discoverer (GraphDiscovery): The GraphDiscovery object
        run_values (dict): A dictionary of run values
    Returns:
        Optional[Experiment]: The loaded experiment or None
    """
    if run_values['load_model'] is not None:
        discoverer.load_model(run_values['load_model'])
        # REX stores multiple entries; others store a single one
        return next(reversed(discoverer.trainer.values()))

    discoverer.create_experiments()
    return None


def _train_if_needed(
    discoverer: GraphDiscovery,
    run_values: dict,
    result: Optional[Experiment]
) -> Optional[Experiment]:
    """
    Train the model if needed.
    Args:
        discoverer (GraphDiscovery): The GraphDiscovery object
        run_values (dict): A dictionary of run values
        result (Optional[Experiment]): The loaded experiment or None
    Returns:
        Optional[Experiment]: The trained experiment or the loaded one
    """
    if run_values['no_train']:
        return result

    discoverer.fit_experiments(
        hpo_iterations=run_values['hpo_iterations'],
        bootstrap_iterations=run_values['bootstrap_iterations'],
        prior=run_values['prior'],
        bootstrap_tolerance=run_values.get('bootstrap_tolerance'),
        quiet=run_values.get('quiet', False)
    )
    return discoverer.combine_and_evaluate_dags(
        run_values['prior'], combine_op=run_values['combine_op'])


def _ensure_result(result: Optional[Experiment]) -> Experiment:
    """
    Ensure that the result is not None.
    Args:
        result (Optional[Experiment]): The loaded experiment or None
    Returns:
        Experiment: The loaded or trained experiment
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
        result (Experiment): The loaded or trained experiment
    Returns:
        nx.DiGraph: The resulting DAG
    """
    dag = result.dag
    if dag is None:
        raise RuntimeError(
            "No DAG available from the experiment. Ensure training or model "
            "loading produced a graph."
        )
    return dag


def main():
    header_()
    args = parse_args()
    run_values = check_args_validity(args)
    start_time = time.time()

    discoverer = _init_discoverer(run_values)
    result = _load_or_prepare(discoverer, run_values)
    result = _train_if_needed(discoverer, run_values, result)
    result = _ensure_result(result)
    dag = _ensure_dag(result)

    elapsed_time, units = utils.format_time(time.time() - start_time)
    print(f"(Elapsed time: {elapsed_time:.1f}{units})")
    discoverer.printout_results(dag, result.metrics, run_values['combine_op'])

    if run_values['output_path'] is not None:
        discoverer.save_model(run_values['model_filename'])

    if run_values['output_dag_file'] is not None:
        utils.graph_to_dot_file(dag, run_values['output_dag_file'])
        print(f"Saved DAG to {run_values['output_dag_file']}")


if __name__ == "__main__":
    main()
