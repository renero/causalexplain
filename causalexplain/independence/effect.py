#
# This file contains the methods used to estimate the effect of a treatment in
# an outcome, given a graph.
#
import logging
from typing import Tuple

import networkx as nx
import pandas as pd
from dowhy import CausalModel
from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_refuter import CausalRefuter
from sklearn.preprocessing import StandardScaler

from causalexplain.common import utils

log = logging.getLogger(__name__)


def estimate(
        digraph: nx.DiGraph, 
        data: pd.DataFrame, 
        in_place: bool = True, 
        verbose=False) -> nx.DiGraph:
    """
    Takes the original digraph passed as argument, and computes the effect of
    each treatment in the outcome, given the graph. Treament and outcome are
    the pairs formed by traversing the edges in the oriented DAG.
    The resulting ATE and RefuteResult are computed by calling the method
    `estimate_edge` and incorporated to each edge by adding the ATE and
    RefuteResult as attributes.

    Parameters
    ----------
    digraph : nx.DiGraph
        The causal graph.
    data : pd.DataFrame
        The data.
    in_place : bool, optional
        Whether to modify the graph in place. Defaults to True.
    verbose : bool, optional
        If True, print the results. Defaults to False.

    Returns
    -------
    nx.DiGraph
        The estimated causal graph.
    """
    # Create a copy of the graph
    graph = digraph if in_place else digraph.copy()

    # Compute the effect of each treatment in the outcome
    for t, o in graph.edges():
        ate, refute_pval = estimate_edge(graph, t, o, data, verbose)
        graph.add_edge(t, o, ate=ate, refute_pval=refute_pval)

    log.debug("%s", graph.edges(data=True))

    return graph


def estimate_edge(
        digraph: nx.DiGraph,
        treatment: str,
        outcome: str,
        data: pd.DataFrame,
        verbose=False) -> Tuple[CausalEstimate, CausalRefuter]:
    """
    Estimate the effect of a treatment in an outcome, given a graph.

    Parameters
    ----------
    digraph : nx.DiGraph
        The causal graph.
    treatment : str
        The name of the treatment variable.
    outcome : str
        The name of the outcome variable.
    data : pd.DataFrame
        The data.
    verbose : bool, optional
        If True, print the results. Defaults to False.

    Returns
    -------
    CausalEstimate
        The estimated effect.
    """
    # Definir el modelo causal
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=digraph
    )

    # Identificar el efecto causal
    identified_estimand = model.identify_effect(
        proceed_when_unidentifiable=True)
    log.debug("%s", identified_estimand)

    # Estimar el efecto causal
    estimate = model.estimate_effect(
        identified_estimand,
        target_units="ate",
        method_name="backdoor.linear_regression"
    )
    log.debug("%s", estimate)

    # Pruebas de refutación
    refute = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="random_common_cause"
    )
    log.debug("%s", refute)
    return (estimate.value, refute.refutation_result['p_value'])


def main(exp_name,
         data_path="/Users/renero/phd/data/RC3/",
         output_path="/Users/renero/phd/output/RC4/",
         scale=False):
    """
    Runs a custom main function for the given experiment name.

    Args:
        experiment_name (str): The name of the experiment to run.
        path (str): The path to the data files.
        output_path (str): The path to the output files.

    Returns:
        None
    """

    ref_graph = utils.graph_from_dot_file(f"{data_path}{exp_name}.dot")
    data: pd.DataFrame = pd.read_csv(f"{data_path}{exp_name}.csv")
    if scale:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    # Split the dataframe into train and test

    rex = utils.load_experiment(f"{exp_name}_nn", output_path)
    log.info("Loaded experiment %s", exp_name)

    n_dag = estimate(rex.G_shap, data, verbose=True)


if __name__ == "__main__":
    main("rex_generated_linear_0")
