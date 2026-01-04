import math

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from typing import Optional

# from causalexplain.common.utils import graph_from_dot_file, load_experiment
from ...common import utils

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid_name, C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, R1702:too-many-branches
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements


class Knowledge:
    """
    This class collects everything we know about each edge in the proposed graph
    in terms of the following properties:

    - origin: the origin node
    - target: the target node
    - is_edge: whether the edge is in the reference graph
    - is_root_cause: whether the origin is a root cause
    - is_leaf_node: whether the origin is a leaf node
    - correlation: the correlation between the individual SHAP values and the origin node
    - KS_pval: the p-value of the Kolmogorov-Smirnov test between the origin and the target
    - shap_edge: whether the edge is in the graph constructed after evaluating mean
        SHAP values.
    - shap_skedastic_pval: the p-value of the skedastic test for the SHAP values
    - parent_skedastic_pval: the p-value of the skedastic test for the parent values
    - mean_shap: the mean of the SHAP values between the origin and the target
    - slope_shap: the slope of the linear regression for target vs. SHAP values
    - slope_target: the slope of the linear regression for the target vs. origin values
    - potential_root: whether the origin is a potential root cause
    - regression_err: the regression error of the origin to the target
    - err_contrib: the error contribution of the origin to the target
    - con_ind_pval: the p-value of the conditional independence test between the origin
        and the target
    """

    def __init__(self, rex: object, ref_graph: nx.DiGraph):
        """
        Arguments:
        ----------
            shaps (ShapEstimator): The shap estimator.
            ref_graph (nx.DiGraph): The reference graph, or ground truth.
        """
        assert rex is not None, "Rex is None"
        assert rex.hierarchies is not None, "Hierarchies is None"
        assert rex.shaps is not None, "ShapEstimator is None"
        assert rex.pi is not None, "PIEstimator is None"

        self.K = 180.0 / math.pi
        self.shaps = rex.shaps
        self.pi = rex.pi
        self.hierarchies = rex.hierarchies
        self.indep = rex.indep
        self.feature_names = rex.feature_names
        self.scoring = rex.models.scoring
        self.ref_graph = ref_graph
        self.G_shap = rex.G_shap
        self.root_causes = rex.root_causes

        self.correlation_th = rex.correlation_th

    def info(self):
        """Returns a dataframe with the knowledge about each edge in the graph"""
        rows = []
        ci = self.indep.compute_cond_indep_pvals()
        for target in self.feature_names:
            for parent in self.feature_names:
                if target == parent:
                    continue

                all_features = [
                    f for f in self.feature_names if f != target]
                feature_pos = all_features.index(parent)

                sd = self.shaps.shap_discrepancies[target][parent]
                pi = self.pi.pi[target]['importances_mean'][feature_pos]

                b0_s, beta1_s = sd.shap_model.params[0], sd.shap_model.params[1]
                b0_y, beta1_y = sd.parent_model.params[0], sd.parent_model.params[1]
                shap_slope = math.atan(beta1_s)*self.K
                parent_slope = math.atan(beta1_y)*self.K
                rows.append({
                    'origin': parent,
                    'target': target,
                    'is_edge': int((parent, target) in self.ref_graph.edges()),
                    'o_is_root': int(nx.ancestors(self.ref_graph, parent) == set()),
                    't_is_leaf': int(nx.descendants(self.ref_graph, target) == set()),
                    'correlation': self.hierarchies.correlations[target][parent],
                    'shap_corr': sd.shap_correlation,
                    'shap_gof': sd.shap_gof,
                    'ks_pval': sd.ks_pvalue,
                    'shap_edge': int(parent in set(self.G_shap.predecessors(target))),
                    'shap_sk_pval': sd.shap_p_value,
                    'parent_sk_pval': sd.parent_p_value,
                    'mean_shap': self.shaps.shap_mean_values[target][feature_pos],
                    'mean_pi': pi,
                    'slope_shap': shap_slope,
                    'slope_target': parent_slope,
                    'pot_root': int(parent in self.root_causes),
                    'regr_err': self.scoring[feature_pos],
                    'err_contrib': self.shaps.error_contribution.loc[target, parent],
                    'cond_ind_pval': ci[(target, parent)]
                })
        self.results = pd.DataFrame.from_dict(rows)
        return self.results

    def retrieve(self, origin: str, target: str, what: Optional[str] = None):
        """Returns the knowledge about a specific edge"""
        if what is None:
            return self.results[(self.results.origin == origin) &
                                (self.results.target == target)]

        return self.results[(self.results.origin == origin) &
                            (self.results.target == target)][what].values[0]


if __name__ == "__main__":

    # Display Options
    np.set_printoptions(precision=4, linewidth=100)
    pd.set_option('display.precision', 4)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # Paths
    path = "/Users/renero/phd/data/RC3/"
    output_path = "/Users/renero/phd/output/RC3/"
    # experiment_name = 'rex_generated_linear_1'
    experiment_name = 'custom_rex'

    # Read the data
    reference_graph = utils.graph_from_dot_file(f"{path}{experiment_name}.dot")
    data = pd.read_csv(f"{path}{experiment_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    custom = utils.load_experiment(f"{experiment_name}", output_path)
    custom.is_fitted_ = True
    print(f"Loaded experiment {experiment_name}")

    custom.feature_names = list(data.columns)
    custom.models.score(data)
    custom.knowledge(reference_graph)
