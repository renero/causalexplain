import math
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causalexplain.estimators.rex.knowledge import Knowledge


class DummyShapDiscrepancy:
    def __init__(self, shap_beta=1.0, parent_beta=2.0, shap_corr=0.4):
        self.shap_model = SimpleNamespace(params=np.array([0.0, shap_beta]))
        self.parent_model = SimpleNamespace(params=np.array([0.0, parent_beta]))
        self.shap_correlation = shap_corr
        self.shap_gof = 0.9
        self.ks_pvalue = 0.05
        self.shap_p_value = 0.1
        self.parent_p_value = 0.2


@pytest.fixture
def base_rex_stub():
    feature_names = ["A", "B"]
    ref_graph = nx.DiGraph()
    ref_graph.add_edge("B", "A")

    shap_discrepancies = {
        "A": {"B": DummyShapDiscrepancy()},
        "B": {"A": DummyShapDiscrepancy(shap_beta=-0.5, parent_beta=-1.0, shap_corr=0.2)},
    }

    class Shaps:
        def __init__(self):
            self.shap_mean_values = {"A": [0.3], "B": [0.7]}
            self.shap_discrepancies = shap_discrepancies
            self.error_contribution = pd.DataFrame(
                [[0.0, 0.2], [0.1, 0.0]], index=feature_names, columns=feature_names
            )

    class PiEstimator:
        pi = {"A": {"importances_mean": [0.8]}, "B": {"importances_mean": [0.4]}}

    class Hierarchies:
        correlated_features = {"A": [], "B": []}
        correlations = {"A": {"B": 0.6}, "B": {"A": -0.1}}

    class Indep:
        def compute_cond_indep_pvals(self):
            return {("A", "B"): 0.33, ("B", "A"): 0.44}

    class Models:
        scoring = [0.05, 0.15]

    rex = SimpleNamespace(
        shaps=Shaps(),
        pi=PiEstimator(),
        hierarchies=Hierarchies(),
        indep=Indep(),
        feature_names=feature_names,
        models=Models(),
        ref_graph=ref_graph,
        G_shap=nx.DiGraph([("B", "A")]),
        root_causes={"B"},
        correlation_th=None,
    )

    return rex, ref_graph


def test_info_builds_dataframe_and_retrieve(base_rex_stub):
    rex, ref_graph = base_rex_stub
    knowledge = Knowledge(rex, ref_graph)

    results = knowledge.info()

    assert set(results[["origin", "target"]].apply(tuple, axis=1)) == {("B", "A"), ("A", "B")}
    row = knowledge.retrieve("B", "A").iloc[0]
    assert row["is_edge"] == 1
    assert pytest.approx(row["slope_shap"]) == math.atan(1.0) * knowledge.K
    assert knowledge.retrieve("A", "B", "pot_root") == 0


def test_info_does_not_skip_correlated_features(monkeypatch, base_rex_stub):
    rex, ref_graph = base_rex_stub
    rex.correlation_th = 0.5
    rex.hierarchies.correlated_features = {"A": ["B"], "B": []}
    knowledge = Knowledge(rex, ref_graph)

    results = knowledge.info()

    assert set(results[["origin", "target"]].apply(tuple, axis=1)) == {("B", "A"), ("A", "B")}
