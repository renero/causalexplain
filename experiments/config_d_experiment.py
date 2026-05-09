"""
Config D — Sanity check / baseline
N=1000, p=6, sparse DAG (parents_max=2), linear SCM

Compares sklearn vs XGBoost GBT backends across 15 seeds.
No HPO (tune_model=False) to keep runs fast; evaluates default behavior.

Run:
    python experiments/config_d_experiment.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import time
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler

from causalexplain.generators.generators import AcyclicGraphGenerator
from causalexplain.estimators.rex.rex import Rex
from causalexplain.metrics.compare_graphs import evaluate_graph

SEEDS = list(range(15))
N = 1000
P = 6
PARENTS_MAX = 2
MECHANISM = "linear"
BACKENDS = ["sklearn", "xgboost"]


def run_one(seed: int, backend: str) -> dict:
    np.random.seed(seed)

    gen = AcyclicGraphGenerator(MECHANISM, points=N, nodes=P, parents_max=PARENTS_MAX)
    true_graph_int, raw_data = gen.generate()
    # Relabel integer nodes to match DataFrame column names (V0..Vp-1)
    true_graph = nx.relabel_nodes(true_graph_int, {i: f"V{i}" for i in range(P)})

    scaler = StandardScaler()
    data = pd.DataFrame(
        scaler.fit_transform(raw_data), columns=raw_data.columns
    )
    train = data.sample(frac=0.8, random_state=seed)
    test = data.drop(train.index)

    n_true_edges = true_graph.number_of_edges()

    rex = Rex(
        name=f"cfgD_{backend}_s{seed}",
        model_type="gbt",
        explainer="tree",
        tune_model=False,
        prog_bar=False,
        silent=True,
        random_state=seed,
        gbt_backend=backend,
    )

    t0 = time.perf_counter()
    rex.fit_predict(train, test, true_graph)
    elapsed = time.perf_counter() - t0

    m = evaluate_graph(true_graph, rex.dag, list(data.columns))

    r2_scores = None
    if rex.models is not None and hasattr(rex.models, "scoring"):
        # scoring = 1 - R2 per target; convert back to R2
        r2_scores = 1.0 - rex.models.scoring

    return {
        "seed": seed,
        "backend": backend,
        "true_edges": n_true_edges,
        "pred_edges": rex.dag.number_of_edges() if rex.dag else 0,
        "shd": m.shd,
        "f1": m.f1,
        "precision": m.precision,
        "recall": m.recall,
        "Tp": m.Tp,
        "Fp": m.Fp,
        "Fn": m.Fn,
        "mean_r2": float(np.mean(r2_scores)) if r2_scores is not None else float("nan"),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    rows = []
    total = len(SEEDS) * len(BACKENDS)
    done = 0

    print(f"\nConfig D  |  N={N}  p={P}  parents_max={PARENTS_MAX}  mechanism={MECHANISM}")
    print(f"{'Seed':>5}  {'Backend':>10}  {'SHD':>5}  {'F1':>6}  {'Prec':>6}  "
          f"{'Rec':>6}  {'TP':>3}  {'FP':>3}  {'FN':>3}  {'R²':>6}  {'t(s)':>6}")
    print("-" * 80)

    for seed in SEEDS:
        for backend in BACKENDS:
            row = run_one(seed, backend)
            rows.append(row)
            done += 1
            print(
                f"{row['seed']:>5}  {row['backend']:>10}  {row['shd']:>5.1f}  "
                f"{row['f1']:>6.3f}  {row['precision']:>6.3f}  {row['recall']:>6.3f}  "
                f"{row['Tp']:>3}  {row['Fp']:>3}  {row['Fn']:>3}  "
                f"{row['mean_r2']:>6.3f}  {row['elapsed_s']:>6.1f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    out_path = "experiments/config_d_results.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("SUMMARY (mean ± std across 15 seeds)\n")
    summary = (
        df.groupby("backend")[["shd", "f1", "precision", "recall", "mean_r2", "elapsed_s"]]
        .agg(["mean", "std"])
    )
    print(summary.round(3).to_string())

    print("\n--- Decision gate ---")
    sk = df[df.backend == "sklearn"]
    xg = df[df.backend == "xgboost"]
    shd_diff = xg["shd"].mean() - sk["shd"].mean()
    flag = "PASS" if abs(shd_diff) <= 0.5 else "FAIL"
    print(f"  SHD(XGBoost) - SHD(sklearn) = {shd_diff:+.2f}  [{flag}]")
    print(f"  (Gate: |diff| ≤ 0.5 on Config D for XGBoost to remain a candidate)")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
