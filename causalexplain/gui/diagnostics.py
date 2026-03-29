"""Helpers for shaping ReX diagnostics for GUI consumption."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def normalize_rex_diagnostics(
    diagnostics_by_regressor: Dict[str, Dict[str, pd.DataFrame]]
) -> Dict[str, object]:
    """Stack per-regressor diagnostics into GUI-ready dataframes."""
    if not diagnostics_by_regressor:
        raise ValueError("At least one diagnostics bundle is required.")

    errors_frames = []
    shap_frames = []
    bootstrap_edge_frames = []
    bootstrap_matrices: Dict[str, pd.DataFrame] = {}
    feature_names: list[str] = []

    for regressor, bundle in diagnostics_by_regressor.items():
        errors = bundle["regressor_errors"].copy()
        errors["regressor"] = regressor
        errors_frames.append(errors)

        shap_values = bundle["shap_mean_long"].copy()
        shap_values["regressor"] = regressor
        shap_frames.append(shap_values)

        bootstrap_edges = bundle["bootstrap_edges"].copy()
        bootstrap_edges["regressor"] = regressor
        bootstrap_edge_frames.append(bootstrap_edges)

        matrix = bundle["bootstrap_matrix"].copy()
        bootstrap_matrices[regressor] = matrix
        if not feature_names:
            feature_names = [str(name) for name in matrix.columns.tolist()]

    errors_long = pd.concat(errors_frames, ignore_index=True)
    shap_long = pd.concat(shap_frames, ignore_index=True)
    bootstrap_edges_long = pd.concat(bootstrap_edge_frames, ignore_index=True)

    return {
        "feature_names": feature_names,
        "regressors": list(diagnostics_by_regressor.keys()),
        "errors_long": errors_long,
        "shap_long": shap_long,
        "bootstrap_matrices": bootstrap_matrices,
        "bootstrap_edges_long": bootstrap_edges_long,
    }


def regression_errors_for_target(
    errors_long: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Filter regression errors for a selected target variable."""
    filtered = errors_long.loc[errors_long["target"] == target].copy()
    return filtered.loc[:, ["target", "regressor", "error"]].reset_index(drop=True)


def shap_values_for_target(
    shap_long: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Filter and sort SHAP mean values for a selected target variable."""
    filtered = shap_long.loc[shap_long["target"] == target].copy()
    if filtered.empty:
        return filtered.loc[:, ["target", "predictor", "regressor", "mean_shap"]]

    ordering = (
        filtered.groupby("predictor")["mean_shap"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    filtered["predictor"] = pd.Categorical(
        filtered["predictor"], categories=ordering, ordered=True)
    filtered = filtered.sort_values(["predictor", "regressor"])
    return filtered.loc[:, ["target", "predictor", "regressor", "mean_shap"]].reset_index(drop=True)


def bootstrap_details_for_regressor(
    bootstrap_matrices: Dict[str, pd.DataFrame],
    bootstrap_edges_long: pd.DataFrame,
    regressor: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the matrix and edge table for the selected regressor."""
    if regressor not in bootstrap_matrices:
        raise KeyError(f"Unknown regressor '{regressor}'.")

    matrix = bootstrap_matrices[regressor].copy()
    edges = bootstrap_edges_long.loc[
        bootstrap_edges_long["regressor"] == regressor,
        ["source", "target", "regressor", "weight"],
    ].reset_index(drop=True)
    return matrix, edges
