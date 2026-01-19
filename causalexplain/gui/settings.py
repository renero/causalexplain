"""Default settings and persistence helpers for the GUI."""

from __future__ import annotations

from typing import Any, Dict

from causalexplain.common import (
    DEFAULT_BOOTSTRAP_TOLERANCE,
    DEFAULT_BOOTSTRAP_TRIALS,
    DEFAULT_HPO_TRIALS,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_REGRESSORS,
    DEFAULT_SEED,
)


def default_train_settings() -> Dict[str, Any]:
    """Return default training settings for the GUI."""
    return {
        "dataset_path": "",
        "true_dag_path": "",
        "prior_path": "",
        "method": "rex",
        "hpo_iterations": DEFAULT_HPO_TRIALS,
        "bootstrap_iterations": DEFAULT_BOOTSTRAP_TRIALS,
        "bootstrap_tolerance": DEFAULT_BOOTSTRAP_TOLERANCE,
        "combine_op": "union",
        "device": "cpu",
        "parallel_jobs": 0,
        "bootstrap_parallel_jobs": 0,
        "adaptive_shap_sampling": True,
        "max_shap_samples": DEFAULT_MAX_SAMPLES,
        "seed": DEFAULT_SEED,
        "regressors": DEFAULT_REGRESSORS[:],
        "explainer": "gradient",
        "corr_method": "spearman",
        "corr_alpha": 0.6,
        "corr_clusters": 15,
        "condlen": 1,
        "condsize": 0,
        "mean_pi_percentile": 0.8,
        "discrepancy_threshold": 0.99,
        "bootstrap_sampling_split": "auto",
        "save_model_path": "",
        "output_dag_path": "",
    }


def default_load_settings() -> Dict[str, Any]:
    """Return default load/evaluation settings for the GUI."""
    return {
        "model_path": "",
        "true_dag_path": "",
    }


def default_generate_settings() -> Dict[str, Any]:
    """Return default dataset generation settings for the GUI."""
    return {
        "mechanism": "linear",
        "nodes": 10,
        "samples": 500,
        "max_parents": 3,
        "seed": DEFAULT_SEED,
        "rescale": True,
        "timeout_s": 30,
        "max_retries": 50,
        "min_edges": 0,
        "max_edges": 30,
        "output_dir": "",
        "output_name": "generated_dataset",
    }


def merge_settings(stored: Any, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Merge persisted settings into the provided defaults."""
    merged = defaults.copy()
    if isinstance(stored, dict):
        for key, value in stored.items():
            merged[key] = value
    return merged
