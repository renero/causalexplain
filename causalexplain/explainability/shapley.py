"""
This module builds the causal graph based on the informacion that we derived
from the SHAP values. The main idea is to use the SHAP values to compute the
discrepancy between the SHAP values and the target values. This discrepancy
is then used to build the graph.
"""

# pylint: disable=E1101:no-member, W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=C0103:invalid-name, R0902:too-many-instance-attributes
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, W0621:redefined-outer-name
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=W0106:expression-not-assigned, R1702:too-many-branches

import inspect
import math
import multiprocessing
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import colorama
import networkx as nx
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import statsmodels.stats.api as sms
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mlforge.progbar import ProgBar  # type: ignore
from scipy.stats import kstest, spearmanr
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from ..independence.feature_selection import select_features
from ..common import DEFAULT_MAX_SAMPLES, utils

RED = colorama.Fore.RED
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Style.RESET_ALL

AnyGraph = Union[nx.DiGraph, nx.Graph]
K = 180.0 / math.pi


@dataclass
class ShapDiscrepancy:
    """
    A class representing the discrepancy between the SHAP value and the parent
    value for a given feature.

    Attributes:
        - target (str): The name of the target feature.
        - parent (str): The name of the parent feature.
        - shap_heteroskedasticity (bool): Whether the SHAP value exhibits
            heteroskedasticity.
        - parent_heteroskedasticity (bool): Whether the parent value exhibits
            heteroskedasticity.
        - shap_p_value (float): The p-value for the SHAP value.
        - parent_p_value (float): The p-value for the parent value.
        - shap_model (sm.regression.linear_model.RegressionResultsWrapper): The
            regression model for the SHAP value.
        - parent_model (sm.regression.linear_model.RegressionResultsWrapper): The
            regression model for the parent value.
        - shap_discrepancy (float): The discrepancy between the SHAP value and the
            parent value.
        - shap_correlation (float): The correlation between the SHAP value and the
            parent value.
        - shap_gof (float): The goodness of fit for the SHAP value.
        - ks_pvalue (float): The p-value for the Kolmogorov-Smirnov test.
        - ks_result (str): The result of the Kolmogorov-Smirnov test.
    """
    target: str
    parent: str
    shap_heteroskedasticity: bool
    parent_heteroskedasticity: bool
    shap_p_value: float
    parent_p_value: float
    shap_model: sm.regression.linear_model.RegressionResultsWrapper
    parent_model: sm.regression.linear_model.RegressionResultsWrapper
    shap_discrepancy: float
    shap_correlation: float
    shap_gof: float
    ks_pvalue: float
    ks_result: str


ShapResult = Any
SamplingMode = Literal["no_sampling", "single_sample", "multi_sample"]


@dataclass
class ShapRunDiagnostics:
    backend: str
    mode: SamplingMode
    m: int
    n_background: int
    K: int
    seeds: List[int]
    n_explain: int
    stability: Dict[str, Any]
    warnings: List[str]


def _normalize_tabular(X: Any) -> Any:
    """
    Normalize list/array/Series inputs to a tabular 2D shape.

    Args:
        X: Input data in pandas, numpy, or list-like form.

    Returns:
        A 2D tabular representation preserving the input type when possible.
    """
    if isinstance(X, pd.Series):
        return X.to_frame()
    if isinstance(X, np.ndarray):
        return X.reshape(-1, 1) if X.ndim == 1 else X
    if isinstance(X, list):
        arr = np.asarray(X)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr
    return X


def _ensure_2d_array(X: Any) -> np.ndarray:
    """
    Ensure numpy array output is 2D for downstream SHAP code.

    Args:
        X: Input data to normalize.

    Returns:
        A 2D numpy array view of the input.
    """
    arr = np.asarray(X)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _safe_indexing(X: Any, indices: Union[np.ndarray, List[int]]) -> Any:
    """
    Index pandas or numpy inputs with a shared code path.

    Args:
        X: Input data to index.
        indices: Row indices to select.

    Returns:
        The indexed subset in the same container type as the input.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[indices]
    return np.asarray(X)[indices]


def sample_rows(
        X: Any,
        n: Optional[int],
        stratify: Optional[Any] = None,
        seed: Optional[int] = None,
        return_indices: bool = False
        ) -> Union[Any, Tuple[Any, np.ndarray]]:
    """
    Sample rows from X with optional stratification.

    Args:
        X: Input data to sample from.
        n: Number of rows to sample (None uses all rows).
        stratify: Optional labels for stratified sampling.
        seed: Random seed for deterministic sampling.
        return_indices: Whether to return sampled row indices.

    Returns:
        Sampled data, and optionally the sampled indices.
    """
    if n is None:
        indices = np.arange(len(X))
    else:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        m = len(X)
        if n >= m:
            indices = np.arange(m)
        else:
            if stratify is not None:
                stratify_arr = np.asarray(stratify)
                if stratify_arr.ndim > 1:
                    if stratify_arr.shape[1] == 1:
                        stratify_arr = stratify_arr.reshape(-1)
                    else:
                        raise ValueError("stratify must be 1D array-like.")
                if stratify_arr.shape[0] != m:
                    raise ValueError(
                        "stratify must have the same length as X.")
                splitter = StratifiedShuffleSplit(
                    n_splits=1, train_size=n, random_state=seed)
                indices, _ = next(
                    splitter.split(np.zeros(m), stratify_arr))
            else:
                rng = np.random.default_rng(seed)
                indices = rng.choice(m, size=n, replace=False)
    sample = _safe_indexing(X, indices)
    if return_indices:
        return sample, np.asarray(indices)
    return sample


def _make_seeds(random_state: Optional[int], K: int) -> List[int]:
    """
    Generate K deterministic seeds from a base random state.

    Args:
        random_state: Base random seed.
        K: Number of seeds to generate.

    Returns:
        A list of integer seeds.
    """
    if K <= 0:
        return []
    seed_seq = np.random.SeedSequence(random_state)
    return [int(seq.generate_state(1)[0]) for seq in seed_seq.spawn(K)]


def _default_kernel_explain_cap(n_features: int) -> int:
    """
    Heuristic cap for Kernel SHAP explain rows based on feature count.

    Args:
        n_features: Number of features in the dataset.

    Returns:
        Suggested maximum number of explained rows.
    """
    return min(200, max(50, 2 * n_features))


def _default_kernel_nsamples(n_features: int) -> int:
    """
    Heuristic for Kernel SHAP nsamples based on feature count.

    Args:
        n_features: Number of features in the dataset.

    Returns:
        Suggested nsamples value for KernelExplainer.
    """
    return int(min(2 * n_features + 2048, 5000))


def _unwrap_model(model: Any) -> Any:
    """
    Return the underlying model if wrapped (e.g., .model attribute).

    Args:
        model: Input model or wrapper object.

    Returns:
        The unwrapped model instance when available.
    """
    return model.model if hasattr(model, "model") else model


def _is_torch_model(model: Any) -> bool:
    """
    Check if the model is a torch.nn.Module.

    Args:
        model: Input model instance.

    Returns:
        True if the model is a torch.nn.Module.
    """
    return isinstance(model, torch.nn.Module)


def _is_tensorflow_like(model: Any) -> bool:
    """
    Detect TensorFlow/Keras models using module path heuristics.

    Args:
        model: Input model instance.

    Returns:
        True if the model appears to be TensorFlow/Keras based.
    """
    module = getattr(model, "__class__", type(model)).__module__.lower()
    return "tensorflow" in module or "keras" in module


def _is_xgboost_booster(model: Any) -> bool:
    """
    Detect XGBoost Booster objects via module/class name.

    Args:
        model: Input model instance.

    Returns:
        True if the model looks like an XGBoost Booster.
    """
    module = model.__class__.__module__.lower()
    name = model.__class__.__name__.lower()
    return "xgboost" in module and name == "booster"


def _get_torch_device(model: Any) -> torch.device:
    """
    Get the device for torch models, falling back to CPU.

    Args:
        model: Torch model instance.

    Returns:
        The torch device associated with the model parameters.
    """
    if hasattr(model, "parameters"):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    return torch.device("cpu")


def _torch_predict_fn(model: Any) -> Callable[[Any], np.ndarray]:
    """
    Build a numpy-in, numpy-out prediction callable for torch models.

    Args:
        model: Torch model instance.

    Returns:
        A callable that maps numpy inputs to numpy outputs.
    """
    torch_model = _unwrap_model(model)
    torch_model.eval()
    device = _get_torch_device(torch_model)

    def _predict(X):
        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X_arr).to(device)
        with torch.no_grad():
            output = torch_model(X_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.detach().cpu().numpy()

    return _predict


def _xgboost_predict_fn(model: Any) -> Callable[[Any], Any]:
    """
    Build a callable that adapts XGBoost Booster predict API.

    Args:
        model: XGBoost Booster instance.

    Returns:
        A callable that accepts tabular data and returns predictions.
    """
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "xgboost is required to adapt Booster models for SHAP.") from exc

    def _predict(X):
        dmatrix = xgb.DMatrix(np.asarray(X))
        return model.predict(dmatrix)

    return _predict


def _resolve_model_callable(
        model: Any,
        prefer_proba: bool = True) -> Tuple[Callable[[Any], Any], str]:
    """
    Resolve a model into a callable suitable for SHAP explainers.

    Args:
        model: Input model instance or wrapper.
        prefer_proba: Whether to prefer predict_proba when available.

    Returns:
        A tuple of (callable, description) for SHAP usage.
    """
    unwrapped = _unwrap_model(model)
    if _is_xgboost_booster(unwrapped):
        return _xgboost_predict_fn(unwrapped), "xgboost.predict"
    if _is_torch_model(unwrapped):
        return _torch_predict_fn(unwrapped), "torch.predict"
    if prefer_proba and hasattr(unwrapped, "predict_proba"):
        return unwrapped.predict_proba, "predict_proba"
    if hasattr(unwrapped, "predict"):
        return unwrapped.predict, "predict"
    if callable(unwrapped):
        return unwrapped, "callable"
    raise TypeError("Model is not callable and has no predict methods.")


def _call_explainer_silent(explainer: Any, X: Any) -> Any:
    """
    Call a SHAP explainer with silent output when supported.

    Args:
        explainer: SHAP explainer instance.
        X: Input data to explain.

    Returns:
        SHAP output object from the explainer.
    """
    try:
        signature = inspect.signature(explainer.__call__)
    except (TypeError, ValueError):
        return explainer(X)
    if "silent" in signature.parameters:
        return explainer(X, silent=True)
    return explainer(X)


def _is_kernel_like_explainer(explainer: Any) -> bool:
    """
    Detect kernel-like explainers by class/module naming.

    Args:
        explainer: SHAP explainer instance.

    Returns:
        True if the explainer is kernel/permutation-like.
    """
    name = explainer.__class__.__name__.lower()
    module = explainer.__class__.__module__.lower()
    return ("kernel" in name or "kernel" in module or
            "permutation" in name or "permutation" in module)


def _coerce_shap_arrays(shap_result: Any, n_features: int) -> List[np.ndarray]:
    """
    Normalize SHAP outputs into a list of 2D arrays.

    Args:
        shap_result: SHAP output object (arrays or Explanation).
        n_features: Expected number of features.

    Returns:
        A list of 2D numpy arrays with shape (n_samples, n_features).
    """
    values = shap_result.values if isinstance(
        shap_result, shap.Explanation) else shap_result
    if isinstance(values, list):
        arrays = [np.asarray(v) for v in values]
    else:
        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr.reshape(-1, n_features)
        if arr.ndim == 2:
            arrays = [arr]
        elif arr.ndim == 3:
            if arr.shape[1] == n_features:
                arrays = [arr[:, :, i] for i in range(arr.shape[2])]
            elif arr.shape[2] == n_features:
                arrays = [arr[:, i, :] for i in range(arr.shape[1])]
            else:
                arrays = [arr.reshape(arr.shape[0], -1)]
        else:
            arrays = [arr.reshape(arr.shape[0], -1)]

    normalized = []
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] != n_features:
            raise ValueError(
                "SHAP values feature dimension does not match X.")
        normalized.append(arr)
    return normalized


def _global_importance_from_shap(
        shap_result: Any,
        n_features: int,
        class_index: Optional[int] = None) -> np.ndarray:
    """
    Compute global mean(|SHAP|) importance from SHAP outputs.

    Args:
        shap_result: SHAP output object.
        n_features: Expected number of features.
        class_index: Optional class index for multi-output models.

    Returns:
        A 1D numpy array of global importances.
    """
    arrays = _coerce_shap_arrays(shap_result, n_features)
    if class_index is not None:
        if class_index >= len(arrays):
            raise ValueError("class_index is out of range for SHAP outputs.")
        arrays = [arrays[class_index]]
    per_output = [np.mean(np.abs(arr), axis=0) for arr in arrays]
    if len(per_output) == 1:
        return per_output[0]
    return np.mean(np.vstack(per_output), axis=0)


def _mean_pairwise_spearman(imp_vectors: List[np.ndarray]) -> float:
    """
    Compute mean pairwise Spearman correlation for importance vectors.

    Args:
        imp_vectors: List of importance vectors.

    Returns:
        Mean Spearman rank correlation across all pairs.
    """
    if len(imp_vectors) < 2:
        return 1.0
    corrs = []
    for i in range(len(imp_vectors)):
        for j in range(i + 1, len(imp_vectors)):
            corr = spearmanr(imp_vectors[i], imp_vectors[j]).correlation
            if corr is None or np.isnan(corr):
                continue
            corrs.append(corr)
    return float(np.mean(corrs)) if corrs else float("nan")


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine distance in [0, 2].
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1 - np.dot(a, b) / denom)


def _representative_run_index(
        imp_vectors: List[np.ndarray],
        imp_mean: np.ndarray) -> int:
    """
    Pick the run closest to the mean importance vector.

    Args:
        imp_vectors: List of importance vectors.
        imp_mean: Mean importance vector.

    Returns:
        Index of the representative run.
    """
    distances = [_cosine_distance(v, imp_mean) for v in imp_vectors]
    return int(np.argmin(distances))


def _build_stability(
        imp_vectors: List[np.ndarray],
        feature_names: List[Union[str, int]],
        topN_important: int,
        warn_threshold_cv: float,
        warn_threshold_rankcorr: float,
        early_stopped: bool = False) -> Dict[str, Any]:
    """
    Aggregate stability metrics across SHAP runs.

    Args:
        imp_vectors: List of importance vectors per run.
        feature_names: Feature names for reporting top features.
        topN_important: Number of top features to track.
        warn_threshold_cv: Coefficient of variation threshold.
        warn_threshold_rankcorr: Rank correlation threshold.
        early_stopped: Whether the run stopped early.

    Returns:
        Dictionary of stability statistics and warning flags.
    """
    imp_stack = np.vstack(imp_vectors)
    imp_mean = imp_stack.mean(axis=0)
    imp_std = imp_stack.std(axis=0, ddof=0)
    eps = 1e-12
    imp_cv = imp_std / (imp_mean + eps)
    topN = min(int(topN_important), imp_mean.shape[0])
    top_idx = np.argsort(imp_mean)[::-1][:topN]
    mean_rankcorr = _mean_pairwise_spearman(imp_vectors)
    max_cv = float(np.max(imp_cv[top_idx])) if top_idx.size else 0.0
    warned = bool(
        max_cv > warn_threshold_cv or mean_rankcorr < warn_threshold_rankcorr)
    top_features = [feature_names[i] for i in top_idx] if feature_names else \
        top_idx.tolist()
    return {
        "imp_mean": imp_mean,
        "imp_std": imp_std,
        "imp_cv": imp_cv,
        "mean_rankcorr": mean_rankcorr,
        "warned": warned,
        "thresholds": {
            "cv": warn_threshold_cv,
            "rankcorr": warn_threshold_rankcorr,
        },
        "top_features": top_features,
        "top_indices": top_idx.tolist(),
        "max_cv": max_cv,
        "early_stopped": early_stopped,
    }


def _concat_shap_batches(batch_values: List[Any]) -> Any:
    """
    Concatenate SHAP outputs produced in batches.

    Args:
        batch_values: List of SHAP outputs per batch.

    Returns:
        Concatenated SHAP output in the original structure.
    """
    if not batch_values:
        return np.array([])
    first = batch_values[0]
    if isinstance(first, list):
        combined = []
        for idx in range(len(first)):
            combined.append(np.concatenate(
                [np.asarray(b[idx]) for b in batch_values], axis=0))
        return combined
    arrays = [np.asarray(
        b.values if isinstance(b, shap.Explanation) else b)
        for b in batch_values]
    return np.concatenate(arrays, axis=0)


def build_kernel_explainer(model: Any, X_bg: Any) -> Any:
    """
    Create a KernelExplainer with a resolved model callable.

    Args:
        model: Model to explain.
        X_bg: Background data for KernelExplainer.

    Returns:
        An initialized KernelExplainer instance.
    """
    # BACKEND TAILORING NOTE (kernel):
    # Why this exists:
    # Kernel SHAP depends on the model callable output; for classifiers we
    # prefer probabilities to align feature attributions with class likelihoods.
    # What runtime/accuracy tradeoff it introduces:
    # Using predict_proba can increase output dimensionality (multi-class),
    # which increases compute and memory versus raw predictions.
    # When a user might want to override it:
    # Provide a custom wrapper model callable or set class_index to focus a class.
    model_fn, _ = _resolve_model_callable(model, prefer_proba=True)
    return shap.KernelExplainer(model_fn, X_bg)


def compute_kernel_shap(
        explainer: Any,
        X_explain: Any,
        n_features: int,
        nsamples: Optional[int] = None,
        class_index: Optional[int] = None) -> Any:
    """
    Compute Kernel SHAP values with optional class selection.

    Args:
        explainer: KernelExplainer instance.
        X_explain: Data to explain.
        n_features: Number of features in X_explain.
        nsamples: KernelExplainer nsamples parameter.
        class_index: Optional class index for multi-class outputs.

    Returns:
        SHAP values in backend-specific format.
    """
    shap_values = explainer.shap_values(X_explain, nsamples=nsamples)
    # BACKEND TAILORING NOTE (kernel):
    # Why this exists:
    # Kernel SHAP returns per-class arrays for multi-class outputs; selecting a
    # single class reduces memory when only one class matters.
    # What runtime/accuracy tradeoff it introduces:
    # Focusing on one class drops information about other classes but speeds up
    # downstream aggregation and storage.
    # When a user might want to override it:
    # Leave class_index=None to retain per-class outputs for analysis.
    if class_index is not None:
        arrays = _coerce_shap_arrays(shap_values, n_features)
        if class_index >= len(arrays):
            raise ValueError("class_index is out of range for SHAP outputs.")
        return arrays[class_index]
    return shap_values


def build_gradient_explainer(model: Any, X_bg: Any) -> Any:
    """
    Create a GradientExplainer with backend validation.

    Args:
        model: Differentiable model to explain.
        X_bg: Background data for GradientExplainer.

    Returns:
        An initialized GradientExplainer instance.
    """
    # BACKEND TAILORING NOTE (gradient):
    # Why this exists:
    # GradientExplainer needs a differentiable model; we fail fast so users do
    # not spend time on unsupported models with misleading results.
    # What runtime/accuracy tradeoff it introduces:
    # It avoids wasted runtime rather than changing accuracy.
    # When a user might want to override it:
    # Use backend="explainer" or backend="kernel" for non-differentiable models.
    unwrapped = _unwrap_model(model)
    is_torch = _is_torch_model(unwrapped)
    is_tf = _is_tensorflow_like(unwrapped)
    if not (is_torch or is_tf):
        raise ValueError(
            "Gradient backend requires a differentiable model. "
            "Use backend='explainer' or backend='kernel'.")
    if is_torch:
        torch_model = unwrapped
        torch_model.eval()
        device = _get_torch_device(torch_model)
        background = _ensure_2d_array(np.asarray(X_bg, dtype=np.float32))
        background_tensor = torch.from_numpy(background).to(device)
        explainer = shap.GradientExplainer(torch_model, [background_tensor])
        explainer._cex_is_torch = True
        explainer._cex_device = device
        return explainer
    background = _ensure_2d_array(np.asarray(X_bg))
    explainer = shap.GradientExplainer(unwrapped, background)
    explainer._cex_is_torch = False
    explainer._cex_device = None
    return explainer


def _call_gradient_explainer(explainer: Any, inputs: Any) -> Any:
    """
    Call GradientExplainer consistently across SHAP versions.

    Args:
        explainer: GradientExplainer instance.
        inputs: Input data (tensor or array) to explain.

    Returns:
        SHAP values for the input batch.
    """
    try:
        return explainer.shap_values(inputs)
    except Exception:
        result = explainer(inputs)
        return result.values if isinstance(result, shap.Explanation) else result


def compute_gradient_shap(
        explainer: Any,
        X_explain: Any,
        batch_size: int = 128) -> Any:
    """
    Compute Gradient SHAP values in batches.

    Args:
        explainer: GradientExplainer instance.
        X_explain: Data to explain.
        batch_size: Batch size for memory-safe execution.

    Returns:
        SHAP values in backend-specific format.
    """
    # BACKEND TAILORING NOTE (gradient):
    # Why this exists:
    # GradientExplainer can exhaust GPU/CPU memory when explaining too many
    # rows at once; batching limits memory spikes.
    # What runtime/accuracy tradeoff it introduces:
    # Smaller batches add overhead but preserve accuracy while avoiding OOMs.
    # When a user might want to override it:
    # Increase batch_size on larger memory hardware for faster throughput.
    X_array = _ensure_2d_array(np.asarray(X_explain, dtype=np.float32))
    is_torch = bool(getattr(explainer, "_cex_is_torch", False))
    device = getattr(explainer, "_cex_device", None)
    batch_values = []
    for start in range(0, X_array.shape[0], batch_size):
        end = min(start + batch_size, X_array.shape[0])
        batch = X_array[start:end]
        if is_torch:
            batch_tensor = torch.from_numpy(batch).to(device)
            batch_input = [batch_tensor]
        else:
            batch_input = batch
        batch_values.append(_call_gradient_explainer(explainer, batch_input))
    return _concat_shap_batches(batch_values)


def build_generic_explainer(model: Any, X_bg: Any) -> Any:
    """
    Create a generic shap.Explainer with a tabular masker.

    Args:
        model: Model or callable to explain.
        X_bg: Background data for the masker.

    Returns:
        A shap.Explainer instance.
    """
    # BACKEND TAILORING NOTE (explainer):
    # Why this exists:
    # The generic Explainer needs a masker for tabular data; Independent keeps
    # feature perturbations consistent with background distributions.
    # What runtime/accuracy tradeoff it introduces:
    # Independent masking is fast but can ignore feature dependencies.
    # When a user might want to override it:
    # Supply a custom masker if feature dependence is important.
    masker = shap.maskers.Independent(X_bg)
    model_for_explainer = model
    if _is_torch_model(model):
        model_for_explainer = _torch_predict_fn(model)
    return shap.Explainer(model_for_explainer, masker)


def compute_generic_shap(explainer: Any, X_explain: Any) -> Any:
    """
    Compute SHAP values for generic explainers.

    Args:
        explainer: shap.Explainer instance.
        X_explain: Data to explain.

    Returns:
        SHAP values in backend-specific format.
    """
    return _call_explainer_silent(explainer, X_explain)


def compute_shap_adaptive(
        X: Any,
        model: Any,
        backend: Literal["kernel", "gradient", "explainer"],
        y: Optional[Any] = None,
        max_shap_samples: int = DEFAULT_MAX_SAMPLES,
        K_max: int = 5,
        max_explain_samples: Optional[int] = None,
        random_state: Optional[int] = None,
        stratify: Optional[Any] = None,
        warn_threshold_cv: float = 0.10,
        warn_threshold_rankcorr: float = 0.90,
        topN_important: int = 20,
        verbose: bool = False,
        kernel_nsamples: Optional[int] = None,
        batch_size: int = 128,
        class_index: Optional[int] = None,
        adaptive_shap_sampling: bool = True
        ) -> Tuple[ShapResult, ShapRunDiagnostics]:
    """
    Compute SHAP values with adaptive background sampling controls.

    Args:
        X: Input data to explain.
        model: Trained model to explain.
        backend: SHAP backend name ("kernel", "gradient", "explainer").
        y: Optional target values for stratification or diagnostics.
        max_shap_samples: Background cap for adaptive sampling.
        K_max: Maximum number of repeated sampling runs.
        max_explain_samples: Optional cap for explained rows.
        random_state: Random seed for deterministic sampling.
        stratify: Optional stratification labels for sampling.
        warn_threshold_cv: CV threshold for stability warnings.
        warn_threshold_rankcorr: Rank correlation threshold for warnings.
        topN_important: Number of top features to track.
        verbose: Whether to print diagnostic warnings.
        kernel_nsamples: Optional KernelExplainer nsamples override.
        batch_size: Batch size for gradient explainer runs.
        class_index: Optional class index for multi-class outputs.
        adaptive_shap_sampling: Enable adaptive sampling and stability checks.

    Returns:
        A tuple of (shap_result, diagnostics) with backend-specific outputs.
    """
    X = _normalize_tabular(X)
    if backend not in {"kernel", "gradient", "explainer"}:
        raise ValueError("backend must be one of: kernel, gradient, explainer.")
    if max_shap_samples <= 0:
        raise ValueError("max_shap_samples must be > 0.")
    if K_max <= 0:
        raise ValueError("K_max must be > 0.")
    if topN_important <= 0:
        raise ValueError("topN_important must be > 0.")

    X_array = _ensure_2d_array(np.asarray(X))
    m, n_features = X_array.shape
    feature_names = list(
        X.columns) if isinstance(X, pd.DataFrame) else list(range(n_features))
    effective_stratify = stratify if stratify is not None else y
    warn_messages: List[str] = []

    # SAFETY/RUNTIME NOTE:
    # Why we warn when adaptive sampling is disabled:
    # Without adaptive sampling, SHAP can scale poorly and appear to hang on
    # large tables, especially with Kernel-based explainers.
    # What dataset size threshold is used and why:
    # We warn when m > 2000 to be conservative about runtime/memory blowups.
    # How users can mitigate (enable adaptive sampling, cap explain set, etc):
    # Turn on adaptive_shap_sampling or reduce rows via max_shap_samples,
    # max_explain_samples, or external subsampling.
    if (not adaptive_shap_sampling) and m > 2000:
        warning_text = (
            "Adaptive SHAP sampling is disabled and the dataset has "
            f"{m} rows (>2000). SHAP computation may take a very long time, "
            "use excessive memory, or fail to halt. Consider enabling "
            "adaptive_shap_sampling=True or reducing rows via "
            "max_shap_samples, max_explain_samples, or subsampling.")
        warn_messages.append(warning_text)
        warnings.warn(warning_text, UserWarning)
        if verbose:
            print(warning_text)

    if not adaptive_shap_sampling:
        mode: SamplingMode = "no_sampling"
        n_background = m
        K_target = 1
    elif m <= max_shap_samples:
        mode = "no_sampling"
        n_background = m
        K_target = 1
    elif m <= 2 * max_shap_samples:
        mode = "single_sample"
        n_background = max_shap_samples
        K_target = 1
    else:
        mode = "multi_sample"
        n_background = min(max_shap_samples, m)
        K_target = min(int(K_max), 5)

    seeds = _make_seeds(random_state, K_target) if mode != "no_sampling" else []

    n_explain = m
    if backend == "kernel" and adaptive_shap_sampling:
        # BACKEND TAILORING NOTE (kernel):
        # Why this exists:
        # Kernel SHAP runtime scales roughly with n_explain * n_background *
        # p * nsamples; capping n_explain prevents lockups on large datasets.
        # What runtime/accuracy tradeoff it introduces:
        # Fewer explained rows reduce global attribution stability.
        # When a user might want to override it:
        # Set max_explain_samples explicitly for more accuracy.
        if max_explain_samples is None:
            n_explain = min(m, _default_kernel_explain_cap(n_features))
        else:
            n_explain = min(m, max_explain_samples)
    elif max_explain_samples is not None:
        n_explain = min(m, max_explain_samples)

    try:
        if n_explain < m:
            X_explain, _ = sample_rows(
                X, n_explain, stratify=effective_stratify,
                seed=random_state, return_indices=True)
        else:
            X_explain = X
    except ValueError as exc:
        warn_messages.append(
            f"Stratified explain sampling failed; using uniform sampling. {exc}")
        X_explain, _ = sample_rows(
            X, n_explain, stratify=None, seed=random_state, return_indices=True)

    prebuilt_explainer = None
    if backend == "explainer":
        if mode == "no_sampling":
            first_bg = X
        else:
            try:
                first_bg, _ = sample_rows(
                    X, n_background, stratify=effective_stratify,
                    seed=seeds[0], return_indices=True)
            except ValueError as exc:
                warn_messages.append(
                    f"Stratified background sampling failed; using uniform sampling. {exc}")
                first_bg, _ = sample_rows(
                    X, n_background, stratify=None,
                    seed=seeds[0], return_indices=True)
        prebuilt_explainer = build_generic_explainer(model, first_bg)
        if adaptive_shap_sampling and \
                _is_kernel_like_explainer(prebuilt_explainer) and \
                max_explain_samples is None:
            # BACKEND TAILORING NOTE (explainer):
            # Why this exists:
            # shap.Explainer can select Kernel/Permutation backends that scale
            # poorly with explained rows; capping avoids runaway runtime.
            # What runtime/accuracy tradeoff it introduces:
            # Fewer explained rows may reduce global importance stability.
            # When a user might want to override it:
            # Set max_explain_samples or choose a faster backend explicitly.
            kernel_like_cap = min(m, _default_kernel_explain_cap(n_features))
            if kernel_like_cap < n_explain:
                try:
                    X_explain, _ = sample_rows(
                        X, kernel_like_cap, stratify=effective_stratify,
                        seed=random_state, return_indices=True)
                    n_explain = len(X_explain)
                except ValueError as exc:
                    warn_messages.append(
                        "Stratified explain sampling failed; using uniform "
                        f"sampling. {exc}")
                    X_explain, _ = sample_rows(
                        X, kernel_like_cap, stratify=None,
                        seed=random_state, return_indices=True)
                    n_explain = len(X_explain)

    if backend == "kernel" and kernel_nsamples is None:
        # BACKEND TAILORING NOTE (kernel):
        # Why this exists:
        # KernelExplainer runtime scales linearly with nsamples; this heuristic
        # keeps per-feature cost bounded while limiting Monte Carlo variance.
        # What runtime/accuracy tradeoff it introduces:
        # Lower nsamples speeds up explanations but increases noise.
        # When a user might want to override it:
        # Provide kernel_nsamples for more precise attributions.
        kernel_nsamples = _default_kernel_nsamples(n_features)

    shap_outputs = []
    imp_vectors: List[np.ndarray] = []
    used_seeds: List[int] = []
    stability = {}

    for run_idx in range(K_target):
        if mode == "no_sampling":
            X_bg = X
        else:
            seed = seeds[run_idx]
            used_seeds.append(seed)
            try:
                X_bg, _ = sample_rows(
                    X, n_background, stratify=effective_stratify,
                    seed=seed, return_indices=True)
            except ValueError as exc:
                warn_messages.append(
                    f"Stratified background sampling failed; using uniform sampling. {exc}")
                X_bg, _ = sample_rows(
                    X, n_background, stratify=None,
                    seed=seed, return_indices=True)

        if backend == "kernel":
            explainer = build_kernel_explainer(model, X_bg)
            shap_result = compute_kernel_shap(
                explainer,
                X_explain,
                n_features=n_features,
                nsamples=kernel_nsamples,
                class_index=class_index)
        elif backend == "gradient":
            explainer = build_gradient_explainer(model, X_bg)
            shap_result = compute_gradient_shap(
                explainer, X_explain, batch_size=batch_size)
        else:
            if run_idx == 0 and prebuilt_explainer is not None:
                explainer = prebuilt_explainer
            else:
                explainer = build_generic_explainer(model, X_bg)
            shap_result = compute_generic_shap(explainer, X_explain)

        shap_outputs.append(shap_result)
        imp_vectors.append(_global_importance_from_shap(
            shap_result, n_features=n_features, class_index=class_index))

        if adaptive_shap_sampling and mode == "multi_sample" and len(imp_vectors) >= 2:
            stability = _build_stability(
                imp_vectors,
                feature_names=feature_names,
                topN_important=topN_important,
                warn_threshold_cv=warn_threshold_cv,
                warn_threshold_rankcorr=warn_threshold_rankcorr,
                early_stopped=False)
            if not stability["warned"]:
                stability["early_stopped"] = True
                break

    if not stability and adaptive_shap_sampling:
        stability = _build_stability(
            imp_vectors,
            feature_names=feature_names,
            topN_important=topN_important,
            warn_threshold_cv=warn_threshold_cv,
            warn_threshold_rankcorr=warn_threshold_rankcorr,
            early_stopped=False)
    elif adaptive_shap_sampling and mode == "multi_sample" and len(imp_vectors) < K_target:
        stability["early_stopped"] = True

    if (adaptive_shap_sampling and stability.get("warned")):
        warning_text = (
            "SHAP stability warning: importance variability across background "
            "samples exceeded thresholds. Consider increasing max_shap_samples, "
            "increasing K_max, adjusting max_explain_samples, or providing "
            "stratify for rare groups.")
        warn_messages.append(warning_text)
        warnings.warn(warning_text)
        if verbose:
            print(warning_text)
    elif not adaptive_shap_sampling:
        topN = min(int(topN_important), n_features)
        imp_mean = imp_vectors[0]
        top_idx = np.argsort(imp_mean)[::-1][:topN]
        stability = {
            "imp_mean": imp_mean,
            "imp_std": np.zeros_like(imp_mean),
            "imp_cv": np.zeros_like(imp_mean),
            "mean_rankcorr": float("nan"),
            "warned": False,
            "thresholds": {
                "cv": warn_threshold_cv,
                "rankcorr": warn_threshold_rankcorr,
            },
            "top_features": [feature_names[i] for i in top_idx],
            "top_indices": top_idx.tolist(),
            "max_cv": 0.0,
            "early_stopped": False,
            "skipped": True,
            "note": "stability checks skipped (adaptive_shap_sampling=False)",
        }

    if len(shap_outputs) == 1:
        shap_result_out = shap_outputs[0]
    else:
        rep_idx = _representative_run_index(
            imp_vectors, stability["imp_mean"])
        # We do not average SHAP values by default because multi-output shapes
        # vary across backends and can be memory-heavy for large datasets.
        shap_result_out = {
            "global_importance": stability["imp_mean"],
            "representative_shap": shap_outputs[rep_idx],
            "representative_index": rep_idx,
        }

    diagnostics = ShapRunDiagnostics(
        backend=backend,
        mode=mode,
        m=m,
        n_background=n_background,
        K=len(imp_vectors),
        seeds=used_seeds,
        n_explain=len(X_explain),
        stability=stability,
        warnings=warn_messages,
    )
    return shap_result_out, diagnostics


def compute_shap(
        X: Any,
        model: Any,
        backend: Literal["kernel", "gradient", "explainer"],
        y: Optional[Any] = None,
        adaptive_shap_sampling: bool = True,
        **kwargs: Any) -> Tuple[ShapResult, ShapRunDiagnostics]:
    """
    High-level wrapper for SHAP computation with optional adaptive sampling.

    Args:
        X: Input data to explain.
        model: Trained model to explain.
        backend: SHAP backend name ("kernel", "gradient", "explainer").
        y: Optional target values for stratification or diagnostics.
        adaptive_shap_sampling: Enable adaptive sampling behavior.
        **kwargs: Forwarded keyword arguments to compute_shap_adaptive.

    Returns:
        A tuple of (shap_result, diagnostics).
    """
    return compute_shap_adaptive(
        X,
        model,
        backend,
        y=y,
        adaptive_shap_sampling=adaptive_shap_sampling,
        **kwargs)


def _example_usage_adaptive_shap() -> None:
    """
    Example usage snippet with dummy model adapters.

    Args:
        None.

    Returns:
        None.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 8))
    y = (X[:, 0] + rng.normal(scale=0.1, size=500) > 0).astype(int)

    class DummySklearnModel:
        def predict(self, X_in):
            return np.sum(X_in, axis=1)

        def predict_proba(self, X_in):
            logits = X_in[:, 0]
            probs = 1 / (1 + np.exp(-logits))
            return np.column_stack([1 - probs, probs])

    class DummyTorchModel(torch.nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.linear = torch.nn.Linear(n_features, 1)

        def forward(self, X_in):
            return self.linear(X_in)

    _ = compute_shap_adaptive(
        X,
        DummySklearnModel(),
        backend="kernel",
        stratify=y,
        random_state=7,
    )
    _ = compute_shap_adaptive(
        X,
        DummyTorchModel(X.shape[1]),
        backend="gradient",
        random_state=7,
    )


class ShapEstimator(BaseEstimator):
    """
    A class for computing SHAP values and building a causal graph from them.

    Parameters
    ----------
    explainer : str, default="explainer"
        The SHAP explainer to use. Possible values are "kernel", "gradient",
        "explainer", and "tree".
    models : BaseEstimator, default=None
        The models to use for computing SHAP values. If None, a linear regression
        model is used for each feature.
    correlation_th : float, default=None
        Deprecated; retained for backward compatibility. No features are dropped.
    mean_shap_percentile : float, default=0.8
        The percentile threshold for selecting features based on their mean SHAP value.
    iters : int, default=20
        The number of iterations to use for the feature selection method.
    reciprocity : bool, default=False
        Whether to enforce reciprocity in the causal graph.
    min_impact : float, default=1e-06
        The minimum impact threshold for selecting features.
    exhaustive : bool, default=False
        Whether to use the exhaustive (recursive) method for selecting features.
        If this is True, the threshold parameter must be provided, and the
        clustering is performed until remaining values to be clustered are below
        the given threshold.
    threshold : float, default=None
        The threshold to use when exhaustive is True. If None, exception is raised.
    on_gpu : bool, default=False
        Whether to use the GPU for computing SHAP values.
    verbose : bool, default=False
        Whether to print verbose output.
    prog_bar : bool, default=True
        Whether to show a progress bar.
    silent : bool, default=False
        Whether to suppress all output.
    """

    device = utils.select_device("cpu")
    explainer = "explainer"
    models = None
    correlation_th = None
    mean_shap_percentile = 0.8
    shap_discrepancies = None
    iters = 20
    reciprocity = False
    min_impact = 1e-06
    exhaustive = False
    background_size = 200
    background_method = "sample"
    background_seed = None
    parallel_jobs = 0
    on_gpu = False
    prog_bar = True
    verbose = False
    silent = False


    def __init__(
            self,
            explainer: str = "explainer",
            models: Optional[BaseEstimator] = None,
            correlation_th: Optional[float] = None,
            mean_shap_percentile: float = 0.8,
            iters: int = 20,
            reciprocity: bool = False,
            min_impact: float = 1e-06,
            exhaustive: bool = False,
            background_size: Optional[int] = 200,
            background_method: str = "sample",
            background_seed: Optional[int] = None,
            parallel_jobs: int = 0,
            on_gpu: bool = False,
            verbose: bool = False,
            prog_bar: bool = True,
            silent: bool = False) -> None:
        """
        Initialize the ShapEstimator object.

        Parameters
        ----------
        explainer : str, default="explainer"
            The SHAP explainer to use. Possible values are "kernel", "gradient",
            "explainer", and "tree".
        models : BaseEstimator, default=None
            The models to use for computing SHAP values. If None, a linear regression
            model is used for each feature.
        correlation_th : float, default=None
            Deprecated; retained for backward compatibility. No features are dropped.
        mean_shap_percentile : float, default=0.8
            The percentile threshold for selecting features based on their
            mean SHAP value.
        iters : int, default=20
            The number of iterations to use for the feature selection method.
        reciprocity : bool, default=False
            Whether to enforce reciprocity in the causal graph.
        min_impact : float, default=1e-06
            The minimum impact threshold for selecting features.
        exhaustive : bool, default=False
            Whether to use the exhaustive (recursive) method for selecting features.
            If this is True, the threshold parameter must be provided, and the
            clustering is performed until remaining values to be clustered are below
            the given threshold.
        background_size : int, optional
            Maximum number of background samples used by SHAP explainers. If None
            or larger than the available samples, all rows are used.
        background_method : str, default="sample"
            Background selection strategy: "sample" or "kmeans".
        background_seed : int, optional
            Random seed for background sampling when using "sample".
        threshold : float, default=None
            The threshold to use when exhaustive is True. If None, exception is raised.
        on_gpu : bool, default=False
            Whether to use the GPU for computing SHAP values.
        verbose : bool, default=False
            Whether to print verbose output.
        prog_bar : bool, default=True
            Whether to show a progress bar.
        silent : bool, default=False
            Whether to suppress all output.

        Args:
            explainer: SHAP explainer name to use.
            models: Optional estimator collection used for SHAP computation.
            correlation_th: Deprecated; retained for backward compatibility.
            mean_shap_percentile: Percentile used to compute SHAP threshold.
            iters: Number of iterations for feature selection.
            reciprocity: Whether to enforce reciprocal edges.
            min_impact: Minimum SHAP impact threshold for selection.
            exhaustive: Whether to run exhaustive feature selection.
            background_size: Background sample size for SHAP explainers.
            background_method: Background selection method ("sample" or "kmeans").
            background_seed: Random seed for background sampling.
            parallel_jobs: Parallel worker count.
            on_gpu: Whether to use GPU for SHAP computation.
            verbose: Enable verbose output.
            prog_bar: Whether to show progress bars.
            silent: Suppress all output.

        Returns:
            None.
        """
        self.explainer = explainer
        self.models = models
        self.correlation_th = correlation_th
        self.mean_shap_percentile = mean_shap_percentile
        self.iters = iters
        self.reciprocity = reciprocity
        self.min_impact = min_impact
        self.exhaustive = exhaustive
        self.background_size = background_size
        self.background_method = background_method
        self.background_seed = background_seed
        self.corr_matrix = None
        self.correlated_features = defaultdict(list)
        self.parallel_jobs = parallel_jobs
        self.on_gpu = on_gpu
        self.verbose = verbose
        self.prog_bar = prog_bar
        self.silent = silent

        self._fit_desc = f"Running SHAP explainer ({self.explainer})"
        self._pred_desc = "Building graph skeleton"

    def _select_background(
            self,
            X_train: np.ndarray,
            allow_kmeans: bool = True) -> np.ndarray:
        """
        Select the background samples for SHAP explainers.

        Parameters
        ----------
        X_train : np.ndarray
            The training data.
        allow_kmeans : bool, default=True
            Whether to allow kmeans background selection.
        Returns
        -------
        np.ndarray
            The selected background samples.

        Args:
            X_train: Training data used to draw background samples.
            allow_kmeans: Whether kmeans-based sampling is permitted.

        Returns:
            Selected background samples as a numpy array.
        """
        if self.background_size is None:
            return X_train
        if not isinstance(self.background_size, int) or self.background_size <= 0:
            raise ValueError("background_size must be a positive integer or None.")
        n_rows = X_train.shape[0]
        if n_rows <= self.background_size:
            return X_train
        method = self.background_method or "sample"
        if method == "kmeans":
            if allow_kmeans:
                return shap.kmeans(X_train, self.background_size)
            rng = np.random.default_rng(self.background_seed)
            indices = rng.choice(n_rows, size=self.background_size, replace=False)
            return X_train[indices]
        if method == "sample":
            rng = np.random.default_rng(self.background_seed)
            indices = rng.choice(n_rows, size=self.background_size, replace=False)
            return X_train[indices]
        raise ValueError(
            "background_method must be 'sample' or 'kmeans'."
        )

    def _call_explainer(
            self,
            explainer: Any,
            X_test: Any,
            silent: bool = True) -> Any:
        """
        Invoke a SHAP explainer with optional silent mode.

        Args:
            explainer: SHAP explainer instance.
            X_test: Data to explain.
            silent: Whether to suppress explainer output.

        Returns:
            SHAP output from the explainer.
        """
        if not silent:
            return explainer(X_test)
        try:
            signature = inspect.signature(explainer.__call__)
        except (TypeError, ValueError):
            return explainer(X_test)
        call_kwargs = {}
        if "silent" in signature.parameters:
            call_kwargs["silent"] = True
        try:
            return explainer(X_test, **call_kwargs)
        except Exception as exc:                         # pylint: disable=broad-except
            explainer_error = getattr(getattr(shap, "utils", None), "_exceptions", None)
            explainer_error_cls = getattr(explainer_error, "ExplainerError", None)
            if explainer_error_cls and isinstance(exc, explainer_error_cls) and \
                    "check_additivity" in signature.parameters:
                # BACKEND TAILORING NOTE (explainer):
                # Why this exists:
                # TreeExplainer additivity checks can fail on numerically noisy
                # models even with correct input shapes, halting SHAP runs.
                # What runtime/accuracy tradeoff it introduces:
                # Disabling the check skips a consistency assertion but keeps
                # the attribution values for downstream analysis.
                # When a user might want to override it:
                # If exact additivity is required, keep the default behavior
                # and fix the upstream model/data mismatch instead.
                call_kwargs["check_additivity"] = False
                return explainer(X_test, **call_kwargs)
            raise

    def __str__(self) -> str:
        """
        Return a compact string representation for logging.

        Args:
            None.

        Returns:
            String representation of the estimator.
        """
        return utils.stringfy_object(self)

    # Define the process_target function at the top level
    @staticmethod
    def _shap_fit_target_variable(
        target_name: str,
        models: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str],
        on_gpu: bool,
        verbose: bool,
        run_selected_shap_explainer_func: Callable[..., Any],
    ) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single target in the SHAP fit process.

        Args:
            target_name: Target variable name to explain.
            models: Model container used for SHAP computation.
            X_train: Training data including all features.
            X_test: Test data including all features.
            feature_names: Full list of feature names.
            on_gpu: Whether to place torch models on GPU.
            verbose: Whether to print debug output.
            run_selected_shap_explainer_func: Callable that runs SHAP.

        Returns:
            A tuple of (target_name, shap_values, feature_order, shap_means).
        """
        # Get the model and the data (tensor form)
        if hasattr(models.regressor[target_name], "model"):
            model = models.regressor[target_name].model
            model = model.cuda() if on_gpu else model.cpu()
        else:
            model = models.regressor[target_name]

        X_train_target = X_train.drop(target_name, axis=1).values
        X_test_target = X_test.drop(target_name, axis=1).values

        # Run the selected SHAP explainer
        shap_values_target = run_selected_shap_explainer_func(
            target_name, model, X_train_target, X_test_target
        )

        # Create the order list of features, in decreasing mean SHAP value
        feature_order_target = np.argsort(
            np.sum(np.abs(shap_values_target), axis=0))
        shap_mean_values_target = np.abs(shap_values_target).mean(0)

        # Optionally, print verbose output
        if verbose:
            feature_order_str = ", ".join(
                str(idx) for idx in feature_order_target.tolist()
            )
            print(f"    > Feature order for '{target_name}': [{feature_order_str}]")
            print(f"      Target({target_name}) -> ", end="")
            srcs = [src for src in feature_names if src != target_name]
            shap_mean_values_display = np.asarray(shap_mean_values_target)
            if shap_mean_values_display.ndim > 1:
                # Reduce any extra axes so each feature prints a single scalar.
                shap_mean_values_display = shap_mean_values_display.mean(
                    axis=tuple(range(1, shap_mean_values_display.ndim))
                )
            for i in range(len(shap_mean_values_display)):
                value = float(shap_mean_values_display[i])
                print(f"{srcs[i]}:{value:.3f};", end="")
            print()

        # Return results
        return target_name, shap_values_target, feature_order_target, \
            shap_mean_values_target


    def fit(self, X: pd.DataFrame) -> "ShapEstimator":
        """
        Fit the ShapleyExplainer model to the given dataset.

        Parameters:
        - X: The input dataset.

        Returns:
        - self: The fitted ShapleyExplainer model.

        Args:
            X: Input dataset used to compute SHAP values.

        Returns:
            The fitted estimator instance.
        """
        assert self.models is not None, "shap.models must be set"

        self.feature_names = list(self.models.regressor.keys())  # type: ignore
        self.shap_explainer = {}
        self.shap_values = {}
        self.shap_mean_values = {}
        self.feature_order = {}
        self.all_mean_shap_values = np.empty((0,), dtype=np.float16)
        # Initialize the progress bar
        if self.prog_bar and not self.verbose:
            caller_name = self._get_method_caller_name()
            pbar_name = f"({caller_name}) SHAP_fit"
            pbar = ProgBar().start_subtask(pbar_name, len(self.feature_names))
        else:
            pbar = None

        self.X_train, self.X_test = train_test_split(
            X, test_size=min(0.2, 250 / len(X)), random_state=42
        )

        # Prepare arguments for partial function
        partial_process_target = partial(
            ShapEstimator._shap_fit_target_variable,
            models=self.models,
            X_train=self.X_train,
            X_test=self.X_test,
            feature_names=self.feature_names,
            on_gpu=self.on_gpu,
            verbose=self.verbose,
            run_selected_shap_explainer_func=self._run_selected_shap_explainer,
        )

        if self.parallel_jobs != 0:
            if self.parallel_jobs == -1:
                nr_processes = min(multiprocessing.cpu_count(), len(self.feature_names))
            else:
                nr_processes = min(self.parallel_jobs, multiprocessing.cpu_count())

            # Use 'spawn' start method to avoid semaphore leaks
            with get_context('spawn').Pool(processes=nr_processes) as pool:
                results = []
                for result in pool.imap_unordered(partial_process_target, self.feature_names):
                    results.append(result)
                    if pbar:
                        pbar.update_subtask(pbar_name, len(results))  # type: ignore
                pbar.remove(pbar_name) if pbar else None            # type: ignore
                pbar = None
                pool.close()
                pool.join()
        else:
            # Sequential processing
            results = []
            for target_name in self.feature_names:
                result = partial_process_target(target_name)
                results.append(result)
                if pbar:
                    pbar.update_subtask(pbar_name, len(results))  # type: ignore

        # Collect results
        for target_name, shap_values_target, feature_order_target, shap_mean_values_target in results:
            self.shap_values[target_name] = shap_values_target
            self.feature_order[target_name] = feature_order_target
            self.shap_mean_values[target_name] = shap_mean_values_target

            # If dimensions of all_mean_shap_values is not (m, 1), reshape it
            if len(self.all_mean_shap_values.shape) == 1:
                self.all_mean_shap_values = self.all_mean_shap_values.reshape(-1, 1)

            if len(shap_mean_values_target.shape) == 1:
                shap_mean_values_target = shap_mean_values_target.reshape(-1, 1)

            self.all_mean_shap_values = np.concatenate(
                (self.all_mean_shap_values, shap_mean_values_target)
            )

        if pbar:
            pbar.remove(pbar_name)  # type: ignore

        self.all_mean_shap_values = self.all_mean_shap_values.flatten()
        self._compute_scaled_shap_threshold()

        self.is_fitted_ = True

        return self

    def _compute_scaled_shap_threshold(self) -> None:
        """
        Compute the scaled SHAP threshold based on the given percentile.
        If the percentile is 0.0 or None, then the threshold is set to 0.0.

        Args:
            None.

        Returns:
            None.
        """
        if self.mean_shap_percentile:
            self.mean_shap_threshold = np.quantile(
                self.all_mean_shap_values, self.mean_shap_percentile)
        else:
            self.mean_shap_threshold = 0.0

    def _run_selected_shap_explainer(
            self,
            target_name: str,
            model: Any,
            X_train: np.ndarray,
            X_test: np.ndarray) -> np.ndarray:
        """
        Run the selected SHAP explainer, according to the given parameters.

        Parameters
        ----------
        target_name : str
            The name of the target feature.
        model : torch.nn.Module
            The model for the given target.
        X_train : PyTorch.Tensor object
            The training data.
        X_test : PyTorch.Tensor object
            The testing data.

        Returns
        -------
        shap_values : np.ndarray
            The SHAP values for the given target.

        Args:
            target_name: Name of the target feature.
            model: Trained model for the target.
            X_train: Training data used for background.
            X_test: Data to explain.

        Returns:
            SHAP values for the target feature.
        """
        if self.explainer == "kernel":
            background = self._select_background(X_train)
            self.shap_explainer[target_name] = shap.KernelExplainer(
                model.predict, background)
            shap_values = self.shap_explainer[target_name].\
                shap_values(X_test)[0]
        elif self.explainer == "gradient":
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            background = self._select_background(X_train, allow_kmeans=False)
            background = np.asarray(background, dtype=np.float32)
            X_test = np.asarray(X_test, dtype=np.float32)
            X_train_tensor = torch.from_numpy(background).float()
            X_test_tensor = torch.from_numpy(X_test).float()
            model_device = self.device
            if hasattr(model, "parameters"):
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    model_device = self.device
            self.shap_explainer[target_name] = shap.GradientExplainer(
                model, [X_train_tensor.to(model_device)])
            shap_values = self.shap_explainer[target_name](
                [X_test_tensor.to(model_device)]).values
        elif self.explainer == "explainer":
            background = self._select_background(X_train)
            if isinstance(model, torch.nn.Module):
                self.shap_explainer[target_name] = shap.Explainer(
                    model.predict, background)
            else:
                try:
                    self.shap_explainer[target_name] = shap.Explainer(
                        model, background)
                except Exception:
                    self.shap_explainer[target_name] = shap.Explainer(
                        model.predict, background)
            explanation = self._call_explainer(
                self.shap_explainer[target_name], X_test, silent=True)
            shap_values = explanation.values
        elif self.explainer == "tree":
            background = self._select_background(X_train)
            self.shap_explainer[target_name] = shap.TreeExplainer(
                model, background)
            explanation = self._call_explainer(
                self.shap_explainer[target_name], X_test, silent=True)
            shap_values = (
                explanation.values
                if hasattr(explanation, "values")
                else explanation
            )
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
        else:
            raise ValueError(
                f"Unknown explainer: {self.explainer}. "
                f"Please select one of: kernel, gradient, explainer, tree.")

        return shap_values

    def predict(
            self,
            X: pd.DataFrame,
            root_causes: list[str]|None = None,
            prior: list[list[str]]|None = None) -> nx.DiGraph:
        """
        Builds a causal graph from the shap values using a selection mechanism based
        on clustering, knee or abrupt methods.

        Parameters
        ----------
        X : pd.DataFrame
            The input data. Consists of all the features in a pandas DataFrame.
        root_causes : List[str], optional
            The root causes of the graph. If None, all features are considered as
            root causes, by default None.
        prior : List[List[str]], optional
            The prior knowledge about the connections between the features. If None,
            all features are considered as valid candidates for the connections, by
            default None.

        Returns
        -------
        nx.DiGraph
            The causal graph.

        Args:
            X: Input dataset used for prediction.
            root_causes: Optional list of root-cause feature names.
            prior: Optional prior knowledge constraints.

        Returns:
            The inferred causal graph.
        """
        if self.verbose:
            print("-----\nshap.predict()")

        if not self.is_fitted_:
            raise ValueError("This Rex instance is not fitted yet. \
                Call 'fit' with appropriate arguments before using this estimator.")
        self.prior = prior

        # Recompute mean_shap_percentile here, in case it was changed
        self._compute_scaled_shap_threshold()

        # Who is calling me?
        caller_name = self._get_method_caller_name()

        if self.prog_bar and (not self.verbose):
            pbar_name = f"({caller_name}) SHAP_predict"
            pbar = ProgBar().start_subtask(pbar_name, 4 + len(self.feature_names))
        else:
            pbar = None

        # Reshape SHAP values if necessary
        for feature in self.feature_names:
            if self.shap_values[feature].shape[-1] == 1:
                self.shap_values[feature] = self.shap_values[feature]\
                    .reshape(self.shap_values[feature].shape[:-1])

        # Compute error contribution at this stage, since it needs the individual
        # SHAP values
        self.compute_error_contribution()
        pbar.update_subtask(pbar_name, 1) if pbar else None     # type: ignore

        self._compute_discrepancies(self.X_test)                # type: ignore
        pbar.update_subtask(pbar_name, 2) if pbar else None     # type: ignore

        self.connections = {}
        for target_idx, target in enumerate(self.feature_names):
            # The valid parents are the features that are in the same level of the
            # hierarchy as the target, or in previous levels. In case prior is not
            # provided, all features are valid candidates.
            candidate_causes = utils.valid_candidates_from_prior(
                self.feature_names, target, self.prior)

            # feature_names_wo_target = [
            #     f for f in candidate_causes if f != target]

            # Debug output
            if self.verbose:
                print(
                    f"> Selecting features for target {target}...")
                print(f"  > Candidate causes for target '{target}': {candidate_causes}")

            # Select the features that are connected to the target
            self.connections[target] = select_features(
                values=self.shap_values[target],
                feature_names=candidate_causes, # feature_names_wo_target,
                min_impact=self.min_impact,
                exhaustive=self.exhaustive,
                threshold=float(self.mean_shap_threshold),
                verbose=self.verbose)
            pbar.update_subtask(
                pbar_name, target_idx + 3) if pbar else None     # type: ignore

        dag = utils.digraph_from_connected_features(
            X, self.feature_names, self.models, self.connections, root_causes, prior,
            reciprocity=self.reciprocity, anm_iterations=self.iters,
            verbose=self.verbose)
        pbar.update_subtask(pbar_name, # type: ignore
            len( self.feature_names) + 3) if pbar else None

        dag = utils.break_cycles_if_present(
            dag, self.shap_discrepancies,                       # type: ignore
            self.prior, verbose=self.verbose)
        pbar.update_subtask(pbar_name,                          # type: ignore
            len(self.feature_names) + 4) if pbar else None

        pbar.remove(pbar_name) if pbar else None                # type: ignore

        return dag

    def adjust(
            self,
            graph: nx.DiGraph,
            increase_tolerance: float = 0.0,
            sd_upper: float = 0.1) -> nx.DiGraph:
        """
        Adjust graph edges based on SHAP discrepancy thresholds.

        Args:
            graph: Graph to adjust.
            increase_tolerance: Tolerance scaling applied to discrepancy bounds.
            sd_upper: Upper bound for discrepancy difference.

        Returns:
            Adjusted directed graph.
        """

        # self._compute_shap_discrepancies(X)
        new_graph = self._adjust_edges_from_shap_discrepancies(
            graph, increase_tolerance, sd_upper)
        return new_graph

    def _compute_discrepancies(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the discrepancies between the SHAP values and the target values
        for all features and all targets.

        Parameters
        ----------
        X : pd.DataFrame
            The input data. Consists of all the features in a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the discrepancies for all features and all targets.

        Args:
            X: Input dataset with all features.

        Returns:
            DataFrame of discrepancies indexed by target and parent features.
        """
        if not self.is_fitted_:
            raise ValueError("This Rex instance is not fitted yet. \
                Call 'fit' with appropriate arguments before using this estimator.")
        self.discrepancies = pd.DataFrame(columns=self.feature_names)
        self.shap_discrepancies = defaultdict(dict)
        for target_name in self.feature_names:

            X_features = X.drop(target_name, axis=1)
            y = X[target_name].values

            feature_names = [
                f for f in self.feature_names if f != target_name]

            self.discrepancies.loc[target_name] = 0
            self.shap_discrepancies[target_name] = {}

            # Loop through all features and compute the discrepancy
            for parent_name in feature_names:
                # Take the data that is needed at this iteration
                parent_data = X_features[parent_name].values
                parent_pos = feature_names.index(parent_name)
                shap_data = self.shap_values[target_name][:, parent_pos]

                # Form three vectors to compute the discrepancy
                x = parent_data.reshape(-1, 1)
                s = shap_data.reshape(-1, 1)

                # Compute the discrepancy
                self.shap_discrepancies[target_name][parent_name] = \
                    self._compute_discrepancy(
                        x, y, s,
                        target_name,
                        parent_name)
                SD = self.shap_discrepancies[target_name][parent_name]
                self.discrepancies.loc[target_name,
                                       parent_name] = SD.shap_discrepancy

        return self.discrepancies

    def _compute_discrepancy(
            self,
            x: Union[np.ndarray, pd.Series, pd.DataFrame],
            y: Union[np.ndarray, pd.Series, pd.DataFrame],
            s: Union[np.ndarray, pd.Series, pd.DataFrame],
            target_name: str,
            parent_name: str) -> ShapDiscrepancy:
        """
        Compute discrepancy statistics between SHAP values and targets.

        Args:
            x: Parent feature values.
            y: Target feature values.
            s: SHAP values associated with the parent.
            target_name: Name of the target feature.
            parent_name: Name of the parent feature.

        Returns:
            ShapDiscrepancy dataclass with per-feature diagnostics.
        """
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x = x.values
        elif not isinstance(x, np.ndarray):
            x = np.array(x)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        elif not isinstance(y, np.ndarray):
            y = np.array(y)

        X = sm.add_constant(x)
        model_y = sm.OLS(y, X).fit()
        model_s = sm.OLS(s, X).fit()

        # Heteroskedasticity tests:
        # The null hypothesis (H0): Signifies that Homoscedasticity is present
        # The alternative hypothesis (H1): Signifies that Heteroscedasticity is present
        # If the p-value is less than the significance level (0.05), we reject the
        # null hypothesis and conclude that heteroscedasticity is present.
        def _as_float(value) -> float:
            if isinstance(value, (np.ndarray, list, tuple)):
                arr = np.asarray(value)
                if arr.size == 0:
                    return float("nan")
                return float(arr.reshape(-1)[0])
            return float(value)

        try:
            test_shap = sms.het_breuschpagan(model_s.resid, X)
        except ValueError:
            test_shap = (0, 0)
        shap_p_value = _as_float(test_shap[1])
        shap_heteroskedasticity = bool(shap_p_value < 0.05)

        try:
            test_parent = sms.het_breuschpagan(model_y.resid, X)
        except ValueError:
            test_parent = (0, 0)
        parent_p_value = _as_float(test_parent[1])
        parent_heteroskedasticity = bool(parent_p_value < 0.05)

        s_flat = np.asarray(s).reshape(-1)
        y_flat = np.asarray(y).reshape(-1)
        finite_mask = np.isfinite(s_flat) & np.isfinite(y_flat)
        # Guard against NaNs in SHAP outputs to keep downstream metrics stable.
        s_eval = s_flat[finite_mask]
        y_eval = y_flat[finite_mask]
        if s_eval.size >= 2:
            corr_result = spearmanr(s_eval, y_eval)
            corr_value = corr_result.correlation if hasattr(
                corr_result, "correlation") else corr_result[0]
            corr_value = _as_float(corr_value)
            if not np.isfinite(corr_value):
                corr_value = 0.0
        else:
            corr_value = 0.0
        discrepancy = 1 - np.abs(corr_value)
        # The p-value is below 5%: we reject the null hypothesis that the two
        # distributions are the same, with 95% confidence.
        if s_eval.size >= 1:
            ks_result = kstest(s_eval, y_eval)
            ks_pvalue = ks_result.pvalue if hasattr(
                ks_result, "pvalue") else ks_result[1]
            ks_pvalue = _as_float(ks_pvalue)
        else:
            ks_pvalue = float("nan")

        return ShapDiscrepancy(
            target=target_name,
            parent=parent_name,
            shap_heteroskedasticity=shap_heteroskedasticity,
            parent_heteroskedasticity=parent_heteroskedasticity,
            shap_p_value=shap_p_value,
            parent_p_value=parent_p_value,
            shap_model=model_s,
            parent_model=model_y,
            shap_discrepancy=discrepancy,
            shap_correlation=corr_value,
            shap_gof=r2_score(y_eval, s_eval) if s_eval.size >= 2 else 0.0,
            ks_pvalue=ks_pvalue,
            ks_result="Equal" if ks_pvalue > 0.05 else "Different"
        )

    def _adjust_edges_from_shap_discrepancies(
            self,
            graph: nx.DiGraph,
            increase_tolerance: float = 0.0,
            sd_upper: float = 0.1) -> nx.DiGraph:
        """
        Adjust the edges of the graph based on the discrepancy index. This method
        removes edges that have a discrepancy index larger than the given standard
        deviation tolerance. The method also removes edges that have a discrepancy
        index larger than the discrepancy index of the target.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph to adjust the edges of.
        increase_tolerance : float, optional
            The tolerance for the increase in the discrepancy index, by default 0.0.
        sd_upper : float, optional
            The upper bound for the Shap Discrepancy, by default 0.1. This is the max
            difference between the SHAP Discrepancy in both causal directions.

        Returns
        -------
        networkx.DiGraph
            The graph with the edges adjusted.

        Args:
            graph: Graph to adjust.
            increase_tolerance: Tolerance scaling for discrepancy bounds.
            sd_upper: Upper bound for discrepancy difference.

        Returns:
            The adjusted directed graph.
        """

        new_graph = nx.DiGraph()
        new_graph.add_nodes_from(graph.nodes(data=True))
        new_graph.add_edges_from(graph.edges())
        edges_reversed = []

        # Experimental
        if self._increase_upper_tolerance(self.discrepancies):
            increase_upper_tolerance = True
        else:
            increase_upper_tolerance = False

        for target in self.feature_names:
            target_mean = np.mean(self.discrepancies.loc[target].values)
            # Experimental
            if increase_upper_tolerance:
                tolerance = target_mean * increase_tolerance
            else:
                tolerance = 0.0

            # Iterate over the features and check if the edge should be reversed.
            for feature in self.feature_names:
                # If the edge is already reversed, skip it.
                if (target, feature) in edges_reversed or \
                        feature == target or \
                        not new_graph.has_edge(feature, target):
                    continue

                # Normalize pandas/numpy scalars to plain floats for type checkers.
                forward_sd = float(np.asarray(
                    self.discrepancies.at[target, feature]).item())
                reverse_sd = float(np.asarray(
                    self.discrepancies.at[feature, target]).item())
                diff = np.abs(forward_sd - reverse_sd)
                vector = [forward_sd, reverse_sd, diff, 0., 0., 0., 0.]

                # If the forward standard deviation is less than the reverse
                # standard deviation and within the tolerance range, attempt
                # to reverse the edge.
                if (forward_sd < reverse_sd) and \
                        (forward_sd < (target_mean + tolerance)):
                    # If the difference between the standard deviations is within
                    # the acceptable range, reverse the edge and check for cycles
                    # in the graph.
                    if diff < sd_upper and diff > 0.002:
                        new_graph.remove_edge(feature, target)
                        new_graph.add_edge(target, feature)
                        edges_reversed.append((feature, target))
                        cycles = list(nx.simple_cycles(new_graph))
                        # If reversing the edge creates a cycle that includes both
                        # the target and feature nodes, reverse the edge back to
                        # its original direction and log the decision as discarded.
                        if len(cycles) > 0 and \
                                self._nodes_in_cycles(cycles, feature, target):
                            new_graph.remove_edge(target, feature)
                            edges_reversed.remove((feature, target))
                            new_graph.add_edge(feature, target)
                            self._debugmsg("Discarded(cycles)",
                                           target, target_mean, feature, tolerance,
                                           vector, sd_upper, cycles)
                        # If reversing the edge does not create a cycle, log the
                        # decision as reversed.
                        else:
                            self._debugmsg("(*) Reversed edge",
                                           target, target_mean, feature, tolerance,
                                           vector, sd_upper, cycles)
                    # If the difference between the standard deviations is not
                    # within the acceptable range, log the decision as discarded.
                    else:
                        self._debugmsg("Outside tolerance",
                                       target, target_mean, feature, tolerance,
                                       vector, sd_upper, [])
                # If the forward standard deviation is greater than the reverse
                # standard deviation and within the tolerance range, log the
                # decision as ignored.
                else:
                    self._debugmsg("Ignored edge",
                                   target, target_mean, feature, tolerance,
                                   vector, sd_upper, [])

        return new_graph

    def _nodes_in_cycles(
            self,
            cycles: List[List[str]],
            feature: str,
            target: str) -> bool:
        """
        Check if the given nodes are in any of the cycles.

        Args:
            cycles: List of cycles from the graph.
            feature: Feature node name.
            target: Target node name.

        Returns:
            True if both nodes appear in the same cycle.
        """
        for cycle in cycles:
            if feature in cycle and target in cycle:
                return True
        return False

    def _increase_upper_tolerance(self, discrepancies: pd.DataFrame) -> bool:
        """
        Increase the upper tolerance if the discrepancy matrix properties are
        suspicious. We found these suspicious values empirically in the polymoial case.

        Args:
            discrepancies: Discrepancy matrix data.

        Returns:
            True if the tolerance should be increased.
        """
        D = discrepancies.values
        D = np.nan_to_num(D)
        det = np.linalg.det(D)
        norm = np.linalg.norm(D)
        cond = np.linalg.cond(D)
        m1 = "(*)" if det < -0.5 else "   "
        m2 = "(*)" if norm > .7 else "   "
        m3 = "(*)" if cond > 1500 else "   "
        if self.verbose:
            print(f"    {m2}{norm=:.2f} & ({m1}{det=:.2f} | {m3}{cond=:.2f})")
        if norm > 7.0 and (det < -0.5 or cond > 2000):
            return True
        return False

    def _input_vector(
            self,
            discrepancies: pd.DataFrame,
            target: str,
            feature: str,
            target_mean: float) -> np.ndarray:
        """
        Builds a vector with the values computed from the discrepancy index.
        Used to feed the model in _adjust_from_model method.

        Args:
            discrepancies: Discrepancy matrix data.
            target: Target feature name.
            feature: Feature name.
            target_mean: Mean discrepancy for the target.

        Returns:
            Feature vector used by downstream models.
        """
        source_mean = np.mean(discrepancies.loc[feature].values)
        forward_sd = discrepancies.loc[target, feature]
        reverse_sd = discrepancies.loc[feature, target]
        diff = np.abs(forward_sd - reverse_sd)
        sdiff = np.abs(forward_sd + reverse_sd)
        d1 = diff / sdiff
        d2 = diff / forward_sd

        # Build the input for the model
        input_vector = np.array(
            [forward_sd, reverse_sd, diff, d1, d2,
             target_mean, source_mean])

        return input_vector

    def _adjust_predictions_shape(
            self,
            predictions: Union[np.ndarray, List[np.ndarray]],
            target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Normalize predictions into a 2D array with the target shape.

        Args:
            predictions: Raw predictions to normalize.
            target_shape: Desired output shape.

        Returns:
            Normalized numpy array matching target_shape.
        """
        # Concatenate if predictions is a list
        if isinstance(predictions, list):
            predictions = np.concatenate(predictions)
        else:
            predictions = np.array(predictions)

        # Reshape if necessary
        if predictions.shape != target_shape:
            # Flatten
            predictions = predictions.reshape(target_shape)

        return predictions

    def compute_error_contribution(self) -> pd.DataFrame:
        """
        Computes the error contribution of each feature for each target.
        If this value is positive, then it means that, on average, the presence of
        the feature in the model leads to a higher error. Thus, without that feature,
        the prediction would have been generally better. In other words, the feature
        is making more harm than good!
        On the contrary, the more negative this value, the more beneficial
        the feature is for the predictions since its presence leads to smaller errors.

        Returns:
        --------
        err_contrib: pd.DataFrame
            Error contribution of each feature for each target.

        Args:
            None.

        Returns:
            DataFrame of per-feature error contributions.
        """
        error_contribution = dict()
        predictions = self.models.predict(self.X_test)   # type: ignore

        predictions = self._adjust_predictions_shape(
            predictions, (self.X_test.shape[0], self.X_test.shape[1]))

        y_hat = pd.DataFrame(predictions, columns=self.feature_names)
        y_true = self.X_test
        for target in self.feature_names:
            shap_values = pd.DataFrame(
                self.shap_values[target],
                columns=[c for c in self.feature_names if c != target])
            error_contribution[target] = self._individual_error_contribution(
                shap_values, y_true[target], y_hat[target])
            # Add a 0.0 value at index target in the error contribution series
            error_contribution[target] = pd.concat([
                error_contribution[target],
                pd.Series({target: 0.0})
            ])
            # Sort the series by index
            error_contribution[target] = error_contribution[target].sort_index()

        self.error_contribution = pd.DataFrame(error_contribution)
        return self.error_contribution

    def _individual_error_contribution(
            self,
            shap_values: pd.DataFrame,
            y_true: pd.Series,
            y_pred: pd.Series) -> pd.Series:
        """
        Compute the error contribution of each feature.
        If this value is positive, then it means that, on average, the presence of
        the feature in the model leads to a higher error. Thus, without that feature,
        the prediction would have been generally better. In other words, the feature
        is making more harm than good!
        On the contrary, the more negative this value, the more beneficial the
        feature is for the predictions since its presence leads to smaller errors.

        Parameters:
        -----------
        shap_values: pd.DataFrame
            Shap values for a given target.
        y_true: pd.Series
            Ground truth values for a given target.
        y_pred: pd.Series
            Predicted values for a given target.

        Returns:
        --------
        error_contribution: pd.Series
            Error contribution of each feature.

        Args:
            shap_values: SHAP values for a given target.
            y_true: Ground truth values for a given target.
            y_pred: Predicted values for a given target.

        Returns:
            Series of per-feature error contributions.
        """
        abs_error = (y_true - y_pred).abs()
        y_pred_wo_feature = shap_values.apply(lambda feature: y_pred - feature)
        abs_error_wo_feature = y_pred_wo_feature.apply(
            lambda feature: (y_true-feature).abs())
        error_diff = abs_error_wo_feature.apply(
            lambda feature: abs_error - feature)
        ind_error_contribution = error_diff.mean()

        return ind_error_contribution

    def _debugmsg(
            self,
            msg: str,
            target: str,
            target_threshold: float,
            feature: str,
            tolerance: float,
            vector: Union[List[float], np.ndarray],
            sd_upper: float,
            cycles: List[List[str]]) -> None:
        """
        Print verbose diagnostics for edge adjustment decisions.

        Args:
            msg: Debug message prefix.
            target: Target node name.
            target_threshold: Threshold for the target discrepancy.
            feature: Feature node name.
            tolerance: Current tolerance value.
            vector: Diagnostic vector of discrepancy stats.
            sd_upper: Upper bound for discrepancy differences.
            cycles: List of detected cycles.

        Returns:
            None.
        """
        if not self.verbose:
            return
        forward_sd, reverse_sd, diff, _, _, _, _ = vector
        fwd_bwd = f"{GREEN}<{RESET}" if forward_sd < reverse_sd else f"{RED}≮{RESET}"
        fwd_tgt = f"{GREEN}<{RESET}" if forward_sd < target_threshold + \
            tolerance else f"{RED}>{RESET}"
        diff_upper = f"{GREEN}<{RESET}" if diff < sd_upper else f"{RED}>{RESET}"
        print(f" -  {msg:<17s}: {feature} -> {target}",
              f"fwd|bwd({forward_sd:.3f}{fwd_bwd}{reverse_sd:.3f});",
              f"fwd{fwd_tgt}⍴({target_threshold:.3f}+{tolerance:.2f});",
              f"𝛿({diff:.3f}){diff_upper}Up({sd_upper:.2f}); "
              # f"𝛿({diff:.3f}){diff_tol}tol({sd_tol:.2f});",
              f"µ:{forward_sd/target_threshold:.3f}"
              )
        if len(cycles) > 0 and self._nodes_in_cycles(cycles, feature, target):
            print(f"    ~~ Cycles: {cycles}")

    def _get_method_caller_name(self) -> str:
        """
        Determine the name of the method that called it. It does this by using
        the inspect module to get the outermost frame of the call stack and then
        extracting the name of the third frame. If the name is either __call__ or
        _run_step, it returns "ReX". If any exception occurs during this
        process, it returns "unknown".

        Args:
            None.

        Returns:
            Caller method name or "unknown".
        """
        try:
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            caller_name = calframe[2][3]
            if caller_name == "__call__" or caller_name == "_run_step":
                caller_name = "ReX"
        except Exception:                               # pylint: disable=broad-except
            caller_name = "unknown"
        return caller_name

    def _plot_shap_summary(
            self,
            target_name: str,
            ax: Optional[Any],
            max_features_to_display: int = 20,
            **kwargs: Any) -> Any:
        """
        Plots the summary of the SHAP values for a given target.

        Arguments:
        ----------
            feature_names: List[str]
                The names of the features.
            mean_shap_values: np.array
                The mean SHAP values for each feature.
            feature_inds: np.array
                The indices of the features to be plotted.
            selected_features: List[str]
                The names of the selected features.
            ax: Axis
                The axis in which to plot the summary. If None, a new figure is created.
            **kwargs: Dict
                Additional arguments to be passed to the plot.

        Args:
            target_name: Target feature name.
            ax: Matplotlib axis to plot into (optional).
            max_features_to_display: Max number of features to show.
            **kwargs: Additional plotting arguments.

        Returns:
            Matplotlib figure with the SHAP summary plot.
        """

        figsize_ = kwargs.get('figsize', (6, 3))
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize_)

        feature_inds = self.feature_order[target_name][:max_features_to_display]
        feature_names = [
            f for f in self.feature_names if f != target_name]
        selected_features = list(self.connections[target_name])

        y_pos = np.arange(len(feature_inds))
        ax.grid(True, axis='x')
        ax.barh(y_pos, self.shap_mean_values[target_name][feature_inds],
                0.7, align='center', color="#0e73fa", alpha=0.8)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        ax.set_yticks(y_pos, [feature_names[i] for i in feature_inds])
        ax.set_xlabel("Avg. SHAP value")
        ax.set_title(
            target_name + r' $\leftarrow$ ' +
            (','.join(selected_features) if selected_features else 'ø'))

        # Recompute mean_shap_percentile here, in case it was changed
        self._compute_scaled_shap_threshold()

        xlims = ax.get_xlim()
        if xlims[1] < self.mean_shap_threshold:
            right_limit = float(self.mean_shap_threshold) + float(
                (xlims[1] - xlims[0]) / 20)
            ax.set_xlim(right=right_limit)

        ax.axvline(x=float(self.mean_shap_threshold), color='red', linestyle='--',
                   linewidth=0.5)

        fig = ax.figure if fig is None else fig
        return fig

    def _plot_discrepancies(
            self,
            target_name: str,
            threshold: float = 10.0,
            **kwargs: Any) -> None:
        """
        Plot the discrepancies between the target variable and each feature.

        Parameters
        ----------
        - target_name (str)
            The name of the target variable.
        - threshold (float)
            The threshold to use for selecting what features to plot. Only
            features with a discrepancy index below this threshold will be plotted.

        Returns:
            None

        Args:
            target_name: Name of the target variable.
            threshold: Threshold for selecting features to plot.
            **kwargs: Additional plotting arguments.

        Returns:
            None.
        """
        pass


def custom_main(
        exp_name: str,
        path: str = "/Users/renero/phd/data/RC4/",
        output_path: str = "/Users/renero/phd/output/RC4/",
        scale: bool = False) -> None:
    """
    Runs a custom main function for the given experiment name.

    Args:
        exp_name: The name of the experiment to run.
        path: The path to the data files.
        output_path: The path to the output files.
        scale: Whether to scale data before running.

    Returns:
        None.
    """

    ref_graph = utils.graph_from_dot_file(f"{path}{exp_name}.dot")
    data = pd.read_csv(f"{path}{exp_name}.csv")
    if scale:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    # Split the dataframe into train and test
    train = data.sample(frac=0.9, random_state=42)
    test = data.drop(train.index)

    rex = utils.load_experiment(f"{exp_name}_nn", output_path)
    rex.is_fitted_ = True
    print(f"Loaded experiment {exp_name}")

    rex.shaps = ShapEstimator(
        explainer="gradient",
        models=rex.models,
        correlation_th=0.5,
        mean_shap_percentile=0.8,
        iters=20,
        reciprocity=True,
        min_impact=1e-06,
        on_gpu=False,
        verbose=False,
        prog_bar=True,
        silent=False)
    rex.shaps.fit(train)
    rex.shaps.predict(test, rex.root_causes)


def sachs_main() -> None:
    """
    Run a demo experiment on the Sachs dataset.

    Args:
        None.

    Returns:
        None.
    """
    experiment_name = "sachs_long"
    path = "/Users/renero/phd/data/RC3/"
    output_path = "/Users/renero/phd/output/RC3/"

    ref_graph = utils.graph_from_dot_file(f"{path}{experiment_name}.dot")
    data = pd.read_csv(f"{path}{experiment_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Split the dataframe into train and test
    train = data.sample(frac=0.9, random_state=42)
    test = data.drop(train.index)

    rex = utils.load_experiment(f"{experiment_name}_gbt", output_path)
    rex.is_fitted_ = True
    rex.shaps.is_fitted_ = True
    print(f"Loaded experiment {experiment_name}")

    rex.shaps.prog_bar = False
    rex.shaps.verbose = True
    rex.shaps.iters = 100
    rex.shaps.predict(test, rex.root_causes)
    print("fininshed")


if __name__ == "__main__":
    custom_main('toy_dataset')
