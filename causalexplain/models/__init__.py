"""Model wrappers used by causal discovery estimators."""

from .gbt import GBTRegressor
from .dnn import NNRegressor

__all__ = [
    'GBTRegressor',
    'NNRegressor',
]
