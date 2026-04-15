"""
CausalExplain: A Python package for causal discovery and inference.

This package provides tools for discovering and analyzing causal relationships
in data using various methods and algorithms.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._version import __version__

__all__ = [
    "common",
    "estimators",
    "explainability",
    "generators",
    "independence",
    "metrics",
    "models",
    "GraphDiscovery",
    "__version__",
]

if TYPE_CHECKING:
    from . import common, estimators, explainability, generators, independence, metrics, models
    from .causalexplainer import GraphDiscovery


def __getattr__(name: str) -> Any:
    """Load heavy subpackages lazily so basic package import stays lightweight."""
    if name in {
        "common",
        "estimators",
        "explainability",
        "generators",
        "independence",
        "metrics",
        "models",
    }:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name == "GraphDiscovery":
        graph_discovery = import_module(".causalexplainer", __name__).GraphDiscovery
        globals()[name] = graph_discovery
        return graph_discovery
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazily available exports to interactive tooling."""
    return sorted(set(globals()) | set(__all__))
