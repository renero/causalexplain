"""GUI entry points for causalexplain."""

from __future__ import annotations

from typing import Any


def run_gui(*args: Any, **kwargs: Any) -> Any:
    """Lazy import to avoid circular dependencies when loading GUI helpers."""
    from .app import run_gui as _run_gui

    return _run_gui(*args, **kwargs)


__all__ = ["run_gui"]
