"""Centralized logging configuration for causalexplain."""
import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_console: Optional[Console] = None


def setup(
    verbose: bool = False,
    silent: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Configure the root causalexplain logger.

    Call once at program entry (CLI or notebook). Subsequent
    ``logging.getLogger(__name__)`` calls in any module pick up the level
    automatically.

    Args:
        verbose: Show DEBUG-level messages (maps to --verbose flag).
        silent:  Suppress everything below CRITICAL (maps to --quiet flag).
        log_file: Optional path to also write logs to a file.
    """
    global _console

    if silent:
        level = logging.CRITICAL
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    _console = Console(stderr=True)

    root = logging.getLogger("causalexplain")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    if not silent:
        handler = RichHandler(
            console=_console,
            show_time=False,
            show_level=verbose,   # show DEBUG/INFO badges only in verbose mode
            show_path=verbose,    # show file:line only in verbose mode
            markup=True,
            rich_tracebacks=True,
        )
        handler.setLevel(level)
        root.addHandler(handler)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s  %(message)s"))
        root.addHandler(fh)

    # Suppress noisy third-party loggers unconditionally.
    for name in ("pytorch_lightning", "pytorch-lightning",
                 "optuna", "shap", "matplotlib", "torch"):
        logging.getLogger(name).setLevel(logging.WARNING)
        logging.getLogger(name).propagate = False


def get_console() -> Console:
    """Return the shared Rich console (creates one if setup() not yet called)."""
    global _console
    if _console is None:
        _console = Console(stderr=True)
    return _console
