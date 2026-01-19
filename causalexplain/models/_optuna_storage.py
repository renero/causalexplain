"""Shared helpers for handling Optuna storage paths."""

from __future__ import annotations

import os
import tempfile


def _is_readonly_storage_error(exc: Exception) -> bool:
    """
    Check if the exception is due to a read-only Optuna storage.

    Args:
        exc (Exception): The exception to check.
    Returns:
        bool: True if the exception is due to a read-only storage, False otherwise.
    """
    message = str(exc).lower()
    if "readonly" in message and "database" in message:
        return True
    cause = getattr(exc, "__cause__", None)
    while cause is not None:
        message = str(cause).lower()
        if "readonly" in message and "database" in message:
            return True
        cause = getattr(cause, "__cause__", None)
    return False


def _fallback_optuna_storage(storage: str | None, study_name: str | None) -> str | None:
    """
    Fallback to a temporary Optuna storage in case the original storage is
    read-only.

    Args:
        storage (str | None): The original storage string.
        study_name (str | None): The name of the study.
    Returns:
        str | None: The fallback storage string.
    """
    if not storage or not isinstance(storage, str):
        return storage
    if not storage.startswith("sqlite:///"):
        return storage
    db_path = storage.replace("sqlite:///", "", 1)
    if not db_path or db_path == ":memory:":
        return storage
    base = study_name or os.path.splitext(os.path.basename(db_path))[0] or "optuna"
    filename = f"{base}_tuning.db"
    return f"sqlite:///{os.path.join(tempfile.gettempdir(), filename)}"


def _ensure_writable_optuna_storage(
        storage: str | None, study_name: str | None) -> str | None:
    """
    Ensure that the Optuna storage is writable. If not, fallback to a temporary
    storage.

    Args:
        storage (str | None): The original storage string.
        study_name (str | None): The name of the study.
    Returns:
        str | None: The writable storage string.
    """
    if not storage or not isinstance(storage, str):
        return storage
    if not storage.startswith("sqlite:///"):
        return storage
    db_path = storage.replace("sqlite:///", "", 1)
    if not db_path or db_path == ":memory:":
        return storage
    if not os.path.isabs(db_path):
        db_path = os.path.abspath(db_path)
    db_dir = os.path.dirname(db_path) or os.getcwd()
    try:
        os.makedirs(db_dir, exist_ok=True)
    except OSError:
        return _fallback_optuna_storage(storage, study_name)
    if os.path.exists(db_path) and not os.access(db_path, os.W_OK):
        return _fallback_optuna_storage(storage, study_name)
    if not os.access(db_dir, os.W_OK):
        return _fallback_optuna_storage(storage, study_name)
    return f"sqlite:///{db_path}"
