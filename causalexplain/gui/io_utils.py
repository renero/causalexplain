"""File and path utilities for the GUI."""

from __future__ import annotations

import os
from typing import Iterable, Optional, Union


def ensure_file(path: str, suffixes: Optional[Union[Iterable[str], str]]) -> str:
    """Ensure the file exists and matches the expected suffixes."""
    if not path:
        raise ValueError("File path is required.")
    if suffixes:
        if isinstance(suffixes, str):
            suffix_list = (suffixes,)
        else:
            suffix_list = tuple(suffixes)
        if not any(path.lower().endswith(suffix) for suffix in suffix_list):
            raise ValueError(
                "Expected file extension: " + ", ".join(suffix_list)
            )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    return path


def ensure_output_dir(path: str) -> None:
    """Create the output directory for the given file path when needed."""
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def normalize_output_value(raw_value: str, required_ext: str) -> Optional[str]:
    """Normalize an output path by enforcing the required extension."""
    raw = raw_value.strip()
    if not raw:
        return None
    _, ext = os.path.splitext(raw)
    if not ext:
        return f"{raw}{required_ext}"
    if ext.lower() != required_ext:
        return ""
    return raw


def sanitize_output_name(name: str) -> str:
    """Strip extensions and whitespace from a dataset base name."""
    value = name.strip()
    for ext in (".csv", ".dot"):
        if value.lower().endswith(ext):
            value = value[: -len(ext)].strip()
    return value
