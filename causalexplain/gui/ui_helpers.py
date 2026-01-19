"""UI helper utilities for NiceGUI widgets."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from causalexplain.gui.io_utils import normalize_output_value


def update_settings(
    storage: Any, settings_key: str, target: Dict[str, Any], key: str, value: Any
) -> None:
    """Update a settings dict and persist it to NiceGUI storage."""
    target[key] = value
    storage[settings_key] = target


def bind_setting(
    element: Any,
    storage: Any,
    settings_key: str,
    settings_ref: Dict[str, Any],
    field: str,
) -> None:
    """Bind UI change events to a settings dictionary."""
    if hasattr(element, "on_value_change"):
        element.on_value_change(
            lambda e, key=field: update_settings(
                storage, settings_key, settings_ref, key, e.value
            )
        )
    else:
        element.on(
            "change",
            lambda e, key=field, el=element: update_settings(
                storage,
                settings_key,
                settings_ref,
                key,
                getattr(e, "value", getattr(el, "value", None)),
            ),
        )


def set_input_value(input_el: Any, value: str) -> None:
    """Update a NiceGUI input value and refresh the widget."""
    input_el.value = value
    input_el.update()


async def save_upload(
    file_upload: Any, upload_dir: str, suffix: Optional[str] = None
) -> str:
    """Persist an uploaded file to the GUI upload folder."""
    import os
    import time

    filename = getattr(file_upload, "name", "upload")
    if suffix and not filename.lower().endswith(suffix):
        filename = f"{filename}{suffix}"
    timestamp = int(time.time() * 1000)
    base = os.path.basename(filename)
    path = os.path.join(upload_dir, f"{timestamp}_{base}")
    await file_upload.save(path)
    return path


def make_upload_handler(
    input_el: Any,
    storage: Any,
    settings_key: str,
    settings_ref: Dict[str, Any],
    field: str,
    upload_dir: str,
    suffix: Optional[str] = None,
    status_label: Optional[Any] = None,
) -> Callable[[Any], Any]:
    """Create an async upload handler that updates UI state."""

    async def _handler(event: Any) -> None:
        """Persist an uploaded file and update UI state."""
        file_upload = getattr(event, "file", None)
        if file_upload is None:
            raise ValueError("Upload event missing file payload.")
        path = await save_upload(file_upload, upload_dir, suffix)
        set_input_value(input_el, path)
        update_settings(storage, settings_key, settings_ref, field, path)
        if status_label is not None:
            import os

            status_label.text = f"Loaded: {os.path.basename(path)}"
            status_label.update()

    return _handler


def normalize_output_path(
    input_el: Any,
    storage: Any,
    settings_key: str,
    settings_ref: Dict[str, Any],
    field: str,
    required_ext: str,
) -> Optional[str]:
    """Normalize a UI output path and persist the setting."""
    raw = (getattr(input_el, "value", "") or "").strip()
    normalized = normalize_output_value(raw, required_ext)
    if normalized is None or normalized == raw:
        return normalized
    if normalized:
        set_input_value(input_el, normalized)
        update_settings(storage, settings_key, settings_ref, field, normalized)
    return normalized
