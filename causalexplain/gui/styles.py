"""Style helpers for the causalexplain GUI."""

from __future__ import annotations

import os


def app_styles_path() -> str:
    """Return the absolute path to the GUI CSS file."""
    return os.path.join(os.path.dirname(__file__), "static", "app.css")


def read_app_styles() -> str:
    """Read the GUI CSS content from disk."""
    path = app_styles_path()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def register_app_styles() -> None:
    """Inject the GUI CSS into the NiceGUI head."""
    from nicegui import ui

    css = read_app_styles()
    ui.add_head_html(f"<style>{css}</style>", shared=True)
