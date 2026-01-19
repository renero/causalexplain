"""NiceGUI app for local causalexplain workflows."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from causalexplain.gui.cytoscape import ensure_cytoscape_assets
from causalexplain.gui.settings import (
    default_generate_settings,
    default_load_settings,
    default_train_settings,
    merge_settings,
)
from causalexplain.gui.styles import register_app_styles
from causalexplain.gui.tabs import GenerateTab, LoadTab, TrainTab


def run_gui(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the NiceGUI application."""
    try:
        from nicegui import app, run, ui
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "NiceGUI is required for the GUI. Install it with: pip install nicegui"
        ) from exc

    upload_dir = os.path.join(os.getcwd(), ".gui_uploads")
    os.makedirs(upload_dir, exist_ok=True)

    register_app_styles()
    ensure_cytoscape_assets()

    @ui.page("/")
    def main_page() -> None:
        """Render the main GUI page."""
        storage = app.storage.user

        train_settings = merge_settings(
            storage.get("train_settings"), default_train_settings()
        )
        load_settings = merge_settings(
            storage.get("load_settings"), default_load_settings()
        )
        generate_settings = merge_settings(
            storage.get("generate_settings"), default_generate_settings()
        )
        notify_anchor: Optional[Any] = None

        def notify_user(message: str, level: Optional[str] = None) -> None:
            """Show a toast notification anchored in the content area."""
            if notify_anchor is None:
                return
            with notify_anchor:
                ui.notify(message, type=level)

        with ui.element("div").classes("app-root"):
            with ui.element("div").classes("sidebar material"):
                ui.label("CausalExplain").classes("sidebar-header")
                ui.label("Choose a task").classes("subtle")

                panel_list = ui.column().classes("sidebar-list")
                panel_rows: Dict[str, Any] = {}

                def select_panel(panel_id: str) -> None:
                    """Switch the active panel and update the sidebar."""
                    for key, row in panel_rows.items():
                        if key == panel_id:
                            row.classes(add="selected")
                        else:
                            row.classes(remove="selected")
                    if panel_id == "train":
                        tabs.value = tab_train
                    elif panel_id == "load":
                        tabs.value = tab_load
                    else:
                        tabs.value = tab_generate
                    tabs.update()

                def add_panel_row(
                    panel_id: str, icon: str, title: str, subtitle: str
                ) -> None:
                    """Create a sidebar row that switches to a tab."""
                    row = ui.element("div").classes("sidebar-item")
                    with row:
                        ui.label(icon).classes("sidebar-icon")
                        with ui.element("div"):
                            ui.label(title).classes("sidebar-title")
                            ui.label(subtitle).classes("sidebar-subtitle")
                    row.on("click", lambda _: select_panel(panel_id))
                    panel_rows[panel_id] = row

                with panel_list:
                    add_panel_row(
                        "train",
                        "T",
                        "Train Model",
                        "Fit a new causal graph",
                    )
                    add_panel_row(
                        "load",
                        "L",
                        "Load Model",
                        "Evaluate existing runs",
                    )
                    add_panel_row(
                        "generate",
                        "G",
                        "Generate Dataset",
                        "Create synthetic data",
                    )

            with ui.element("div").classes("main-panel"):
                with ui.element("div").classes("content-scroll"):
                    content_container = ui.element("div").classes("content")
                    notify_anchor = content_container
                    with content_container:
                        tabs = ui.tabs().classes("text-sm")
                        tab_train = ui.tab("Train Model")
                        tab_load = ui.tab("Load + Evaluate")
                        tab_generate = ui.tab("Generate Dataset")

                        with ui.tab_panels(tabs, value=tab_train).classes(
                            "w-full"
                        ):
                            with ui.tab_panel(tab_train):
                                TrainTab(
                                    ui,
                                    run,
                                    storage,
                                    train_settings,
                                    upload_dir,
                                ).build()
                            with ui.tab_panel(tab_load):
                                LoadTab(
                                    ui,
                                    run,
                                    storage,
                                    load_settings,
                                    upload_dir,
                                ).build()
                            with ui.tab_panel(tab_generate):
                                GenerateTab(
                                    ui,
                                    run,
                                    storage,
                                    generate_settings,
                                    notify_user,
                                ).build()

                select_panel("train")

    ui.run(
        host=host,
        port=port,
        title="causalexplain",
        reload=False,
        storage_secret="causalexplain-local-gui",
    )
