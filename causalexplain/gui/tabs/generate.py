"""Dataset generation tab for the causalexplain GUI."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional

import numpy as np

from causalexplain.common import DEFAULT_SEED
from causalexplain.common.plot import dag2dot
from causalexplain.generators.generators import AcyclicGraphGenerator
from causalexplain.gui.graph_utils import dag_is_valid
from causalexplain.gui.io_utils import sanitize_output_name
from causalexplain.gui.rendering import render_cytoscape_graph
from causalexplain.gui.ui_helpers import bind_setting


class GenerateTab:
    """Build and manage the Generate Dataset tab."""

    def __init__(
        self,
        ui: Any,
        run: Any,
        storage: Any,
        settings: Dict[str, Any],
        notify: Any,
    ) -> None:
        """Initialize the generate tab with shared GUI dependencies."""
        self.ui = ui
        self.run = run
        self.storage = storage
        self.settings = settings
        self.notify = notify
        self.generate_state: Dict[str, Any] = {
            "graph": None,
            "data": None,
            "adjacency": None,
        }
        self.generate_dag_container: Optional[Any] = None
        self.save_button: Optional[Any] = None
        self.output_dir_input: Optional[Any] = None
        self.output_name_input: Optional[Any] = None

    def build(self) -> None:
        """Render the Generate Dataset tab UI."""
        with self.ui.element("div").classes("section-card w-full"):
            self.ui.label("Generate Dataset").classes("section-title")
            with self.ui.element("div").classes("generate-grid w-full"):
                with self.ui.element("div").classes("nested-panel"):
                    self.ui.label("Generation controls").classes("subtle")
                    timeout_input = self.ui.number(
                        "t (seconds)",
                        value=self.settings.get("timeout_s", 30),
                    ).props("dense").classes("w-full")
                    retries_input = self.ui.number(
                        "R (max retries)",
                        value=self.settings.get("max_retries", 50),
                    ).props("dense").classes("w-full")
                    min_edges_input = self.ui.number(
                        "Min edges",
                        value=self.settings.get("min_edges", 0),
                    ).props("dense").classes("w-full")
                    max_edges_input = self.ui.number(
                        "Max edges",
                        value=self.settings.get("max_edges", 30),
                    ).props("dense").classes("w-full")

                with self.ui.element("div").classes("nested-panel"):
                    self.ui.label("Dataset parameters").classes("subtle")
                    mechanism_select = self.ui.select(
                        [
                            "linear",
                            "polynomial",
                            "sigmoid_add",
                            "sigmoid_mix",
                            "gp_add",
                            "gp_mix",
                        ],
                        value=self.settings.get("mechanism", "linear"),
                        label="Mechanism",
                    ).classes("w-full")
                    nodes_input = self.ui.number(
                        "Variables",
                        value=self.settings.get("nodes", 10),
                    ).props("dense").classes("w-full")
                    samples_input = self.ui.number(
                        "Samples",
                        value=self.settings.get("samples", 500),
                    ).props("dense").classes("w-full")
                    parents_input = self.ui.number(
                        "Max parents",
                        value=self.settings.get("max_parents", 3),
                    ).props("dense").classes("w-full")
                    gen_seed_input = self.ui.number(
                        "Seed",
                        value=self.settings.get("seed", DEFAULT_SEED),
                    ).props("dense").classes("w-full")
                    rescale_switch = self.ui.switch(
                        "Rescale",
                        value=self.settings.get("rescale", True),
                    )

            with self.ui.element("div").classes("generate-actions w-full"):
                self.ui.element("div").classes("generate-spacer")
                self.ui.button(
                    "Generate",
                    on_click=lambda: asyncio.create_task(self.run_generate()),
                ).classes("w-full")

            self.generate_dag_container = self.ui.element("div").classes(
                "dag-frame w-full"
            )
            render_cytoscape_graph(self.generate_dag_container, None)

            with self.ui.element("div").classes("save-row w-full"):
                self.output_dir_input = self.ui.input(
                    "Output directory",
                    value=self.settings.get("output_dir", ""),
                ).props("dense").classes("w-full")
                self.output_name_input = self.ui.input(
                    "Dataset name",
                    value=self.settings.get("output_name", "generated_dataset"),
                ).props("dense").classes("w-full")
                self.save_button = self.ui.button(
                    "SAVE",
                    on_click=lambda: asyncio.create_task(
                        self.run_save_generated()
                    ),
                ).classes("w-full")
                self.save_button.disable()

            bind_setting(
                mechanism_select,
                self.storage,
                "generate_settings",
                self.settings,
                "mechanism",
            )
            bind_setting(
                timeout_input,
                self.storage,
                "generate_settings",
                self.settings,
                "timeout_s",
            )
            bind_setting(
                retries_input,
                self.storage,
                "generate_settings",
                self.settings,
                "max_retries",
            )
            bind_setting(
                min_edges_input,
                self.storage,
                "generate_settings",
                self.settings,
                "min_edges",
            )
            bind_setting(
                max_edges_input,
                self.storage,
                "generate_settings",
                self.settings,
                "max_edges",
            )
            bind_setting(
                nodes_input,
                self.storage,
                "generate_settings",
                self.settings,
                "nodes",
            )
            bind_setting(
                samples_input,
                self.storage,
                "generate_settings",
                self.settings,
                "samples",
            )
            bind_setting(
                parents_input,
                self.storage,
                "generate_settings",
                self.settings,
                "max_parents",
            )
            bind_setting(
                gen_seed_input,
                self.storage,
                "generate_settings",
                self.settings,
                "seed",
            )
            bind_setting(
                rescale_switch,
                self.storage,
                "generate_settings",
                self.settings,
                "rescale",
            )
            bind_setting(
                self.output_dir_input,
                self.storage,
                "generate_settings",
                self.settings,
                "output_dir",
            )
            bind_setting(
                self.output_name_input,
                self.storage,
                "generate_settings",
                self.settings,
                "output_name",
            )

    def _generate_job(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a synthetic dataset in a worker thread."""
        np.random.seed(int(settings.get("seed", DEFAULT_SEED)))
        timeout_s = max(0.0, float(settings.get("timeout_s", 30)))
        max_retries = int(settings.get("max_retries", 50))
        min_edges = int(settings.get("min_edges", 0))
        max_edges = int(settings.get("max_edges", 30))
        if min_edges > max_edges:
            raise ValueError("min_edges must be less than or equal to max_edges.")
        if max_retries < 0:
            max_retries = 0
        max_attempts = max_retries + 1
        start_time = time.monotonic()
        attempt = 0
        while attempt < max_attempts:
            if (time.monotonic() - start_time) >= timeout_s:
                break
            attempt += 1
            generator = AcyclicGraphGenerator(
                settings["mechanism"],
                points=int(settings["samples"]),
                nodes=int(settings["nodes"]),
                parents_max=int(settings["max_parents"]),
                verbose=False,
            )
            graph, data = generator.generate(rescale=bool(settings["rescale"]))
            if not dag_is_valid(graph, min_edges, max_edges):
                continue
            return {
                "graph": graph,
                "data": data,
                "adjacency": generator.adjacency_matrix,
            }
        elapsed = time.monotonic() - start_time
        if elapsed >= timeout_s:
            raise TimeoutError("Timeout reached before a valid DAG was found.")
        raise ValueError("No valid DAG found within the retry limit.")

    async def run_generate(self) -> None:
        """Generate a dataset and update UI widgets."""
        try:
            result = await self.run.io_bound(self._generate_job, self.settings)
        except Exception as exc:
            self.notify(f"Error: {str(exc)}", "negative")
            if self.save_button is not None and self.generate_state.get("graph") is None:
                self.save_button.disable()
            return

        self.generate_state.update(
            {
                "graph": result.get("graph"),
                "data": result.get("data"),
                "adjacency": result.get("adjacency"),
            }
        )
        render_cytoscape_graph(
            self.generate_dag_container, self.generate_state["graph"]
        )
        if self.save_button is not None:
            self.save_button.enable()
        self.notify("Generation completed.", "positive")

    def _save_job(
        self, payload: Dict[str, Any], output_dir: str, output_name: str
    ) -> None:
        """Persist the generated dataset to disk."""
        graph = payload.get("graph")
        data = payload.get("data")
        adjacency = payload.get("adjacency")
        if graph is None or data is None or adjacency is None:
            raise ValueError("Generate a dataset first.")
        output_dir = output_dir.strip()
        output_name = sanitize_output_name(output_name)
        if not output_dir:
            raise ValueError("Output directory is required.")
        if not output_name:
            raise ValueError("Dataset name is required.")
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, output_name)
        data.to_csv(f"{output_base}.csv", index=False)
        dot_obj = dag2dot(graph)
        if dot_obj is None:
            raise ValueError("Unable to save a DAG with no edges.")
        graph_dot_format = dot_obj.to_string()
        graph_dot_format = f"strict {graph_dot_format[:-9]}\n}}"
        with open(f"{output_base}.dot", "w") as handle:
            handle.write(graph_dot_format)

    async def run_save_generated(self) -> None:
        """Save the generated dataset to disk."""
        output_dir = self.settings.get("output_dir", "")
        output_name = self.settings.get("output_name", "")
        try:
            await self.run.io_bound(
                self._save_job,
                self.generate_state,
                output_dir,
                output_name,
            )
        except Exception as exc:
            self.notify(f"Error: {str(exc)}", "negative")
            return
        self.notify("Dataset saved.", "positive")
