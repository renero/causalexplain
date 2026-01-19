"""Training tab for the causalexplain GUI."""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import time
from typing import Any, Dict, Optional

from causalexplain.causalexplainer import GraphDiscovery
from causalexplain.common import (
    DEFAULT_MAX_SAMPLES,
    DEFAULT_REGRESSORS,
    SUPPORTED_METHODS,
    utils,
)
from causalexplain.gui.graph_utils import graph_from_dot
from causalexplain.gui.io_utils import ensure_file, ensure_output_dir
from causalexplain.gui.rendering import render_cytoscape_overlay
from causalexplain.gui.ui_helpers import (
    bind_setting,
    make_upload_handler,
    normalize_output_path,
)


class TrainTab:
    """Build and manage the Train Model tab."""

    def __init__(
        self,
        ui: Any,
        run: Any,
        storage: Any,
        settings: Dict[str, Any],
        upload_dir: str,
    ) -> None:
        """Initialize the train tab with shared GUI dependencies."""
        self.ui = ui
        self.run = run
        self.storage = storage
        self.settings = settings
        self.upload_dir = upload_dir
        self.state: Dict[str, Any] = {
            "task": None,
            "running": False,
            "discoverer": None,
            "dag": None,
        }
        self.method_select: Optional[Any] = None
        self.rex_section: Optional[Any] = None
        self.train_log: Optional[Any] = None
        self.train_progress: Optional[Any] = None
        self.run_button: Optional[Any] = None
        self.cancel_button: Optional[Any] = None
        self.train_overlay_container: Optional[Any] = None
        self.train_overlay_status: Optional[Any] = None
        self.model_output_input: Optional[Any] = None
        self.dag_output_input: Optional[Any] = None

    def build(self) -> None:
        """Render the Train Model tab UI."""
        with self.ui.element("div").classes("train-grid"):
            self._build_inputs_section()
            self._build_method_section()
            self._build_run_section()
            self._build_overlay_section()
            self._build_output_section()

        self._update_rex_visibility()
        if self.method_select is not None:
            self.method_select.on("change", lambda _: self._update_rex_visibility())

    def _build_inputs_section(self) -> None:
        """Build the dataset/prior input section."""
        with self.ui.element("div").classes("section-card span-full"):
            self.ui.label("Inputs + Prior").classes("section-title")

            with self.ui.element("div").classes("file-row"):
                self.ui.label("Dataset CSV").classes("file-label")
                dataset_input = self.ui.input(
                    value=self.settings.get("dataset_path", "")
                ).props("dense").classes("file-input w-full")
                self.ui.upload(
                    label="Browse",
                    on_upload=make_upload_handler(
                        dataset_input,
                        self.storage,
                        "train_settings",
                        self.settings,
                        "dataset_path",
                        self.upload_dir,
                        ".csv",
                    ),
                    auto_upload=True,
                ).props("accept=.csv").classes("upload-inline")

            with self.ui.element("div").classes("file-row"):
                self.ui.label("True DAG").classes("file-label")
                dag_input = self.ui.input(
                    value=self.settings.get("true_dag_path", "")
                ).props("dense").classes("file-input w-full")
                self.ui.upload(
                    label="Browse",
                    on_upload=make_upload_handler(
                        dag_input,
                        self.storage,
                        "train_settings",
                        self.settings,
                        "true_dag_path",
                        self.upload_dir,
                        ".dot",
                    ),
                    auto_upload=True,
                ).props("accept=.dot").classes("upload-inline")

            with self.ui.element("div").classes("file-row"):
                self.ui.label("Prior JSON").classes("file-label")
                prior_input = self.ui.input(
                    value=self.settings.get("prior_path", "")
                ).props("dense").classes("file-input w-full")
                self.ui.upload(
                    label="Browse",
                    on_upload=make_upload_handler(
                        prior_input,
                        self.storage,
                        "train_settings",
                        self.settings,
                        "prior_path",
                        self.upload_dir,
                        ".json",
                    ),
                    auto_upload=True,
                ).props("accept=.json").classes("upload-inline")

            bind_setting(
                dataset_input,
                self.storage,
                "train_settings",
                self.settings,
                "dataset_path",
            )
            bind_setting(
                dag_input,
                self.storage,
                "train_settings",
                self.settings,
                "true_dag_path",
            )
            bind_setting(
                prior_input,
                self.storage,
                "train_settings",
                self.settings,
                "prior_path",
            )

    def _build_method_section(self) -> None:
        """Build the method selection and options section."""
        with self.ui.element("div").classes("section-card"):
            self.ui.label("Method + Core Settings").classes("section-title")
            with self.ui.element("div").classes("field-row"):
                self.method_select = self.ui.select(
                    SUPPORTED_METHODS,
                    value=self.settings.get("method", "rex"),
                    label="Method",
                )
                combine_select = self.ui.select(
                    ["union", "intersection"],
                    value=self.settings.get("combine_op", "union"),
                    label="Combine DAGs",
                )
            with self.ui.element("div").classes("pair-row w-full"):
                seed_input = self.ui.number(
                    "Seed",
                    value=self.settings.get("seed"),
                ).props("dense").classes("w-full")
                tolerance_input = self.ui.number(
                    "Bootstrap tolerance",
                    value=self.settings.get("bootstrap_tolerance"),
                ).props("dense").classes("w-full")
            with self.ui.element("div").classes("pair-row w-full"):
                hpo_input = self.ui.number(
                    "HPO iterations",
                    value=self.settings.get("hpo_iterations"),
                ).props("dense").classes("w-full")
                bootstrap_input = self.ui.number(
                    "Bootstrap iterations",
                    value=self.settings.get("bootstrap_iterations"),
                ).props("dense").classes("w-full")

            bind_setting(
                self.method_select,
                self.storage,
                "train_settings",
                self.settings,
                "method",
            )
            bind_setting(
                seed_input,
                self.storage,
                "train_settings",
                self.settings,
                "seed",
            )
            bind_setting(
                hpo_input,
                self.storage,
                "train_settings",
                self.settings,
                "hpo_iterations",
            )
            bind_setting(
                bootstrap_input,
                self.storage,
                "train_settings",
                self.settings,
                "bootstrap_iterations",
            )
            bind_setting(
                tolerance_input,
                self.storage,
                "train_settings",
                self.settings,
                "bootstrap_tolerance",
            )
            bind_setting(
                combine_select,
                self.storage,
                "train_settings",
                self.settings,
                "combine_op",
            )

            self.rex_section = self.ui.expansion(
                "ReX Options", value=False
            ).classes("w-full")
            with self.rex_section:
                with self.ui.element("div").classes("field-row"):
                    device_select = self.ui.select(
                        ["cpu", "cuda", "mps"],
                        value=self.settings.get("device", "cpu"),
                        label="Device",
                    )
                    parallel_jobs_input = self.ui.number(
                        "Parallel jobs",
                        value=self.settings.get("parallel_jobs", 0),
                    ).props("dense")
                    bootstrap_jobs_input = self.ui.number(
                        "Bootstrap parallel jobs",
                        value=self.settings.get("bootstrap_parallel_jobs", 0),
                    ).props("dense")
                    adaptive_switch = self.ui.switch(
                        "Adaptive SHAP sampling",
                        value=self.settings.get("adaptive_shap_sampling", True),
                    )
                    max_shap_input = self.ui.number(
                        "Max SHAP samples",
                        value=self.settings.get("max_shap_samples", DEFAULT_MAX_SAMPLES),
                    ).props("dense")
                    regressors_input = self.ui.select(
                        ["nn", "gbt"],
                        value=self.settings.get("regressors", DEFAULT_REGRESSORS),
                        label="Regressors",
                        multiple=True,
                    )

                bind_setting(
                    device_select,
                    self.storage,
                    "train_settings",
                    self.settings,
                    "device",
                )
                bind_setting(
                    parallel_jobs_input,
                    self.storage,
                    "train_settings",
                    self.settings,
                    "parallel_jobs",
                )
                bind_setting(
                    bootstrap_jobs_input,
                    self.storage,
                    "train_settings",
                    self.settings,
                    "bootstrap_parallel_jobs",
                )
                bind_setting(
                    adaptive_switch,
                    self.storage,
                    "train_settings",
                    self.settings,
                    "adaptive_shap_sampling",
                )
                bind_setting(
                    max_shap_input,
                    self.storage,
                    "train_settings",
                    self.settings,
                    "max_shap_samples",
                )
                bind_setting(
                    regressors_input,
                    self.storage,
                    "train_settings",
                    self.settings,
                    "regressors",
                )

                with self.ui.expansion("Advanced ReX settings", value=False):
                    with self.ui.element("div").classes("field-row"):
                        explainer_input = self.ui.select(
                            ["gradient", "explainer", "kernel", "tree"],
                            value=self.settings.get("explainer", "gradient"),
                            label="SHAP explainer backend",
                        )
                        corr_method_input = self.ui.select(
                            ["spearman", "pearson", "kendall", "mic"],
                            value=self.settings.get("corr_method", "spearman"),
                            label="Correlation method",
                        )
                        corr_alpha_input = self.ui.number(
                            "Correlation alpha",
                            value=self.settings.get("corr_alpha", 0.6),
                        )
                        corr_clusters_input = self.ui.number(
                            "Correlation clusters",
                            value=self.settings.get("corr_clusters", 15),
                        )
                        condlen_input = self.ui.number(
                            "Condlen",
                            value=self.settings.get("condlen", 1),
                        )
                        condsize_input = self.ui.number(
                            "Condsize",
                            value=self.settings.get("condsize", 0),
                        )
                        mean_pi_input = self.ui.number(
                            "Mean PI percentile",
                            value=self.settings.get("mean_pi_percentile", 0.8),
                        )
                        discrepancy_input = self.ui.number(
                            "Discrepancy threshold",
                            value=self.settings.get("discrepancy_threshold", 0.99),
                        )
                        sampling_input = self.ui.input(
                            "Bootstrap sampling split",
                            value=self.settings.get(
                                "bootstrap_sampling_split", "auto"
                            ),
                        ).props("dense")

                    bind_setting(
                        explainer_input,
                        self.storage,
                        "train_settings",
                        self.settings,
                        "explainer",
                    )
                    bind_setting(
                        corr_method_input,
                        self.storage,
                        "train_settings",
                        self.settings,
                        "corr_method",
                    )
                    bind_setting(
                        corr_alpha_input,
                        self.storage,
                        "train_settings",
                        self.settings,
                        "corr_alpha",
                    )
                    bind_setting(
                        corr_clusters_input,
                        self.storage,
                        "train_settings",
                        self.settings,
                        "corr_clusters",
                    )
                    bind_setting(
                        condlen_input,
                        self.storage,
                        "train_settings",
                        self.settings,
                        "condlen",
                    )
                    bind_setting(
                        condsize_input,
                        self.storage,
                        "train_settings",
                        self.settings,
                        "condsize",
                    )
                    bind_setting(
                        mean_pi_input,
                        self.storage,
                        "train_settings",
                        self.settings,
                        "mean_pi_percentile",
                    )
                    bind_setting(
                        discrepancy_input,
                        self.storage,
                        "train_settings",
                        self.settings,
                        "discrepancy_threshold",
                    )
                    bind_setting(
                        sampling_input,
                        self.storage,
                        "train_settings",
                        self.settings,
                        "bootstrap_sampling_split",
                    )

    def _build_run_section(self) -> None:
        """Build the training action section."""
        with self.ui.element("div").classes("section-card"):
            self.ui.label("Run").classes("section-title")
            with self.ui.element("div").classes("action-row"):
                self.run_button = self.ui.button(
                    "Start training", on_click=self.start_training_task
                )
                self.cancel_button = self.ui.button(
                    "Cancel", on_click=self.cancel_training_task
                ).props("flat")
                self.cancel_button.disable()
            self.train_progress = self.ui.linear_progress(value=0).props(
                "instant-feedback"
            )
            self.train_log = self.ui.log(max_lines=200).classes(
                "w-full train-log"
            )

    def _build_overlay_section(self) -> None:
        """Build the overlay panel section."""
        with self.ui.element("div").classes("section-card span-full"):
            self.ui.label("Overlay vs True DAG").classes("subtle")
            self.train_overlay_container = self.ui.element("div").classes(
                "dag-frame"
            )
            self.train_overlay_status = self.ui.label("").classes("subtle")

    def _build_output_section(self) -> None:
        """Build the output save controls."""
        with self.ui.element("div").classes("section-card span-full"):
            self.ui.label("Outputs").classes("section-title")
            with self.ui.element("div").classes("output-row"):
                self.model_output_input = self.ui.input(
                    "Model pickle path",
                    value=self.settings.get("save_model_path", ""),
                ).props("dense").classes("w-full")
                self.ui.button(
                    "Save model", on_click=self.save_trained_model
                ).classes("mini-button")
            with self.ui.element("div").classes("output-row"):
                self.dag_output_input = self.ui.input(
                    "Output DAG (.dot) path",
                    value=self.settings.get("output_dag_path", ""),
                ).props("dense").classes("w-full")
                self.ui.button(
                    "Save DAG", on_click=self.save_trained_dag
                ).classes("mini-button")

            bind_setting(
                self.model_output_input,
                self.storage,
                "train_settings",
                self.settings,
                "save_model_path",
            )
            bind_setting(
                self.dag_output_input,
                self.storage,
                "train_settings",
                self.settings,
                "output_dag_path",
            )

    def _update_rex_visibility(self) -> None:
        """Show or hide ReX-specific settings."""
        if self.rex_section is None:
            return
        is_rex = self.settings.get("method") == "rex"
        self.rex_section.visible = is_rex
        if is_rex:
            self.rex_section.value = False
        self.rex_section.update()

    def _set_run_state(self, running: bool) -> None:
        """Toggle UI controls based on training state."""
        self.state["running"] = running
        if self.run_button is not None:
            if running:
                self.run_button.disable()
            else:
                self.run_button.enable()
        if self.cancel_button is not None:
            if running:
                self.cancel_button.enable()
            else:
                self.cancel_button.disable()

    def _set_progress_state(self, indeterminate: bool) -> None:
        """Update the training progress indicator."""
        if self.train_progress is None:
            return
        if indeterminate:
            self.train_progress.props("indeterminate")
        else:
            self.train_progress.props(remove="indeterminate")
            self.train_progress.value = 0
        self.train_progress.update()

    def _clear_overlay(self) -> None:
        """Clear the overlay panel and status label."""
        if self.train_overlay_container is not None:
            self.train_overlay_container.clear()
        if self.train_overlay_status is not None:
            self.train_overlay_status.text = ""
            self.train_overlay_status.update()

    def _append_log_lines(self, text: str) -> None:
        """Append text lines to the training log."""
        if self.train_log is None or not text:
            return
        for line in text.splitlines():
            self.train_log.push(line)

    def _rex_extra_kwargs(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ReX-specific keyword arguments from settings."""
        if settings.get("method") != "rex":
            return {}
        split = settings.get("bootstrap_sampling_split", "auto")
        if isinstance(split, str) and split.lower() != "auto":
            split = float(split)
        return {
            "adaptive_shap_sampling": settings.get(
                "adaptive_shap_sampling", True
            ),
            "max_shap_samples": int(
                settings.get("max_shap_samples", DEFAULT_MAX_SAMPLES)
            ),
            "explainer": settings.get("explainer", "gradient"),
            "corr_method": settings.get("corr_method", "spearman"),
            "corr_alpha": float(settings.get("corr_alpha", 0.6)),
            "corr_clusters": int(settings.get("corr_clusters", 15)),
            "condlen": int(settings.get("condlen", 1)),
            "condsize": int(settings.get("condsize", 0)),
            "mean_pi_percentile": float(settings.get("mean_pi_percentile", 0.8)),
            "discrepancy_threshold": float(
                settings.get("discrepancy_threshold", 0.99)
            ),
            "bootstrap_sampling_split": split,
        }

    def _train_job(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Run the training job in a background worker."""
        output: Dict[str, Any] = {}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            dataset_path = ensure_file(settings["dataset_path"], ".csv")
            true_dag_path = settings.get("true_dag_path") or None
            if true_dag_path:
                true_dag_path = ensure_file(true_dag_path, ".dot")
            prior_path = settings.get("prior_path") or ""
            prior = None
            if prior_path:
                prior_path = ensure_file(prior_path, ".json")
                prior = utils.read_json_file(prior_path)
            dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
            discoverer = GraphDiscovery(
                experiment_name=dataset_name,
                model_type=settings["method"],
                csv_filename=dataset_path,
                true_dag_filename=true_dag_path,
                verbose=False,
                seed=int(settings["seed"]),
                device=settings["device"],
                parallel_jobs=int(settings["parallel_jobs"]),
                bootstrap_parallel_jobs=int(
                    settings["bootstrap_parallel_jobs"]
                ),
                max_shap_samples=int(settings["max_shap_samples"]),
            )
            if settings["method"] == "rex":
                regressors = settings.get("regressors", DEFAULT_REGRESSORS)
                if len(regressors) != 2:
                    raise ValueError("ReX requires exactly two regressors.")
                discoverer.regressors = regressors
            start_time = time.time()
            discoverer.run(
                hpo_iterations=int(settings["hpo_iterations"]),
                bootstrap_iterations=int(settings["bootstrap_iterations"]),
                prior=prior,
                bootstrap_tolerance=float(settings["bootstrap_tolerance"]),
                combine_op=settings.get("combine_op", "union"),
                **self._rex_extra_kwargs(settings),
            )
            elapsed_seconds = time.time() - start_time
            ref_graph = graph_from_dot(true_dag_path) if true_dag_path else None
            discoverer.ref_graph = ref_graph
            discoverer.model.ref_graph = ref_graph
            num_variables = 0
            if discoverer.data_columns:
                num_variables = len(discoverer.data_columns)
            output["dag"] = discoverer.dag
            output["discoverer"] = discoverer
            output["metrics"] = discoverer.metrics
            output["ref_graph"] = ref_graph
            output["elapsed_seconds"] = elapsed_seconds
            output["num_variables"] = num_variables
        output["log"] = buffer.getvalue()
        return output

    async def run_training(self) -> None:
        """Execute training and update UI elements."""
        if self.state["running"]:
            return

        self._set_run_state(True)
        if self.train_log is not None:
            self.train_log.clear()
            self.train_log.push("Starting training...")
        self._clear_overlay()
        self._set_progress_state(True)

        try:
            result = await self.run.io_bound(self._train_job, self.settings)
        except asyncio.CancelledError:
            if self.train_log is not None:
                self.train_log.push("Training canceled.")
            self._set_progress_state(False)
            return
        except Exception as exc:
            if self.train_log is not None:
                self.train_log.push(f"Error: {str(exc)}")
            self._set_progress_state(False)
            return
        finally:
            self.state["task"] = None
            self._set_run_state(False)

        self._set_progress_state(False)
        self._append_log_lines(result.get("log") or "")

        discoverer = result.get("discoverer")
        self.state["discoverer"] = discoverer
        self.state["dag"] = result.get("dag")
        metrics = result.get("metrics")
        ref_graph = result.get("ref_graph")
        overlay_error = render_cytoscape_overlay(
            self.train_overlay_container,
            self.train_overlay_status,
            discoverer,
            ref_graph,
            persist_positions=True,
        )
        if self.train_log is not None:
            num_variables = int(result.get("num_variables") or 0)
            elapsed_seconds = float(result.get("elapsed_seconds") or 0.0)
            elapsed_minutes = elapsed_seconds / 60.0
            self.train_log.push(
                "Training completed for "
                f"{num_variables} variables, in "
                f"{elapsed_minutes:.2f} minutes."
            )
            if overlay_error is not None:
                self.train_log.push(
                    f"Overlay render failed: {str(overlay_error)}"
                )
            if metrics is not None:
                self.train_log.push("\n\nEvaluation Metrics:")
                metrics_buffer = io.StringIO()
                with contextlib.redirect_stdout(metrics_buffer):
                    print(metrics)
                self._append_log_lines(metrics_buffer.getvalue())

    def start_training_task(self) -> None:
        """Launch the training coroutine."""
        task = self.state.get("task")
        if task and not task.done():
            return
        self.state["task"] = asyncio.create_task(self.run_training())

    def cancel_training_task(self) -> None:
        """Request cancellation of the running training job."""
        task = self.state.get("task")
        if task and not task.done():
            task.cancel()
            if self.train_log is not None:
                self.train_log.push("Cancel requested.")

    async def save_trained_model(self) -> None:
        """Persist the trained model to disk."""
        if self.model_output_input is None:
            return
        model_path = normalize_output_path(
            self.model_output_input,
            self.storage,
            "train_settings",
            self.settings,
            "save_model_path",
            ".pickle",
        )
        if model_path is None:
            self.ui.notify("Model pickle path is required.", type="warning")
            return
        if not model_path:
            self.ui.notify("Expected a .pickle file path.", type="negative")
            return
        discoverer = self.state.get("discoverer")
        if discoverer is None:
            self.ui.notify("Train a model before saving.", type="warning")
            return

        def _save() -> None:
            """Write the trained model to the requested path."""
            ensure_output_dir(model_path)
            discoverer.save_model(model_path)

        try:
            await self.run.io_bound(_save)
        except Exception as exc:
            self.ui.notify(f"Save failed: {str(exc)}", type="negative")
            return
        self.ui.notify(f"Saved model to {model_path}", type="positive")

    async def save_trained_dag(self) -> None:
        """Persist the trained DAG to disk."""
        if self.dag_output_input is None:
            return
        dag_path = normalize_output_path(
            self.dag_output_input,
            self.storage,
            "train_settings",
            self.settings,
            "output_dag_path",
            ".dot",
        )
        if dag_path is None:
            self.ui.notify("Output DAG path is required.", type="warning")
            return
        if not dag_path:
            self.ui.notify("Expected a .dot file path.", type="negative")
            return
        dag = self.state.get("dag")
        if dag is None:
            self.ui.notify("Train a model before saving.", type="warning")
            return

        def _save() -> None:
            """Write the trained DAG to the requested path."""
            ensure_output_dir(dag_path)
            utils.graph_to_dot_file(dag, dag_path)

        try:
            await self.run.io_bound(_save)
        except Exception as exc:
            self.ui.notify(f"Save failed: {str(exc)}", type="negative")
            return
        self.ui.notify(f"Saved DAG to {dag_path}", type="positive")
