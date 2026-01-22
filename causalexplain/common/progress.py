"""Progress helpers for a single global training bar."""
from __future__ import annotations

from typing import Optional

from mlforge.progbar import ProgBar


class ProgressManager:
    """Manage a global progress bar with fractional phase updates."""

    def __init__(self, total_units: int, name: str = "Training", enabled: bool = True):
        self.enabled = enabled and total_units > 0
        self.name = name
        self.total_units = float(total_units)
        self._main_progress = 0.0
        self._phase_weight = 0.0
        self._phase_fraction = 0.0
        self._phase_substeps = 0
        self._phase_name: Optional[str] = None
        self.pbar: Optional[ProgBar] = None

        if self.enabled:
            # Main bar uses macro-units; phases advance it fractionally.
            self.pbar = ProgBar(
                name=self.name,
                num_steps=total_units,
                verbose=False,
                propagate=False,
            )
            self.pbar.propagate = False

    def start_phase(
        self,
        name: str,
        weight: int = 1,
        substeps: Optional[int] = None,
    ) -> None:
        # substeps define the granularity for fractional main-bar updates.
        if not self.enabled:
            return
        self._phase_name = name
        self._phase_weight = float(max(weight, 0))
        self._phase_substeps = max(substeps or 1, 1)
        self._phase_fraction = 0.0

    def update_phase(self, completed: int, total_substeps: Optional[int] = None) -> None:
        if not self.enabled:
            return
        if total_substeps is not None:
            self._phase_substeps = max(total_substeps, 1)
        if self._phase_substeps <= 0 or self._phase_weight <= 0:
            return
        fraction = min(completed / self._phase_substeps, 1.0)
        delta = (fraction - self._phase_fraction) * self._phase_weight
        if delta <= 0:
            return
        self._phase_fraction = fraction
        self._advance_main(delta)

    def advance_units(self, units: float) -> None:
        if not self.enabled:
            return
        if units <= 0:
            return
        self._advance_main(units)

    def finish_phase(self) -> None:
        if not self.enabled:
            return
        remaining = (1.0 - self._phase_fraction) * self._phase_weight
        if remaining <= 0:
            return
        self._phase_fraction = 1.0
        self._advance_main(remaining)

    def close(self) -> None:
        if not self.enabled or self.pbar is None:
            return
        self.pbar.remove(self.name)
        ProgBar.clear()
        self.pbar = None

    def _advance_main(self, delta: float) -> None:
        if self.pbar is None:
            return
        self._main_progress = min(self._main_progress + delta, self.total_units)
        self.pbar.update_subtask(self.name, self._main_progress)
