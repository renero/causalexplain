from causalexplain.common.progress import ProgressManager


def test_progress_manager_snapshot_tracks_fraction_without_cli_bar() -> None:
    progress = ProgressManager(total_units=10, render_cli=False)

    progress.start_phase("DNN_HPO", weight=4, substeps=4)
    progress.update_phase(1)
    snapshot = progress.snapshot()

    assert snapshot["phase_name"] == "DNN_HPO"
    assert snapshot["completed_units"] == 1.0
    assert snapshot["fraction"] == 0.1
    assert snapshot["percent"] == 10.0

    progress.finish_phase()
    progress.start_phase("SHAP_fit", weight=6, substeps=3)
    progress.update_phase(3)
    snapshot = progress.snapshot()

    assert snapshot["phase_name"] == "SHAP_fit"
    assert snapshot["completed_units"] == 10.0
    assert snapshot["fraction"] == 1.0
    assert snapshot["percent"] == 100.0
