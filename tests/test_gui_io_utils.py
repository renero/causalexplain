import pytest

from causalexplain.gui.io_utils import (
    ensure_file,
    normalize_output_value,
    sanitize_output_name,
)


def test_ensure_file_validates_suffix(tmp_path) -> None:
    file_path = tmp_path / "data.csv"
    file_path.write_text("a,b\n1,2\n")

    assert ensure_file(str(file_path), ".csv") == str(file_path)

    with pytest.raises(ValueError):
        ensure_file(str(file_path), ".json")

    with pytest.raises(FileNotFoundError):
        ensure_file(str(tmp_path / "missing.csv"), ".csv")


def test_normalize_output_value() -> None:
    assert normalize_output_value("", ".dot") is None
    assert normalize_output_value("output", ".dot") == "output.dot"
    assert normalize_output_value("output.dot", ".dot") == "output.dot"
    assert normalize_output_value("output.DOT", ".dot") == "output.DOT"
    assert normalize_output_value("output.csv", ".dot") == ""


def test_sanitize_output_name() -> None:
    assert sanitize_output_name(" data.csv ") == "data"
    assert sanitize_output_name("data.dot") == "data"
    assert sanitize_output_name("data") == "data"
    assert sanitize_output_name("DATA.CSV") == "DATA"
