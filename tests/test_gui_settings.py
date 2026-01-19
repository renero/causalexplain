from causalexplain.gui.settings import merge_settings


def test_merge_settings_overrides_defaults() -> None:
    defaults = {"a": 1, "b": 2}
    stored = {"b": 3, "c": 4}

    merged = merge_settings(stored, defaults)

    assert merged["a"] == 1
    assert merged["b"] == 3
    assert merged["c"] == 4


def test_merge_settings_ignores_non_dict() -> None:
    defaults = {"a": 1}

    merged = merge_settings(["not", "a", "dict"], defaults)

    assert merged == defaults
