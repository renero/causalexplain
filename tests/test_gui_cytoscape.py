from causalexplain.gui import cytoscape as cygui


def test_cytoscape_layout_config_preset() -> None:
    config = cygui._cytoscape_layout_config("dagre", "LR", True)

    assert config == {"name": "preset"}


def test_cytoscape_layout_config_elk_direction() -> None:
    config = cygui._cytoscape_layout_config("elk", "LR", False)

    assert config["name"] == "elk"
    assert config["elk"]["elk.direction"] == "RIGHT"


def test_cytoscape_sanity_check_counts() -> None:
    counts = cygui._cytoscape_sanity_check()

    assert counts["edge_true"] == 1
    assert counts["edge_reversed"] == 1
    assert counts["edge_false_positive"] == 1
    assert counts["edge_false_negative"] == 1
