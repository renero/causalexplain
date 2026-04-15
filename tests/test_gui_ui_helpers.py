import asyncio
import io

from causalexplain.gui.ui_helpers import make_upload_handler


class DummyInput:
    def __init__(self) -> None:
        self.value = ""
        self.updated = False

    def update(self) -> None:
        self.updated = True


class DummyLabel:
    def __init__(self) -> None:
        self.text = ""
        self.updated = False

    def update(self) -> None:
        self.updated = True


def test_make_upload_handler_updates_input_and_settings_for_modern_event(
    tmp_path,
) -> None:
    input_el = DummyInput()
    status_label = DummyLabel()
    storage = {}
    settings = {}
    event = type(
        "UploadEvent",
        (),
        {
            "name": "dataset.csv",
            "content": io.BytesIO(b"a,b\n1,2\n"),
        },
    )()

    handler = make_upload_handler(
        input_el,
        storage,
        "train_settings",
        settings,
        "dataset_path",
        str(tmp_path),
        ".csv",
        status_label=status_label,
    )

    asyncio.run(handler(event))

    assert input_el.updated is True
    assert status_label.updated is True
    assert settings["dataset_path"] == input_el.value
    assert storage["train_settings"]["dataset_path"] == input_el.value
    assert input_el.value.endswith("dataset.csv")
