"""The batch processing lower panel is defined by its .ui form."""

from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.panels.batch_processing import BatchProcessingLowerPanel
from scantailor.app.ui.panels.system_load import SystemLoadWidget


def test_panel_shows_the_forms_system_load_widget_and_beep_checkbox(qtbot: QtBot):
    panel = BatchProcessingLowerPanel()
    qtbot.addWidget(panel)

    assert panel.findChild(SystemLoadWidget, "systemLoadWidget") is not None
    beep = panel.findChild(QtWidgets.QCheckBox, "beepWhenFinished")
    assert beep is not None
    assert beep.text() == "Beep when finished"


def test_beep_checkbox_toggle_is_reported_and_readable(qtbot: QtBot):
    panel = BatchProcessingLowerPanel()
    qtbot.addWidget(panel)

    with qtbot.waitSignal(panel.beep_setting_changed) as blocker:
        panel.set_beep_enabled(True)

    assert blocker.args == [True]
    assert panel.is_beep_enabled()


def test_load_changes_are_forwarded_from_the_system_load_widget(qtbot: QtBot):
    panel = BatchProcessingLowerPanel()
    qtbot.addWidget(panel)
    target = 1 if panel.get_load() != 1 else 2

    with qtbot.waitSignal(panel.load_changed) as blocker:
        panel.set_load(target)

    assert blocker.args == [panel.get_load()]
