"""The batch processing lower panel is defined by its .ui form."""

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.panels import system_load
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


def test_system_load_shows_the_saved_count_without_saving_or_announcing_it(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(system_load.os, "cpu_count", lambda: 8)
    QtCore.QSettings().setValue(system_load._SETTINGS_KEY, 3)
    saves: list[int] = []
    monkeypatch.setattr(SystemLoadWidget, "_save_state", lambda self: saves.append(1))
    announced: list[int] = []
    monkeypatch.setattr(
        SystemLoadWidget,
        "_on_value_changed",
        lambda self, value: announced.append(value),
    )

    widget = SystemLoadWidget()
    qtbot.addWidget(widget)

    assert widget.get_load() == 3
    assert saves == []
    assert announced == []


def test_system_load_with_nothing_saved_starts_at_every_thread_and_stores_nothing(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(system_load.os, "cpu_count", lambda: 8)

    widget = SystemLoadWidget()
    qtbot.addWidget(widget)

    assert widget.get_load() == 8
    assert not QtCore.QSettings().contains(system_load._SETTINGS_KEY)
    assert not widget.findChild(QtWidgets.QToolButton, "plusBtn").isEnabled()


def test_system_load_lays_out_its_form_so_the_panel_can_size_it(qtbot: QtBot):
    widget = SystemLoadWidget()
    qtbot.addWidget(widget)

    assert widget.layout() is not None
    assert widget.sizeHint().isValid()
    assert widget.sizeHint().width() > 0
