"""Cyclic garbage is collected on the GUI thread only, never under Qt's feet."""

import gc
import threading

import pytest
import shiboken6
from PySide6 import QtCore, QtWidgets

from scantailor.app.ui import theme
from scantailor.app.ui.garbage_collection import (
    GuiThreadCollector,
    collect_garbage_on_the_gui_thread,
)


def widget_in_a_cycle() -> tuple[list[QtWidgets.QWidget], list[str]]:
    """Make a top-level widget that only the cyclic collector can free.

    Returns:
        tuple[list[QtWidgets.QWidget], list[str]]: A list holding the only outside
            reference to the widget, and the names of the threads it was
            destroyed on, filled in when that happens.
    """
    widget = QtWidgets.QDialog()
    widget.ensurePolished()
    widget.cycle = widget  # type: ignore[attr-defined]
    destroyed_on: list[str] = []
    widget.destroyed.connect(
        lambda: destroyed_on.append(threading.current_thread().name),
        QtCore.Qt.ConnectionType.DirectConnection,
    )
    return [widget], destroyed_on


def collect_as_python_would_here() -> None:
    """Run the collection that Python starts by itself when automatic collection is on.

    Which allocation starts an automatic collection cannot be pinned down in a
    test, so this stands in for one at the point where it would do harm.
    """
    if gc.isenabled():
        gc.collect()


def make_garbage_past_the_threshold() -> None:
    """Leave enough unreachable cycles behind to start an automatic collection."""
    for _ in range(gc.get_threshold()[0] * 2):
        cycle: list[object] = []
        cycle.append(cycle)


def test_starting_turns_automatic_collection_off_and_stopping_back_on(qapp):
    collector = collect_garbage_on_the_gui_thread(qapp)
    assert not gc.isenabled()
    collector.stop()
    assert gc.isenabled()
    collect_garbage_on_the_gui_thread(qapp)
    assert not gc.isenabled()


def test_installing_twice_keeps_one_collector(qapp):
    first = collect_garbage_on_the_gui_thread(qapp)
    assert collect_garbage_on_the_gui_thread(qapp) is first
    assert len(qapp.findChildren(GuiThreadCollector)) == 1


def test_nothing_is_collected_until_enough_objects_are_pending(qapp):
    gc.collect()
    owner, destroyed_on = widget_in_a_cycle()
    owner.clear()
    assert GuiThreadCollector.collect_if_due() == 0
    assert destroyed_on == []
    make_garbage_past_the_threshold()
    assert GuiThreadCollector.collect_if_due() > 0
    assert destroyed_on == [threading.main_thread().name]


def test_the_timer_collects_on_the_gui_thread(qapp, qtbot):
    gc.collect()
    owner, destroyed_on = widget_in_a_cycle()
    owner.clear()
    make_garbage_past_the_threshold()
    qtbot.waitUntil(lambda: destroyed_on != [], timeout=5000)
    assert destroyed_on == [threading.main_thread().name]


def test_garbage_a_worker_thread_makes_is_freed_on_the_gui_thread(qapp):
    owner, destroyed_on = widget_in_a_cycle()

    def work() -> None:
        owner.clear()
        collect_as_python_would_here()

    worker = threading.Thread(target=work, name="render worker")
    worker.start()
    worker.join()
    assert destroyed_on == []
    gc.collect()
    assert destroyed_on == [threading.main_thread().name]


class ReleasesGarbageWhileRestyling(QtCore.QObject):
    """Drops the last owner of a widget cycle mid-restyle, then collects.

    Qt calls into Python for every widget while it restyles them all, so any of
    those calls can be the one that starts an automatic collection.
    """

    def __init__(self, owner: list[QtWidgets.QWidget]) -> None:
        super().__init__()
        self.owner = owner

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.StyleChange and self.owner:
            self.owner.clear()
            collect_as_python_would_here()
        return False


@pytest.mark.parametrize("first_scheme", [None, "light"])
def test_a_widget_released_mid_restyle_is_not_deleted_under_qt(
    qapp, monkeypatch, first_scheme
):
    monkeypatch.delenv(theme.THEME_ENV_VAR, raising=False)
    style = qapp.style().name()
    palette = qapp.palette()
    try:
        if first_scheme is not None:
            theme.apply_theme(qapp, first_scheme)
        owners = [widget_in_a_cycle() for _ in range(20)]
        dialogs = [widget for owner, _ in owners for widget in owner]
        releaser = ReleasesGarbageWhileRestyling(dialogs)
        for owner, _ in owners:
            owner.clear()
        qapp.installEventFilter(releaser)
        try:
            theme.apply_theme(qapp, "dark")
        finally:
            qapp.removeEventFilter(releaser)
        assert all(destroyed_on == [] for _, destroyed_on in owners)
        gc.collect()
        assert all(destroyed_on != [] for _, destroyed_on in owners)
    finally:
        controller = qapp.findChild(QtCore.QObject, "scantailorThemeController")
        if controller is not None:
            shiboken6.delete(controller)
        qapp.setStyleSheet("")
        qapp.setStyle(style)
        qapp.setPalette(palette)
