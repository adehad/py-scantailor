"""NonOwningWidget keeps its children alive when it is disposed of."""

import shiboken6
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.widgets.non_owning_widget import NonOwningWidget


def _flush_deferred_deletes() -> None:
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_children_survive_delete_later_of_the_container(qtbot: QtBot):
    container = NonOwningWidget()
    child = QtWidgets.QLabel("kept", container)
    nested = QtWidgets.QLabel("nested", child)

    container.deleteLater()
    _flush_deferred_deletes()

    assert not shiboken6.isValid(container)
    assert shiboken6.isValid(child)
    assert child.parent() is None
    assert nested.parent() is child
    child.deleteLater()
    _flush_deferred_deletes()


def test_children_are_released_without_deleting_the_container(qtbot: QtBot):
    container = NonOwningWidget()
    qtbot.addWidget(container)
    first = QtWidgets.QLabel("first", container)
    second = QtWidgets.QLabel("second", container)

    container.release_children()

    assert first.parent() is None
    assert second.parent() is None
    assert container.findChildren(QtWidgets.QWidget) == []


def test_a_plain_parent_still_deletes_its_children(qtbot: QtBot):
    plain = QtWidgets.QWidget()
    child = QtWidgets.QLabel("owned", plain)

    plain.deleteLater()
    _flush_deferred_deletes()

    assert not shiboken6.isValid(child)
