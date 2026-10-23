"""Show a widget under test inside the top-level window that owns it."""

from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot


def show_in_host(
    qtbot: QtBot, widget: QtWidgets.QWidget, width: int, height: int
) -> None:
    """Let ``widget`` fill its window at ``width`` x ``height``, then show it.

    Args:
        qtbot (QtBot): Waits until the window is exposed.
        widget (QtWidgets.QWidget): A child of the ``host`` fixture's window.
        width (int): Width the widget gets, in pixels.
        height (int): Height the widget gets, in pixels.
    """
    window = widget.window()
    layout = window.layout()
    assert layout is not None, "build the widget with parent=host"
    layout.addWidget(widget)
    window.resize(width, height)
    window.show()
    qtbot.wait_exposed(window)
