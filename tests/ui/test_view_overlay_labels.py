"""Labels laid over the Output view's page paint an opaque background."""

from collections.abc import Iterator

import numpy as np
import pytest
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui import theme
from scantailor.app.ui.image_views.output import ImageViewTab, OutputImageView
from scantailor.app.ui.theme.tokens import DARK, LIGHT, Tokens
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation

PAGE_SIZE = 400


@pytest.fixture(params=[("light", LIGHT), ("dark", DARK)], ids=["light", "dark"])
def scheme(
    request: pytest.FixtureRequest,
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Tokens]:
    """The application in one colour scheme, restored afterwards."""
    name, tokens = request.param
    monkeypatch.delenv(theme.THEME_ENV_VAR, raising=False)
    style = qapp.style().name()
    palette = qapp.palette()
    font = qapp.font()
    theme.apply_theme(qapp, name)
    yield tokens
    controller = qapp.findChild(QtCore.QObject, "scantailorThemeController")
    if controller is not None:
        shiboken6.delete(controller)
    qapp.setStyleSheet("")
    qapp.setStyle(style)
    qapp.setPalette(palette)
    qapp.setFont(font)


@pytest.fixture
def view(scheme: Tokens, qtbot: QtBot) -> OutputImageView:
    """An Output view on the Dewarping tab, zoomed so a black page fills it."""
    widget = OutputImageView()
    qtbot.addWidget(widget)
    widget.resize(QtCore.QSize(500, 500))
    widget.show()
    qtbot.waitExposed(widget)
    # A black page, so that a transparent label would show black.
    widget.set_image(
        np.zeros((PAGE_SIZE, PAGE_SIZE), dtype=np.uint8),
        transformation=ImageTransformation.from_size(
            PAGE_SIZE, PAGE_SIZE, Dpi.uniform(300)
        ),
    )
    widget.set_tabs(list(ImageViewTab), ImageViewTab.DEWARPING)
    # Zoomed in, the page fills the view, under the note along its bottom too.
    widget.set_zoom(3.0)
    return widget


def _corner_colour(view: OutputImageView, label: QtWidgets.QLabel) -> str:
    """The colour just inside the label's top left corner, clear of its text."""
    corner = label.geometry().topLeft() + QtCore.QPoint(1, 1)
    return view.grab().toImage().pixelColor(corner).name()


def test_the_dewarping_note_hides_the_page_under_it(
    qtbot: QtBot, view: OutputImageView, scheme: Tokens
) -> None:
    view.set_dewarp_note("2 of 33 lines kept, no page edge.")
    note = view.findChild(QtWidgets.QLabel, "dewarpMetricsNote")
    assert note is not None
    qtbot.waitUntil(note.isVisible)

    assert _corner_colour(view, note) == QtGui.QColor(scheme.viewport).name()


def test_a_tab_message_hides_the_page_under_it(
    qtbot: QtBot, view: OutputImageView, scheme: Tokens
) -> None:
    view.set_tab_message(
        ImageViewTab.DEWARPING, "Picture zones are only available in Mixed mode."
    )
    message = view.findChild(QtWidgets.QLabel, "outputTabMessage")
    assert message is not None
    qtbot.waitUntil(message.isVisible)

    assert _corner_colour(view, message) == QtGui.QColor(scheme.viewport).name()
