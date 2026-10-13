"""The centre guides of the Output view and their context menu toggle."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.output import OutputImageView
from scantailor.app.ui.theme.tokens import view_color
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
VIEW_SIZE = QtCore.QSize(300, 400)


@pytest.fixture
def view(qtbot: QtBot) -> OutputImageView:
    created = OutputImageView()
    qtbot.addWidget(created)
    created.resize(VIEW_SIZE)
    created.show()
    qtbot.waitExposed(created)
    page = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8)
    created.set_image(
        page,
        transformation=ImageTransformation.from_size(
            IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300)
        ),
    )
    return created


@dataclass(frozen=True)
class _Snapshot:
    """What a menu item showed while the menu was open."""

    text: str
    checkable: bool
    checked: bool
    icon_name: str

    @classmethod
    def of(cls, menu: QtWidgets.QMenu) -> list["_Snapshot"]:
        return [
            cls(a.text(), a.isCheckable(), a.isChecked(), a.icon().name())
            for a in menu.actions()
            if a.text()
        ]


def _is_inked(image: QtGui.QImage, x: int, y: int) -> bool:
    return image.pixelColor(x, y) != QtGui.QColor("white")


def _centre_lines_inked(view: OutputImageView) -> tuple[int, int]:
    """Count the non-white pixels on the centre column and on the centre row."""
    image = view.viewport().grab().toImage()
    x, y = image.width() // 2, image.height() // 2
    vertical = sum(_is_inked(image, x, row) for row in range(image.height()))
    horizontal = sum(_is_inked(image, column, y) for column in range(image.width()))
    return vertical, horizontal


class TestGuides:
    def test_hidden_by_default(self, view: OutputImageView):
        assert view.guides_shown() is False
        assert _centre_lines_inked(view) == (0, 0)

    def test_shown_guides_cross_the_view_centre_in_a_dashed_line(
        self, view: OutputImageView
    ):
        view.set_guides_shown(True)

        vertical, horizontal = _centre_lines_inked(view)

        # Dashes cover part of each line, never all of it and never none.
        assert 0 < vertical < VIEW_SIZE.height()
        assert 0 < horizontal < VIEW_SIZE.width()

    def test_guides_use_the_theme_guide_colour_over_the_page(
        self, view: OutputImageView
    ):
        view.set_guides_shown(True)
        image = view.viewport().grab().toImage()
        guide = view_color("guide")
        alpha = guide.alphaF()
        expected = tuple(
            round(channel * alpha + 255 * (1 - alpha))
            for channel in (guide.red(), guide.green(), guide.blue())
        )
        x = image.width() // 2

        inked = {
            image.pixelColor(x, y).getRgb()[:3]
            for y in range(image.height())
            if _is_inked(image, x, y) and abs(y - image.height() // 2) > 1
        }

        assert inked == {expected}

    def test_hiding_removes_them_again(self, view: OutputImageView):
        view.set_guides_shown(True)

        view.set_guides_shown(False)

        assert _centre_lines_inked(view) == (0, 0)

    def test_choice_is_kept_for_the_next_view(
        self, qtbot: QtBot, view: OutputImageView
    ):
        view.set_guides_shown(True)

        later = OutputImageView()
        qtbot.addWidget(later)

        assert later.guides_shown() is True

    def test_choice_is_stored_under_the_documented_key(self, view: OutputImageView):
        view.set_guides_shown(True)

        stored = QtCore.QSettings().value(
            OutputImageView.SHOW_GUIDES_KEY, False, type=bool
        )

        assert stored is True


class TestContextMenu:
    @staticmethod
    def _open_menu(
        view: OutputImageView, on_open: Callable[[QtWidgets.QMenu], None] | None = None
    ) -> list[_Snapshot]:
        """Send a context menu event, look into the menu it opens and close it.

        Args:
            view (OutputImageView): The view to ask for its menu.
            on_open (Callable[[QtWidgets.QMenu], None] | None): Called with the
                menu while it is open.

        Returns:
            list[_Snapshot]: What each item showed while the menu was open.
        """
        opened: list[list[_Snapshot]] = []

        def inspect() -> None:
            menu = QtWidgets.QApplication.activePopupWidget()
            if not isinstance(menu, QtWidgets.QMenu):
                QtCore.QTimer.singleShot(10, inspect)
                return
            opened.append(_Snapshot.of(menu))
            if on_open is not None:
                on_open(menu)
            menu.close()

        QtCore.QTimer.singleShot(0, inspect)
        view.contextMenuEvent(
            QtGui.QContextMenuEvent(
                QtGui.QContextMenuEvent.Reason.Mouse,
                QtCore.QPoint(10, 10),
                view.mapToGlobal(QtCore.QPoint(10, 10)),
            )
        )
        return opened[0]

    @staticmethod
    def _guides_action(menu: QtWidgets.QMenu) -> QtGui.QAction:
        (action,) = [a for a in menu.actions() if a.text() == "Show guides"]
        return action

    def test_menu_offers_a_checkable_show_guides_item_with_the_crosshair(
        self, view: OutputImageView
    ):
        (item,) = [i for i in self._open_menu(view) if i.text == "Show guides"]

        assert item.checkable
        assert not item.checked
        assert item.icon_name == "scantailor:crosshair"

    def test_item_reflects_the_current_state(self, view: OutputImageView):
        view.set_guides_shown(True)

        (item,) = [i for i in self._open_menu(view) if i.text == "Show guides"]

        assert item.checked

    def test_toggling_the_item_shows_the_guides(self, view: OutputImageView):
        self._open_menu(view, lambda menu: self._guides_action(menu).trigger())

        assert view.guides_shown() is True

    def test_toggling_twice_hides_them_again(self, view: OutputImageView):
        def toggle_twice(menu: QtWidgets.QMenu) -> None:
            self._guides_action(menu).trigger()
            self._guides_action(menu).trigger()

        self._open_menu(view, toggle_twice)

        assert view.guides_shown() is False

    def test_zone_items_stay_in_the_menu(self, view: OutputImageView):
        texts = [item.text for item in self._open_menu(view)]

        assert "Create Picture Zone" in texts
        assert "Create Fill Zone" in texts
