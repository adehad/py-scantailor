"""Margin link and guide state that the Page Layout view and options share.

The link buttons live in the options widget and the guides are drawn by the
view, yet both must see one answer to "is left linked to right" and "where are
the guides". They hold one instance of this object and listen to its signals.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from PySide6.QtCore import QObject, QSettings, Signal

LEFT_RIGHT_LINKED_KEY = "margins/leftRightLinked"
TOP_BOTTOM_LINKED_KEY = "margins/topBottomLinked"


@dataclass(eq=False)
class Guide:
    """A horizontal or vertical alignment guide.

    Attributes:
        position_mm: Distance in millimetres from the centre of the page's
            outer rectangle, so a guide means the same on pages of every size;
            y for a horizontal guide and x for a vertical one.
        is_horizontal: True for a horizontal guide.
    """

    position_mm: float
    is_horizontal: bool


class PageLayoutState(QObject):
    """Link, guide and hard-margin-rectangle state of the Page Layout stage.

    Signals:
        links_changed: A margin link was switched.
        guides_changed: A guide was added, moved or removed.
        middle_rect_shown_changed: The hard margins rectangle was shown or hidden.
    """

    links_changed = Signal()
    guides_changed = Signal()
    middle_rect_shown_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        """Start with the links the user last chose, no guides and the margins shown.

        The links are linked when the user never chose. Only a user's choice is
        stored, by ``remember_left_right_link`` and ``remember_top_bottom_link``,
        so a pair released because a page has unequal margins stays linked in the
        next session.

        Args:
            parent (QObject | None): Owner of this object.
        """
        super().__init__(parent)
        settings = QSettings()
        self._left_right_linked = bool(
            settings.value(LEFT_RIGHT_LINKED_KEY, True, type=bool)
        )
        self._top_bottom_linked = bool(
            settings.value(TOP_BOTTOM_LINKED_KEY, True, type=bool)
        )
        self._guides: list[Guide] = []
        self._middle_rect_shown = True

    @property
    def left_right_linked(self) -> bool:
        """Whether the left and right margins change together."""
        return self._left_right_linked

    @left_right_linked.setter
    def left_right_linked(self, linked: bool) -> None:
        if linked != self._left_right_linked:
            self._left_right_linked = linked
            self.links_changed.emit()

    @property
    def top_bottom_linked(self) -> bool:
        """Whether the top and bottom margins change together."""
        return self._top_bottom_linked

    @top_bottom_linked.setter
    def top_bottom_linked(self, linked: bool) -> None:
        if linked != self._top_bottom_linked:
            self._top_bottom_linked = linked
            self.links_changed.emit()

    @staticmethod
    def remember_left_right_link(linked: bool) -> None:
        """Keep the user's choice for the left and right link for later sessions."""
        QSettings().setValue(LEFT_RIGHT_LINKED_KEY, linked)

    @staticmethod
    def remember_top_bottom_link(linked: bool) -> None:
        """Keep the user's choice for the top and bottom link for later sessions."""
        QSettings().setValue(TOP_BOTTOM_LINKED_KEY, linked)

    @property
    def middle_rect_shown(self) -> bool:
        """Whether the hard margins rectangle is shown and can be dragged."""
        return self._middle_rect_shown

    @middle_rect_shown.setter
    def middle_rect_shown(self, shown: bool) -> None:
        if shown != self._middle_rect_shown:
            self._middle_rect_shown = shown
            self.middle_rect_shown_changed.emit()

    def load(
        self, guides: Iterable[tuple[float, bool]], middle_rect_shown: bool
    ) -> None:
        """Take stored guides and the hard margins rectangle choice without signals.

        Args:
            guides (Iterable[tuple[float, bool]]): Distance in millimetres from the
                centre of the outer rectangle and whether the guide is horizontal,
                for each guide.
            middle_rect_shown (bool): Whether the hard margins rectangle is shown.
        """
        self._guides = [Guide(position, horizontal) for position, horizontal in guides]
        self._middle_rect_shown = middle_rect_shown

    @property
    def guides(self) -> list[Guide]:
        """The guides, as a new list."""
        return list(self._guides)

    def add_guide(self, is_horizontal: bool, position_mm: float) -> Guide:
        """Add a guide.

        Args:
            is_horizontal (bool): True for a horizontal guide.
            position_mm (float): Distance from the centre of the outer rectangle.

        Returns:
            Guide: The new guide.
        """
        guide = Guide(position_mm, is_horizontal)
        self._guides.append(guide)
        self.guides_changed.emit()
        return guide

    def move_guide(self, guide: Guide, position_mm: float) -> None:
        """Move a guide.

        Args:
            guide (Guide): A guide of this state.
            position_mm (float): New distance from the centre of the outer rectangle.
        """
        if guide in self._guides and guide.position_mm != position_mm:
            guide.position_mm = position_mm
            self.guides_changed.emit()

    def remove_guide(self, guide: Guide) -> None:
        """Remove a guide.

        Args:
            guide (Guide): The guide to remove; ignored if it is not here.
        """
        if guide in self._guides:
            self._guides.remove(guide)
            self.guides_changed.emit()

    def clear_guides(self) -> None:
        """Remove every guide."""
        if self._guides:
            self._guides.clear()
            self.guides_changed.emit()
