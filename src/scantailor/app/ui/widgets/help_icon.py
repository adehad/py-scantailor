"""A help button beside every control that has a tooltip.

``install_help_icons`` watches a panel and everything inside it. When a widget's
tooltip becomes non-empty, a small info button is placed right after the widget in
its layout, level with the widget's top edge like a badge on its corner. The
button carries the same text: it shows on hover like any tooltip, and on a click
or on Space, so the help can be read without a mouse. Clearing the tooltip hides
the button again, and the button is hidden with its control.

A container with the ``helpGroup`` property set gets one button for its own
tooltip, and the controls inside it keep their tooltips on hover without
buttons of their own. That suits a row of buttons too narrow for an icon each.
"""

from typing import cast

from PySide6 import QtCore, QtGui, QtWidgets

from scantailor.app.ui.icons import icon

HELP_ICON_PROPERTY = "helpIcon"
HELP_GROUP_PROPERTY = "helpGroup"
_GLYPH = "info"
# The glyph reads as a badge on the control rather than as a control itself.
_GLYPH_TO_LINE = 0.75
_GAP = 2


def help_icon_size(font: QtGui.QFont) -> int:
    """Return the edge length of the help glyph for a font.

    Args:
        font (QtGui.QFont): Font of the button or the application.

    Returns:
        int: Three quarters of the font's line height, in logical pixels.
    """
    return round(QtGui.QFontMetrics(font).height() * _GLYPH_TO_LINE)


class HelpButton(QtWidgets.QToolButton):
    """Info button that repeats the tooltip of the control it sits beside."""

    def __init__(
        self, control: QtWidgets.QWidget, parent: QtWidgets.QWidget | None = None
    ) -> None:
        """Create the button for a control.

        Args:
            control (QtWidgets.QWidget): Widget whose tooltip the button shows.
            parent (QtWidgets.QWidget | None): Parent widget, normally the
                control's parent.
        """
        super().__init__(parent)
        self._control = control
        self.setObjectName(f"{control.objectName() or 'control'}Help")
        self.setProperty(HELP_ICON_PROPERTY, True)
        self.setAutoRaise(True)
        self.setIcon(icon(_GLYPH))
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.TabFocus)
        self.setCursor(QtCore.Qt.CursorShape.WhatsThisCursor)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.clicked.connect(self.show_help)
        control.destroyed.connect(self.deleteLater)
        self._fit_to_font()
        self.sync()

    @property
    def control(self) -> QtWidgets.QWidget:
        """The widget whose help this button shows."""
        return self._control

    def sync(self) -> None:
        """Take the control's current tooltip, name and visibility."""
        text = self._control.toolTip()
        self.setToolTip(text)
        self.setAccessibleDescription(text)
        self.setAccessibleName(self._accessible_name())
        self.setHidden(not text or self._control.isHidden())

    def show_help(self) -> None:
        """Show the help text under the button, as hovering would."""
        QtWidgets.QToolTip.showText(
            self.mapToGlobal(QtCore.QPoint(0, self.height())), self.toolTip(), self
        )

    def changeEvent(self, event: QtCore.QEvent) -> None:
        """Follow font and language changes.

        Args:
            event (QtCore.QEvent): The event.
        """
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.FontChange:
            self._fit_to_font()
        elif event.type() == QtCore.QEvent.Type.LanguageChange:
            self.sync()

    def _fit_to_font(self) -> None:
        glyph = help_icon_size(self.font())
        self.setIconSize(QtCore.QSize(glyph, glyph))
        # The button stays a full line square, a larger click target than the
        # glyph it shows.
        line = QtGui.QFontMetrics(self.font()).height()
        self.setFixedSize(line, line)

    def _accessible_name(self) -> str:
        name = self._control.accessibleName()
        if not name and isinstance(self._control, QtWidgets.QAbstractButton):
            name = self._control.text().replace("&", "")
        if not name:
            return QtCore.QCoreApplication.translate("HelpButton", "Help")
        return QtCore.QCoreApplication.translate("HelpButton", "Help: %1").replace(
            "%1", name
        )


def _layout_holding(widget: QtWidgets.QWidget) -> QtWidgets.QLayout | None:
    """Find the layout, possibly nested, that has the widget as a direct item."""
    parent = widget.parentWidget()
    top = parent.layout() if parent is not None else None
    pending = [top] if top is not None else []
    while pending:
        layout = pending.pop()
        if layout.indexOf(widget) >= 0:
            return layout
        for index in range(layout.count()):
            item = layout.itemAt(index)
            inner = item.layout() if item is not None else None
            if inner is not None:
                pending.append(inner)
    return None


def _pair(control: QtWidgets.QWidget, button: HelpButton) -> QtWidgets.QHBoxLayout:
    """Lay the control and its button out as one row."""
    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(_GAP)
    row.addWidget(control)
    row.addWidget(button, 0, QtCore.Qt.AlignmentFlag.AlignTop)
    # A check box or radio button is as wide as its column; the stretch keeps the
    # icon next to the text instead of at the far edge.
    if isinstance(control, QtWidgets.QCheckBox | QtWidgets.QRadioButton):
        row.addStretch(1)
    return row


def _place_after(control: QtWidgets.QWidget, button: HelpButton) -> bool:
    """Put the button right after the control, in the control's layout cell.

    Args:
        control (QtWidgets.QWidget): Widget with the tooltip.
        button (HelpButton): Its help button.

    Returns:
        bool: False when the control is in no layout this function understands.
    """
    layout = _layout_holding(control)
    if layout is None:
        return False
    index = layout.indexOf(control)
    if isinstance(layout, QtWidgets.QGridLayout):
        row, column, row_span, column_span = cast(
            tuple[int, int, int, int], layout.getItemPosition(index)
        )
        layout.takeAt(index)
        layout.addLayout(_pair(control, button), row, column, row_span, column_span)
        return True
    if isinstance(layout, QtWidgets.QBoxLayout):
        stretch = layout.stretch(index)
        layout.takeAt(index)
        layout.insertLayout(index, _pair(control, button), stretch)
        return True
    if isinstance(layout, QtWidgets.QFormLayout):
        form_row, role = cast(
            tuple[int, QtWidgets.QFormLayout.ItemRole],
            layout.getWidgetPosition(control),
        )
        layout.takeAt(index)
        layout.setLayout(form_row, role, _pair(control, button))
        return True
    return False


def help_button_for(control: QtWidgets.QWidget) -> HelpButton | None:
    """Return the help button placed beside a control, if there is one.

    Args:
        control (QtWidgets.QWidget): The control.

    Returns:
        HelpButton | None: Its button, or None when it never had a tooltip.
    """
    parent = control.parentWidget()
    if parent is None:
        return None
    for button in parent.findChildren(
        HelpButton, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly
    ):
        if button.control is control:
            return button
    return None


class HelpIcons(QtCore.QObject):
    """Event filter that keeps a help button beside every widget with a tooltip."""

    def __init__(self, root: QtWidgets.QWidget) -> None:
        """Watch a widget and its descendants, now and as they are added.

        Args:
            root (QtWidgets.QWidget): Panel to watch; it owns the filter.
        """
        super().__init__(root)
        self._watch_tree(root)

    def _watch_tree(self, widget: QtWidgets.QWidget) -> None:
        for each in (widget, *widget.findChildren(QtWidgets.QWidget)):
            if not isinstance(each, HelpButton):
                each.installEventFilter(self)
                self._refresh(each)

    def _in_help_group(self, widget: QtWidgets.QWidget) -> bool:
        root = self.parent()
        ancestor = widget.parentWidget()
        while ancestor is not None and ancestor is not root:
            if ancestor.property(HELP_GROUP_PROPERTY):
                return True
            ancestor = ancestor.parentWidget()
        return False

    def _refresh(self, widget: QtWidgets.QWidget) -> None:
        if self._in_help_group(widget):
            return
        button = help_button_for(widget)
        if button is not None:
            button.sync()
            return
        if not widget.toolTip():
            return
        button = HelpButton(widget, widget.parentWidget())
        if not _place_after(widget, button):
            button.deleteLater()

    def _later(self, widget: QtWidgets.QWidget) -> None:
        """Refresh once control returns to the event loop.

        A widget being constructed, or not yet added to a layout, cannot be
        placed beside yet.
        """

        def refresh() -> None:
            try:
                self._watch_tree(widget)
            except RuntimeError:
                pass  # Deleted before the event loop came back.

        QtCore.QTimer.singleShot(0, self, refresh)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Add, update, show or hide a help button as its control changes.

        Args:
            watched (QtCore.QObject): A watched widget.
            event (QtCore.QEvent): The event.

        Returns:
            bool: Always False; the event is never consumed.
        """
        kind = event.type()
        if kind == QtCore.QEvent.Type.ToolTipChange and isinstance(
            watched, QtWidgets.QWidget
        ):
            button = help_button_for(watched)
            if button is not None:
                button.sync()
            elif not self._in_help_group(watched):
                self._later(watched)
        elif kind in (
            QtCore.QEvent.Type.ShowToParent,
            QtCore.QEvent.Type.HideToParent,
        ) and isinstance(watched, QtWidgets.QWidget):
            button = help_button_for(watched)
            if button is not None:
                button.sync()
        elif kind == QtCore.QEvent.Type.ChildPolished and isinstance(
            event, QtCore.QChildEvent
        ):
            # ChildAdded would arrive while a Python subclass is still in its
            # constructor, and reading event.child() then caches a wrapper of
            # the wrong type for that object.
            child = event.child()
            if isinstance(child, QtWidgets.QWidget) and not isinstance(
                child, HelpButton
            ):
                self._later(child)
        return False


def install_help_icons(root: QtWidgets.QWidget) -> HelpIcons:
    """Give every widget under ``root`` that has a tooltip a help button.

    Args:
        root (QtWidgets.QWidget): Panel to watch, such as a stage's options.

    Returns:
        HelpIcons: The filter, owned by ``root``.
    """
    return HelpIcons(root)
