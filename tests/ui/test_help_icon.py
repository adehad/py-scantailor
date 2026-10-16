"""Help buttons beside controls that have a tooltip, and help on every option."""

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.options_widgets import (
    deskew,
    fix_orientation,
    output,
    page_layout,
    page_split,
    select_content,
)
from scantailor.app.ui.widgets.collapsible_group_box import CollapsibleGroupBox
from scantailor.app.ui.widgets.help_icon import (
    HELP_GROUP_PROPERTY,
    HELP_ICON_PROPERTY,
    HelpButton,
    help_button_for,
    help_icon_size,
    install_help_icons,
)
from scantailor.core.stage_sequence import StageSequence

OPTION_CONTROLS = (
    QtWidgets.QAbstractButton,
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QComboBox,
    QtWidgets.QAbstractSlider,
    QtWidgets.QLineEdit,
)


@pytest.fixture
def panel(qtbot: QtBot) -> QtWidgets.QWidget:
    """A panel with a check box in a column and a spin box in a grid cell."""
    root = QtWidgets.QWidget()
    column = QtWidgets.QVBoxLayout(root)
    with_tip = QtWidgets.QCheckBox("Fill margins", root)
    with_tip.setObjectName("withTip")
    with_tip.setToolTip("Paint the margins white.")
    column.addWidget(with_tip)
    without_tip = QtWidgets.QCheckBox("Plain", root)
    without_tip.setObjectName("withoutTip")
    column.addWidget(without_tip)
    grid = QtWidgets.QGridLayout()
    grid.addWidget(QtWidgets.QLabel("Window:", root), 0, 0)
    spin = QtWidgets.QSpinBox(root)
    spin.setObjectName("spin")
    spin.setToolTip("Size of the window.")
    grid.addWidget(spin, 0, 1)
    column.addLayout(grid)
    qtbot.addWidget(root)
    install_help_icons(root)
    root.show()
    qtbot.wait_exposed(root)
    return root


def _child[T: QtWidgets.QWidget](
    root: QtWidgets.QWidget, kind: type[T], name: str
) -> T:
    found = root.findChild(kind, name)
    assert found is not None, name
    return found


def test_only_controls_with_a_tooltip_get_a_help_button(panel: QtWidgets.QWidget):
    with_tip = _child(panel, QtWidgets.QCheckBox, "withTip")
    without_tip = _child(panel, QtWidgets.QCheckBox, "withoutTip")

    button = help_button_for(with_tip)

    assert button is not None and button.isVisible()
    assert button.toolTip() == "Paint the margins white."
    assert help_button_for(without_tip) is None
    assert len(panel.findChildren(HelpButton)) == 2


def test_help_button_sits_at_the_top_right_corner_of_its_control(
    panel: QtWidgets.QWidget,
):
    spin = _child(panel, QtWidgets.QSpinBox, "spin")
    button = help_button_for(spin)
    assert button is not None

    spin_box = spin.geometry()
    assert button.geometry().left() >= spin_box.right()
    assert button.geometry().left() - spin_box.right() <= 4
    assert button.geometry().top() <= spin_box.center().y()
    assert abs(button.geometry().top() - spin_box.top()) <= 2


def test_tooltip_set_later_adds_a_help_button(qtbot: QtBot, panel: QtWidgets.QWidget):
    without_tip = _child(panel, QtWidgets.QCheckBox, "withoutTip")

    without_tip.setToolTip("Added after the panel was built.")

    qtbot.waitUntil(lambda: help_button_for(without_tip) is not None)
    button = help_button_for(without_tip)
    assert button is not None
    qtbot.waitUntil(button.isVisible)
    assert button.toolTip() == "Added after the panel was built."


def test_control_added_later_with_a_tooltip_gets_a_help_button(
    qtbot: QtBot, panel: QtWidgets.QWidget
):
    late = QtWidgets.QCheckBox("Late", panel)
    late.setToolTip("Built after the helper was installed.")
    layout = panel.layout()
    assert isinstance(layout, QtWidgets.QVBoxLayout)
    layout.addWidget(late)
    late.show()

    qtbot.waitUntil(lambda: help_button_for(late) is not None)


def test_changing_the_tooltip_updates_the_help_and_clearing_it_hides_the_button(
    panel: QtWidgets.QWidget,
):
    with_tip = _child(panel, QtWidgets.QCheckBox, "withTip")
    button = help_button_for(with_tip)
    assert button is not None

    with_tip.setToolTip("New wording.")
    assert button.toolTip() == "New wording."
    assert button.accessibleDescription() == "New wording."

    with_tip.setToolTip("")
    assert button.isHidden()

    with_tip.setToolTip("Back again.")
    assert not button.isHidden()


def test_help_button_hides_and_shows_with_its_control(panel: QtWidgets.QWidget):
    spin = _child(panel, QtWidgets.QSpinBox, "spin")
    button = help_button_for(spin)
    assert button is not None

    spin.setVisible(False)
    assert button.isHidden()
    spin.setVisible(True)
    assert not button.isHidden()


def test_clicking_or_pressing_space_shows_the_tooltip_text(
    qtbot: QtBot, panel: QtWidgets.QWidget, monkeypatch: pytest.MonkeyPatch
):
    shown: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QToolTip,
        "showText",
        lambda _pos, text, _widget=None, *_args: shown.append(text),
    )
    button = help_button_for(_child(panel, QtWidgets.QCheckBox, "withTip"))
    assert button is not None

    qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
    button.setFocus()
    qtbot.keyClick(button, QtCore.Qt.Key.Key_Space)

    assert shown == ["Paint the margins white.", "Paint the margins white."]


def test_help_button_is_reachable_by_keyboard_and_named_for_its_control(
    panel: QtWidgets.QWidget,
):
    button = help_button_for(_child(panel, QtWidgets.QCheckBox, "withTip"))
    spin_button = help_button_for(_child(panel, QtWidgets.QSpinBox, "spin"))
    assert button is not None and spin_button is not None

    assert button.focusPolicy() & QtCore.Qt.FocusPolicy.TabFocus
    assert button.accessibleName() == "Help: Fill margins"
    assert spin_button.accessibleName() == "Help"
    assert button.property(HELP_ICON_PROPERTY) is True


def test_help_glyph_is_three_quarters_of_a_line_and_grows_with_the_font(
    panel: QtWidgets.QWidget,
):
    button = help_button_for(_child(panel, QtWidgets.QCheckBox, "withTip"))
    assert button is not None
    line = QtGui.QFontMetrics(button.font()).height()
    before = button.iconSize().height()
    assert before == round(0.75 * line)
    assert button.size() == QtCore.QSize(line, line)

    font = QtGui.QFont(panel.font())
    font.setPointSizeF(font.pointSizeF() * 2)
    panel.setFont(font)

    after = button.iconSize().height()
    assert after > before
    assert after == help_icon_size(button.font())


def test_a_help_group_gets_one_button_and_its_buttons_keep_their_own_tooltips(
    qtbot: QtBot,
):
    root = QtWidgets.QWidget()
    column = QtWidgets.QVBoxLayout(root)
    row = QtWidgets.QWidget(root)
    row.setProperty(HELP_GROUP_PROPERTY, True)
    row.setToolTip("Pick one mode.")
    row_layout = QtWidgets.QHBoxLayout(row)
    modes = [QtWidgets.QPushButton(text, row) for text in ("Off", "Auto")]
    for mode in modes:
        mode.setToolTip(f"{mode.text()} mode.")
        row_layout.addWidget(mode)
    column.addWidget(row)
    qtbot.addWidget(root)
    install_help_icons(root)

    modes[0].setToolTip("Off, reworded.")
    qtbot.wait(10)

    assert [button.control for button in root.findChildren(HelpButton)] == [row]
    assert modes[0].toolTip() == "Off, reworded."


def _option_widgets() -> list[QtWidgets.QWidget]:
    sequence = StageSequence()
    return [
        fix_orientation.OptionsWidget(sequence.fix_orientation_filter),
        page_split.OptionsWidget(sequence.page_split_filter),
        deskew.OptionsWidget(sequence.deskew_filter),
        select_content.OptionsWidget(sequence.select_content_filter),
        page_layout.OptionsWidget(sequence.page_layout_filter),
        output.OptionsWidget(sequence.output_filter),
    ]


def _help_text(widget: QtWidgets.QWidget, root: QtWidgets.QWidget) -> str:
    """The tooltip a user sees over a widget: its own or its nearest ancestor's."""
    current: QtWidgets.QWidget | None = widget
    while current is not None:
        if current.toolTip():
            return current.toolTip()
        if current is root:
            return ""
        current = current.parentWidget()
    return ""


def _is_group_header(widget: QtWidgets.QWidget) -> bool:
    return isinstance(widget.parentWidget(), CollapsibleGroupBox) and isinstance(
        widget, QtWidgets.QToolButton
    )


def _is_part_of_another_control(widget: QtWidgets.QWidget) -> bool:
    """The editor of a spin box or the scroll bars of a combo's list are no options."""
    parent = widget.parentWidget()
    return isinstance(widget, QtWidgets.QScrollBar) or isinstance(
        parent, QtWidgets.QAbstractSpinBox | QtWidgets.QComboBox
    )


def test_every_option_in_the_six_stages_explains_itself(qtbot: QtBot):
    missing: list[str] = []
    for root in _option_widgets():
        qtbot.addWidget(root)
        for control in root.findChildren(QtWidgets.QWidget):
            if (
                not isinstance(control, OPTION_CONTROLS)
                or isinstance(control, HelpButton)
                or _is_group_header(control)
                or _is_part_of_another_control(control)
            ):
                continue
            if not _help_text(control, root).strip():
                missing.append(
                    f"{type(root).__module__}: {type(control).__name__} "
                    f"{control.objectName() or '(unnamed)'}"
                )
    assert not missing, "Options without a tooltip:\n" + "\n".join(missing)
