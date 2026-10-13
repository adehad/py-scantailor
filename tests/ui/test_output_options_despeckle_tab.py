"""The despeckle slider previews a level at once; on its tab the render waits."""

from pathlib import Path

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.options_widgets.output import OptionsWidget
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.filter import Filter
from scantailor.filters.output.params import Params

PAGE = PageId(image_id=ImageId(file_path=Path("a.tif")), sub_page=SubPage.SINGLE_PAGE)


class _Panel:
    def __init__(self, qtbot: QtBot) -> None:
        self.filter = Filter()
        self.filter.set_params(
            PAGE,
            Params(
                color_mode=ColorMode.BLACK_AND_WHITE,
                despeckle_level=DespeckleLevel.CAUTIOUS,
            ),
        )
        self.widget = OptionsWidget(self.filter)
        qtbot.addWidget(self.widget)
        self.widget.set_current_page(PAGE)
        self.changed: list[PageId] = []
        self.levels: list[DespeckleLevel] = []
        self.widget.params_changed.connect(self.changed.append)
        self.widget.despeckle_level_changed.connect(self.levels.append)

    @property
    def slider(self) -> QtWidgets.QSlider:
        found = self.widget.findChild(QtWidgets.QSlider, "despeckleSlider")
        assert found is not None
        return found


def test_choosing_a_level_announces_it_and_renders_at_once_by_default(qtbot: QtBot):
    panel = _Panel(qtbot)

    panel.slider.setValue(3)

    assert panel.levels == [DespeckleLevel.AGGRESSIVE]
    assert panel.changed == [PAGE]
    assert panel.filter.get_params(PAGE).despeckle_level is DespeckleLevel.AGGRESSIVE


def test_on_the_despeckling_tab_the_render_waits_for_the_delay(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(OptionsWidget, "RELOAD_DELAY_MS", 20)
    panel = _Panel(qtbot)
    panel.widget.set_despeckle_tab_active(True)

    panel.slider.setValue(3)

    assert panel.levels == [DespeckleLevel.AGGRESSIVE]
    assert panel.changed == []
    assert panel.filter.get_params(PAGE).despeckle_level is DespeckleLevel.AGGRESSIVE
    qtbot.waitUntil(lambda: panel.changed == [PAGE], timeout=2000)


def test_showing_a_page_does_not_announce_a_level(qtbot: QtBot):
    panel = _Panel(qtbot)

    panel.widget.set_current_page(
        PageId(image_id=ImageId(file_path=Path("b.tif")), sub_page=SubPage.SINGLE_PAGE)
    )

    assert panel.levels == []


def test_a_drag_announces_each_step_but_renders_once_on_release(qtbot: QtBot):
    panel = _Panel(qtbot)
    panel.slider.setSliderDown(True)

    panel.slider.setValue(2)
    panel.slider.setValue(3)

    assert panel.levels == []
    assert panel.changed == []
    panel.slider.setSliderDown(False)
    assert panel.levels == [DespeckleLevel.AGGRESSIVE]
    assert panel.changed == [PAGE]


def test_a_stored_in_between_strength_shows_its_nearest_step_and_names_itself(
    qtbot: QtBot,
):
    panel = _Panel(qtbot)
    panel.filter.set_params(
        PAGE, panel.filter.get_params(PAGE).with_despeckle_factor(2.3)
    )

    panel.widget.set_current_page(PAGE)

    assert panel.slider.value() == 2
    assert "2.3" in panel.slider.toolTip()
    assert panel.filter.get_params(PAGE).despeckle_strength == 2.3


def test_a_strength_on_a_step_has_no_tooltip_note(qtbot: QtBot):
    panel = _Panel(qtbot)

    assert panel.slider.toolTip() == ""


def test_picking_a_step_replaces_a_stored_in_between_strength(qtbot: QtBot):
    panel = _Panel(qtbot)
    panel.filter.set_params(
        PAGE, panel.filter.get_params(PAGE).with_despeckle_factor(2.3)
    )
    panel.widget.set_current_page(PAGE)

    panel.slider.setValue(3)

    stored = panel.filter.get_params(PAGE)
    assert stored.despeckle_strength == 3.0
    assert stored.despeckle_factor is None
