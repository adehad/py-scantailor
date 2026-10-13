"""Picture Shape control of the Output options panel."""

from pathlib import Path

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.options_widgets.output import OptionsWidget
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.filter import Filter
from scantailor.filters.output.params import ApplyGroup, Params
from scantailor.filters.output.picture_shape import PictureShape, PictureShapeOptions

PAGE = PageId(
    image_id=ImageId(file_path=Path("page.tif")), sub_page=SubPage.SINGLE_PAGE
)


class _Panel:
    def __init__(self, qtbot: QtBot, params: Params) -> None:
        self.filter = Filter()
        self.filter.set_params(PAGE, params)
        self.widget = OptionsWidget(self.filter)
        qtbot.addWidget(self.widget)
        self.widget.set_current_page(PAGE)
        self.changed: list[PageId] = []
        self.widget.params_changed.connect(self.changed.append)

    @property
    def params(self) -> Params:
        return self.filter.get_params(PAGE)

    def find[T: QtWidgets.QWidget](self, kind: type[T], name: str) -> T:
        found = self.widget.findChild(kind, name)
        assert found is not None, name
        return found

    def shown(self, name: str) -> bool:
        return self.find(QtWidgets.QWidget, name).isVisibleTo(self.widget)


def _mixed(**shape: object) -> Params:
    return Params(
        color_mode=ColorMode.MIXED,
        picture_shape=PictureShapeOptions(**shape),  # type: ignore[arg-type]
    )


class TestVisibility:
    @pytest.mark.parametrize(
        ("mode", "shown"),
        [
            (ColorMode.BLACK_AND_WHITE, False),
            (ColorMode.COLOR_GRAYSCALE, False),
            (ColorMode.MIXED, True),
        ],
    )
    def test_the_group_shows_in_mixed_mode_only(
        self, qtbot: QtBot, mode: ColorMode, shown: bool
    ):
        panel = _Panel(qtbot, Params(color_mode=mode))

        assert panel.shown("pictureShapeOptions") is shown

    def test_sensitivity_shows_for_the_rectangular_shape_only(self, qtbot: QtBot):
        free = _Panel(qtbot, _mixed(shape=PictureShape.FREE))
        rectangular = _Panel(qtbot, _mixed(shape=PictureShape.RECTANGULAR))

        assert not free.shown("pictureShapeSensitivityOptions")
        assert rectangular.shown("pictureShapeSensitivityOptions")

    def test_higher_search_sensitivity_hides_when_detection_is_off(self, qtbot: QtBot):
        off = _Panel(qtbot, _mixed(shape=PictureShape.OFF))
        free = _Panel(qtbot, _mixed(shape=PictureShape.FREE))

        assert not off.shown("higherSearchSensitivityCB")
        assert free.shown("higherSearchSensitivityCB")

    def test_the_controls_show_the_stored_options(self, qtbot: QtBot):
        panel = _Panel(
            qtbot,
            _mixed(
                shape=PictureShape.RECTANGULAR,
                sensitivity=35,
                higher_search_sensitivity=True,
            ),
        )

        combo = panel.find(QtWidgets.QComboBox, "pictureShapeSelector")
        assert combo.currentData() == PictureShape.RECTANGULAR
        assert panel.find(QtWidgets.QSpinBox, "pictureShapeSensitivitySB").value() == 35
        assert panel.find(QtWidgets.QCheckBox, "higherSearchSensitivityCB").isChecked()


class TestEditing:
    def test_the_selector_offers_the_upstream_choices_in_order(self, qtbot: QtBot):
        panel = _Panel(qtbot, _mixed())
        combo = panel.find(QtWidgets.QComboBox, "pictureShapeSelector")

        assert [combo.itemText(i) for i in range(combo.count())] == [
            "Off",
            "Free",
            "Rectangular",
        ]

    def test_choosing_a_shape_stores_it_and_reloads_at_once(self, qtbot: QtBot):
        panel = _Panel(qtbot, _mixed(shape=PictureShape.FREE))
        combo = panel.find(QtWidgets.QComboBox, "pictureShapeSelector")

        combo.setCurrentIndex(combo.findData(PictureShape.RECTANGULAR))

        assert panel.params.picture_shape.shape is PictureShape.RECTANGULAR
        assert panel.changed == [PAGE]
        assert panel.shown("pictureShapeSensitivityOptions")

    def test_sensitivity_is_stored_at_once_and_reloads_after_the_delay(
        self, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(OptionsWidget, "RELOAD_DELAY_MS", 20)
        panel = _Panel(qtbot, _mixed(shape=PictureShape.RECTANGULAR))
        spin = panel.find(QtWidgets.QSpinBox, "pictureShapeSensitivitySB")

        spin.setValue(40)
        spin.setValue(45)

        assert panel.params.picture_shape.sensitivity == 45
        assert panel.changed == []
        qtbot.waitUntil(lambda: panel.changed == [PAGE], timeout=2000)

    def test_higher_search_sensitivity_toggle_reloads_at_once(self, qtbot: QtBot):
        panel = _Panel(qtbot, _mixed(shape=PictureShape.FREE))

        panel.find(QtWidgets.QCheckBox, "higherSearchSensitivityCB").click()

        assert panel.params.picture_shape.higher_search_sensitivity
        assert panel.changed == [PAGE]

    def test_the_stored_shape_survives_a_mode_round_trip(self, qtbot: QtBot):
        panel = _Panel(qtbot, _mixed(shape=PictureShape.RECTANGULAR, sensitivity=60))

        panel.find(QtWidgets.QPushButton, "bwBtn").click()
        panel.find(QtWidgets.QPushButton, "mixedBtn").click()

        assert panel.params.picture_shape.shape is PictureShape.RECTANGULAR
        assert panel.params.picture_shape.sensitivity == 60


def test_the_shape_is_part_of_the_colour_mode_apply_group():
    source = _mixed(shape=PictureShape.RECTANGULAR, sensitivity=20)

    merged = Params().with_group_from(source, ApplyGroup.COLOR_MODE)

    assert merged.picture_shape == source.picture_shape
