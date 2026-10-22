"""The Output panel's numeric controls take upstream's ranges."""

from pathlib import Path

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.options_widgets.output import OptionsWidget
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.output.binarization import BinarizationOptions
from scantailor.filters.output.filter import Filter
from scantailor.filters.output.params import Params

PAGE = PageId(
    image_id=ImageId(file_path=Path("page.tif")), sub_page=SubPage.SINGLE_PAGE
)


@pytest.fixture
def widget(qtbot: QtBot) -> OptionsWidget:
    options = OptionsWidget(Filter())
    qtbot.addWidget(options)
    options.set_current_page(PAGE)
    return options


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [
        # output/ChangeDpiDialog.cpp:91-100
        ("dpiSpinBox", 72, 1200),
        # OptionsWidget.ui wienerWindowSize
        ("wienerWindowSpinBox", 3, 9999),
        # OptionsWidgetBinarizationSauvola.ui and Wolf.ui windowSize
        ("windowSizeSpinBox", 3, 9999),
    ],
)
def test_an_integer_control_spans_upstreams_range(
    widget: OptionsWidget, name: str, low: int, high: int
) -> None:
    box = widget.findChild(QtWidgets.QSpinBox, name)
    assert box is not None

    assert (box.minimum(), box.maximum()) == (low, high)


@pytest.mark.parametrize(
    "name",
    [
        # Upstream shows one Sauvola coefficient (0.01 to 9.99) for Sauvola,
        # Bradley, EdgePlus, BlurDiv and EdgeDiv, and the Wolf one for Wolf. The
        # port starts at 0 so that a stored 0 stays as it is.
        "sauvolaCoefSpinBox",
        "wolfCoefSpinBox",
        "bradleyCoefSpinBox",
        "edgePlusCoefSpinBox",
        "blurDivCoefSpinBox",
        "edgeDivKepSpinBox",
        "edgeDivKbdSpinBox",
    ],
)
def test_a_coefficient_control_spans_0_to_9_99(
    widget: OptionsWidget, name: str
) -> None:
    box = widget.findChild(QtWidgets.QDoubleSpinBox, name)
    assert box is not None

    assert (box.minimum(), box.maximum()) == pytest.approx((0.0, 9.99))


def test_a_coefficient_above_one_is_stored(widget: OptionsWidget) -> None:
    filter_ = widget._filter
    box = widget.findChild(QtWidgets.QDoubleSpinBox, "sauvolaCoefSpinBox")
    assert box is not None

    box.setValue(2.5)

    assert filter_.get_params(PAGE).binarization.sauvola_coef == pytest.approx(2.5)


def test_a_window_above_a_thousand_is_stored(widget: OptionsWidget) -> None:
    filter_ = widget._filter
    box = widget.findChild(QtWidgets.QSpinBox, "windowSizeSpinBox")
    assert box is not None

    box.setValue(5000)

    assert filter_.get_params(PAGE).binarization.window_size == 5000


def test_a_stored_zero_coefficient_shows_as_zero(qtbot: QtBot) -> None:
    filter_ = Filter()
    filter_.set_params(
        PAGE,
        Params(binarization=BinarizationOptions(sauvola_coef=0.0, wolf_coef=0.0)),
    )
    options = OptionsWidget(filter_)
    qtbot.addWidget(options)
    options.set_current_page(PAGE)

    for name in ("sauvolaCoefSpinBox", "wolfCoefSpinBox"):
        box = options.findChild(QtWidgets.QDoubleSpinBox, name)
        assert box is not None
        assert box.value() == 0.0
    assert filter_.get_params(PAGE).binarization.sauvola_coef == 0.0
