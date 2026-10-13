"""The Deskew "Top page edge (book scans)" check box."""

import pathlib

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.options_widgets.deskew import OptionsWidget
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.deskew.filter import Filter
from scantailor.filters.deskew.params import Params

PAGE = PageId(
    image_id=ImageId(file_path=pathlib.Path("/page.tiff")),
    sub_page=SubPage.SINGLE_PAGE,
)


@pytest.fixture
def filter_() -> Filter:
    return Filter()


@pytest.fixture
def widget(qtbot: QtBot, filter_: Filter) -> OptionsWidget:
    options = OptionsWidget(filter_)
    qtbot.addWidget(options)
    options.set_current_page(PAGE)
    return options


def _top_edge(widget: OptionsWidget) -> QtWidgets.QCheckBox:
    box = widget.findChild(QtWidgets.QCheckBox, "topEdgeCheckBox")
    assert box is not None
    return box


class TestTopEdgeCheckBox:
    def test_uses_the_upstream_text_and_starts_unchecked(self, widget):
        box = _top_edge(widget)

        assert box.text() == "Top page edge (book scans)"
        assert not box.isChecked()

    def test_checking_it_switches_detection_to_the_top_edge(self, widget, filter_):
        _top_edge(widget).setChecked(True)

        assert filter_.content_based is False

    def test_unchecking_it_returns_to_content_detection(self, widget, filter_):
        _top_edge(widget).setChecked(True)
        _top_edge(widget).setChecked(False)

        assert filter_.content_based is True

    def test_it_shows_a_flag_set_elsewhere_when_refreshed(self, widget, filter_):
        filter_.content_based = False

        widget.refresh()

        assert _top_edge(widget).isChecked()

    def test_an_automatic_page_is_detected_again_and_announced(
        self, widget, filter_, qtbot
    ):
        filter_.set_params(PAGE, Params(deskew_angle_deg=2.0, redetect_rotation=False))

        with qtbot.waitSignal(widget.angle_changed):
            _top_edge(widget).setChecked(True)

        assert filter_.get_params(PAGE).redetect_rotation is True

    def test_a_manual_page_keeps_its_angle_and_is_not_announced(
        self, widget, filter_, qtbot
    ):
        filter_.set_manual_angle(PAGE, 3.0)

        with qtbot.assertNotEmitted(widget.angle_changed):
            _top_edge(widget).setChecked(True)

        assert filter_.get_deskew_angle(PAGE) == 3.0
