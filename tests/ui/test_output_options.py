"""Output options widget: each control writes its field of the page's Params."""

from pathlib import Path

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.apply_to import OUTPUT_GROUP_TITLES
from scantailor.app.ui.icons import is_palette_icon
from scantailor.app.ui.options_widgets.output import OptionsWidget
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.output.binarization import BinarizationMethod
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import FillingColor
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.dewarping_options import DewarpingMode
from scantailor.filters.output.filter import Filter
from scantailor.filters.output.params import ApplyGroup, DistortionModelSpec, Params

PAGE = PageId(
    image_id=ImageId(file_path=Path("page.tif")), sub_page=SubPage.SINGLE_PAGE
)


class _Widget:
    """An options widget on a fresh filter, with the changed signal recorded."""

    def __init__(self, qtbot: QtBot, params: Params | None = None) -> None:
        self.filter = Filter()
        if params is not None:
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

    def enabled(self, name: str) -> bool:
        return self.find(QtWidgets.QWidget, name).isEnabled()

    def click(self, kind: type[QtWidgets.QAbstractButton], name: str) -> None:
        self.find(kind, name).click()

    def set_check(self, name: str, on: bool) -> None:
        box = self.find(QtWidgets.QCheckBox, name)
        assert box.isChecked() != on
        box.click()


def _with_mode(mode: ColorMode) -> Params:
    return Params(color_mode=mode)


@pytest.fixture
def bw(qtbot: QtBot) -> _Widget:
    return _Widget(qtbot, _with_mode(ColorMode.BLACK_AND_WHITE))


@pytest.fixture
def mixed(qtbot: QtBot) -> _Widget:
    return _Widget(qtbot, _with_mode(ColorMode.MIXED))


@pytest.fixture
def gray(qtbot: QtBot) -> _Widget:
    return _Widget(qtbot, _with_mode(ColorMode.COLOR_GRAYSCALE))


class TestControlsWriteParams:
    def test_output_dpi_spinbox_sets_uniform_dpi(self, bw: _Widget):
        bw.find(QtWidgets.QSpinBox, "dpiSpinBox").setValue(300)

        assert bw.params.output_dpi.horizontal == 300
        assert bw.params.output_dpi.vertical == 300
        assert bw.changed == [PAGE]

    @pytest.mark.parametrize(
        ("button", "mode"),
        [
            ("grayBtn", ColorMode.COLOR_GRAYSCALE),
            ("mixedBtn", ColorMode.MIXED),
            ("bwBtn", ColorMode.BLACK_AND_WHITE),
        ],
    )
    def test_mode_buttons_set_color_mode(
        self, mixed: _Widget, button: str, mode: ColorMode
    ):
        mixed.click(QtWidgets.QPushButton, "grayBtn")
        mixed.changed.clear()

        mixed.click(QtWidgets.QPushButton, button)

        assert mixed.params.color_mode == mode

    def test_mode_button_reports_one_change(self, bw: _Widget):
        bw.click(QtWidgets.QPushButton, "mixedBtn")

        assert bw.changed == [PAGE]

    def test_white_margins_checkbox_sets_black_on_white(self, bw: _Widget):
        bw.set_check("blackOnWhiteCB", False)

        assert bw.params.black_on_white is False

    @pytest.mark.parametrize(
        ("name", "field"),
        [("fillMarginsCB", "fill_margins"), ("fillOffcutCB", "fill_offcut")],
    )
    def test_fill_checkboxes_clear_their_flag(
        self, gray: _Widget, name: str, field: str
    ):
        gray.set_check(name, False)

        assert getattr(gray.params.color_common, field) is False
        assert gray.changed == [PAGE]

    def test_fill_outside_page_box_is_off_until_chosen(self, gray: _Widget):
        assert not gray.find(QtWidgets.QCheckBox, "fillOutsidePageBoxCB").isChecked()
        assert gray.params.color_common.fill_outside_page_box is False

    def test_fill_outside_page_box_checkbox_sets_its_flag_and_clears_offcut(
        self, gray: _Widget
    ):
        gray.set_check("fillOutsidePageBoxCB", True)

        common = gray.params.color_common
        assert common.fill_outside_page_box is True
        assert common.fill_offcut is False
        assert not gray.find(QtWidgets.QCheckBox, "fillOffcutCB").isChecked()
        assert gray.changed == [PAGE]

    def test_fill_offcut_clears_fill_outside_page_box(self, gray: _Widget):
        gray.set_check("fillOutsidePageBoxCB", True)

        gray.set_check("fillOffcutCB", True)

        common = gray.params.color_common
        assert common.fill_offcut is True
        assert common.fill_outside_page_box is False
        assert not gray.find(QtWidgets.QCheckBox, "fillOutsidePageBoxCB").isChecked()

    def test_unchecking_fill_outside_page_box_leaves_offcut_off(self, gray: _Widget):
        gray.set_check("fillOutsidePageBoxCB", True)

        gray.set_check("fillOutsidePageBoxCB", False)

        common = gray.params.color_common
        assert common.fill_outside_page_box is False
        assert common.fill_offcut is False

    def test_fill_outside_page_box_explains_itself_and_shows_its_icon(
        self, gray: _Widget
    ):
        box = gray.find(QtWidgets.QCheckBox, "fillOutsidePageBoxCB")

        assert box.text() == "Fill outside page box"
        assert box.toolTip().startswith("Fill the full output page rectangle")
        assert is_palette_icon(box.icon())
        assert box.icon().name() == "scantailor:squares-exclude"

    def test_filling_color_combo_sets_filling_color(self, gray: _Widget):
        combo = gray.find(QtWidgets.QComboBox, "fillingColorBox")

        combo.setCurrentIndex(combo.findData(FillingColor.BLACK))

        assert gray.params.color_common.filling_color == FillingColor.BLACK

    def test_wiener_controls_set_coefficient_and_window(self, gray: _Widget):
        gray.find(QtWidgets.QDoubleSpinBox, "wienerCoefSpinBox").setValue(0.5)
        gray.find(QtWidgets.QSpinBox, "wienerWindowSpinBox").setValue(9)
        gray.widget.flush_pending_reload()

        assert gray.params.color_common.wiener_coef == pytest.approx(0.5)
        assert gray.params.color_common.wiener_window_size == 9
        assert gray.changed == [PAGE]

    def test_savitzky_golay_and_morphological_toggles(self, bw: _Widget):
        bw.set_check("savitzkyGolaySmoothingCB", False)
        bw.set_check("morphologicalSmoothingCB", False)

        assert bw.params.binarization.savitzky_golay_smoothing is False
        assert bw.params.binarization.morphological_smoothing is False

    def test_equalize_illumination_sets_binarization_flag(self, bw: _Widget):
        bw.set_check("equalizeIlluminationCB", False)

        assert bw.params.binarization.normalize_illumination is False

    def test_equalize_color_sets_color_flag_in_grayscale(self, gray: _Widget):
        gray.set_check("equalizeIlluminationColorCB", True)

        assert gray.params.color_common.normalize_illumination is True

    def test_unchecking_black_and_white_equalize_clears_the_color_one_in_mixed(
        self, qtbot: QtBot
    ):
        params = _with_mode(ColorMode.MIXED)
        params = params.with_color_common(
            params.color_common.model_copy(update={"normalize_illumination": True})
        )
        params = params.with_binarization(
            params.binarization.model_copy(update={"normalize_illumination": True})
        )
        mixed = _Widget(qtbot, params)

        mixed.set_check("equalizeIlluminationCB", False)

        assert mixed.params.color_common.normalize_illumination is False
        color_box = mixed.find(QtWidgets.QCheckBox, "equalizeIlluminationColorCB")
        assert color_box.isChecked() is False

    def test_showing_mixed_without_black_and_white_equalize_clears_the_color_one(
        self, qtbot: QtBot
    ):
        params = _with_mode(ColorMode.MIXED)
        params = params.with_color_common(
            params.color_common.model_copy(update={"normalize_illumination": True})
        )
        params = params.with_binarization(
            params.binarization.model_copy(update={"normalize_illumination": False})
        )

        mixed = _Widget(qtbot, params)

        assert mixed.params.color_common.normalize_illumination is False
        color_box = mixed.find(QtWidgets.QCheckBox, "equalizeIlluminationColorCB")
        assert color_box.isChecked() is False

    @pytest.mark.parametrize(
        ("step", "level"),
        [
            (0, DespeckleLevel.OFF),
            (1, DespeckleLevel.CAUTIOUS),
            (2, DespeckleLevel.NORMAL),
            (3, DespeckleLevel.AGGRESSIVE),
        ],
    )
    def test_despeckle_slider_step_sets_level(
        self, bw: _Widget, step: int, level: DespeckleLevel
    ):
        slider = bw.find(QtWidgets.QSlider, "despeckleSlider")
        slider.setValue(0 if step != 0 else 3)
        bw.changed.clear()

        slider.setValue(step)

        assert bw.params.despeckle_level == level
        assert bw.changed == [PAGE]

    def test_despeckle_slider_drag_renders_once_on_release(self, bw: _Widget):
        slider = bw.find(QtWidgets.QSlider, "despeckleSlider")
        slider.setValue(0)
        bw.changed.clear()

        slider.setSliderDown(True)
        slider.setValue(1)
        slider.setValue(3)
        assert bw.changed == []

        slider.setSliderDown(False)

        assert bw.params.despeckle_level == DespeckleLevel.AGGRESSIVE
        assert bw.changed == [PAGE]

    def test_despeckle_middle_labels_sit_under_their_slider_stops(self, bw: _Widget):
        bw.widget.resize(400, 900)
        bw.widget.show()
        slider = bw.find(QtWidgets.QSlider, "despeckleSlider")
        style = slider.style()
        handle = style.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_SliderLength)

        for step, name in ((1, "Cautious"), (2, "Normal")):
            label = bw.find(QtWidgets.QLabel, f"despeckle{name}Label")
            centre = label.mapTo(bw.widget, label.rect().center()).x()
            stop = (
                style.sliderPositionFromValue(0, 3, step, slider.width() - handle)
                + handle // 2
                + slider.mapTo(bw.widget, slider.rect().topLeft()).x()
            )
            assert abs(centre - stop) <= 3, name

    def test_despeckle_slider_labels_use_the_cpp_strings(self, bw: _Widget):
        labels = [
            bw.find(QtWidgets.QLabel, f"despeckle{name}Label").text()
            for name in ("Off", "Cautious", "Normal", "Aggressive")
        ]

        assert labels == ["Off", "Cautious", "Normal", "Aggressive"]
        slider = bw.find(QtWidgets.QSlider, "despeckleSlider")
        assert (slider.minimum(), slider.maximum()) == (0, 3)

    def test_despeckle_apply_button_emits_the_despeckle_group(
        self, bw: _Widget, qtbot: QtBot
    ):
        with qtbot.waitSignal(bw.widget.apply_group_requested) as blocker:
            bw.click(QtWidgets.QPushButton, "apply_despeckle_btn")

        assert blocker.args == [ApplyGroup.DESPECKLE]

    def test_post_deskew_checkbox_needs_dewarping_to_be_enabled(self, bw: _Widget):
        assert not bw.enabled("postDeskewCB")

        bw.click(QtWidgets.QPushButton, "dewarpAutoBtn")

        assert bw.enabled("postDeskewCB")
        bw.set_check("postDeskewCB", False)
        assert bw.params.dewarping.post_deskew is False


class TestSplitting:
    def test_split_output_checkbox_sets_flag(self, mixed: _Widget):
        mixed.set_check("splittingCB", True)

        assert mixed.params.splitting.split_output is True

    def test_foreground_type_radios_set_color_foreground(self, mixed: _Widget):
        mixed.set_check("splittingCB", True)

        mixed.click(QtWidgets.QRadioButton, "colorForegroundRB")
        assert mixed.params.splitting.color_foreground is True

        mixed.click(QtWidgets.QRadioButton, "bwForegroundRB")
        assert mixed.params.splitting.color_foreground is False

    def test_original_background_checkbox_sets_flag(self, mixed: _Widget):
        mixed.set_check("splittingCB", True)

        mixed.set_check("originalBackgroundCB", True)

        assert mixed.params.splitting.original_background is True

    def test_dependent_controls_follow_split_output_and_foreground_type(
        self, mixed: _Widget
    ):
        assert not mixed.enabled("bwForegroundRB")
        assert not mixed.enabled("colorForegroundRB")
        assert not mixed.enabled("originalBackgroundCB")

        mixed.set_check("splittingCB", True)
        assert mixed.enabled("bwForegroundRB")
        assert mixed.enabled("originalBackgroundCB")

        mixed.click(QtWidgets.QRadioButton, "colorForegroundRB")
        assert not mixed.enabled("originalBackgroundCB")


class TestBinarization:
    def test_method_combo_sets_method(self, bw: _Widget):
        combo = bw.find(QtWidgets.QComboBox, "methodCombo")

        combo.setCurrentIndex(combo.findData(BinarizationMethod.WOLF))

        assert bw.params.binarization.method == BinarizationMethod.WOLF

    def test_threshold_slider_sets_adjustment(self, bw: _Widget):
        bw.find(QtWidgets.QSlider, "thresholdSlider").setValue(-25)

        assert bw.params.binarization.threshold_adjustment == -25

    def test_threshold_slider_commits_once_when_released_after_a_drag(
        self, bw: _Widget
    ):
        slider = bw.find(QtWidgets.QSlider, "thresholdSlider")

        slider.setSliderDown(True)
        slider.setValue(30)
        slider.setValue(40)
        assert bw.changed == []
        assert bw.params.binarization.threshold_adjustment == 0

        slider.setSliderDown(False)

        assert bw.params.binarization.threshold_adjustment == 40
        assert bw.changed == [PAGE]

    def test_window_size_sets_window_size(self, bw: _Widget):
        bw.find(QtWidgets.QSpinBox, "windowSizeSpinBox").setValue(120)

        assert bw.params.binarization.window_size == 120

    @pytest.mark.parametrize(
        ("method", "name", "value", "field"),
        [
            (BinarizationMethod.SAUVOLA, "sauvolaCoefSpinBox", 0.5, "sauvola_coef"),
            (BinarizationMethod.WOLF, "wolfCoefSpinBox", 0.4, "wolf_coef"),
            (BinarizationMethod.BRADLEY, "bradleyCoefSpinBox", 0.2, "bradley_coef"),
            (BinarizationMethod.EDGEPLUS, "edgePlusCoefSpinBox", 0.6, "edge_div_kep"),
            (BinarizationMethod.BLURDIV, "blurDivCoefSpinBox", 0.4, "edge_div_kbd"),
            (BinarizationMethod.EDGEDIV, "edgeDivKepSpinBox", 0.7, "edge_div_kep"),
            (BinarizationMethod.EDGEDIV, "edgeDivKbdSpinBox", 0.3, "edge_div_kbd"),
        ],
    )
    def test_method_coefficients_write_their_own_field(
        self,
        qtbot: QtBot,
        method: BinarizationMethod,
        name: str,
        value: float,
        field: str,
    ):
        params = Params(binarization=Params().binarization.with_method(method))
        widget = _Widget(qtbot, params)

        widget.find(QtWidgets.QDoubleSpinBox, name).setValue(value)

        assert getattr(widget.params.binarization, field) == pytest.approx(value)

    def test_combo_offers_the_methods_in_upstream_order(self, bw: _Widget):
        combo = bw.find(QtWidgets.QComboBox, "methodCombo")

        assert [combo.itemText(i) for i in range(combo.count())] == [
            "Otsu",
            "Sauvola",
            "Wolf",
            "Fox",
            "Window",
            "Bradley",
            "Grad",
            "EdgePlus",
            "BlurDiv",
            "EdgeDiv",
        ]

    @pytest.mark.parametrize(
        "method",
        [BinarizationMethod.FOX, BinarizationMethod.WINDOW, BinarizationMethod.GRAD],
    )
    def test_wolf_family_methods_edit_the_wolf_options(
        self, qtbot: QtBot, method: BinarizationMethod
    ):
        widget = _Widget(
            qtbot, Params(binarization=Params().binarization.with_method(method))
        )

        widget.find(QtWidgets.QDoubleSpinBox, "wolfCoefSpinBox").setValue(0.45)
        widget.find(QtWidgets.QSpinBox, "wolfLowerBoundSpinBox").setValue(12)
        widget.find(QtWidgets.QSpinBox, "wolfUpperBoundSpinBox").setValue(240)

        binarization = widget.params.binarization
        assert binarization.method is method
        assert binarization.wolf_coef == pytest.approx(0.45)
        assert (binarization.wolf_lower_bound, binarization.wolf_upper_bound) == (
            12,
            240,
        )
        assert widget.shown("wolfCoefSpinBox")
        assert widget.shown("windowSizeSpinBox")

    @pytest.mark.parametrize("method", ["Fox", "Window", "Grad"])
    def test_choosing_a_new_method_in_the_combo_stores_it(
        self, bw: _Widget, method: str
    ):
        combo = bw.find(QtWidgets.QComboBox, "methodCombo")

        combo.setCurrentIndex(combo.findText(method))

        assert bw.params.binarization.method is BinarizationMethod(method.lower())

    @pytest.mark.parametrize(
        ("name", "field"),
        [
            ("wolfLowerBoundSpinBox", "wolf_lower_bound"),
            ("wolfUpperBoundSpinBox", "wolf_upper_bound"),
        ],
    )
    def test_wolf_bounds_write_their_own_field(
        self, qtbot: QtBot, name: str, field: str
    ):
        params = Params(
            binarization=Params().binarization.with_method(BinarizationMethod.WOLF)
        )
        widget = _Widget(qtbot, params)

        widget.find(QtWidgets.QSpinBox, name).setValue(100)

        assert getattr(widget.params.binarization, field) == 100

    @pytest.mark.parametrize(
        ("method", "page"),
        [
            (BinarizationMethod.OTSU, 0),
            (BinarizationMethod.SAUVOLA, 1),
            (BinarizationMethod.WOLF, 2),
            (BinarizationMethod.FOX, 2),
            (BinarizationMethod.WINDOW, 2),
            (BinarizationMethod.BRADLEY, 3),
            (BinarizationMethod.GRAD, 2),
            (BinarizationMethod.EDGEPLUS, 4),
            (BinarizationMethod.BLURDIV, 5),
            (BinarizationMethod.EDGEDIV, 6),
        ],
    )
    def test_each_method_shows_its_own_options_page(
        self, qtbot: QtBot, method: BinarizationMethod, page: int
    ):
        widget = _Widget(
            qtbot, Params(binarization=Params().binarization.with_method(method))
        )

        stack = widget.find(QtWidgets.QStackedWidget, "methodOptions")
        assert stack.currentIndex() == page
        assert widget.shown("windowSizeSpinBox") == (method != BinarizationMethod.OTSU)


class TestSegmentationAndPosterization:
    def test_segmentation_controls_write_segmenter_options(self, bw: _Widget):
        bw.set_check("colorSegmentationCB", True)
        bw.find(QtWidgets.QSpinBox, "reduceNoiseSB").setValue(3)
        bw.find(QtWidgets.QSpinBox, "redAdjustmentSB").setValue(10)
        bw.find(QtWidgets.QSpinBox, "greenAdjustmentSB").setValue(-20)
        bw.find(QtWidgets.QSpinBox, "blueAdjustmentSB").setValue(30)

        segmenter = bw.params.binarization.color_segmenter
        assert segmenter.enabled is True
        assert segmenter.noise_reduction == 3
        assert segmenter.red_threshold_adjustment == 10
        assert segmenter.green_threshold_adjustment == -20
        assert segmenter.blue_threshold_adjustment == 30

    def test_segmenter_options_are_enabled_with_segmentation(self, bw: _Widget):
        assert not bw.enabled("reduceNoiseSB")

        bw.set_check("colorSegmentationCB", True)

        assert bw.enabled("reduceNoiseSB")

    def test_posterization_controls_write_posterization_options(self, gray: _Widget):
        gray.set_check("posterizeCB", True)
        gray.find(QtWidgets.QSpinBox, "posterizeLevelSB").setValue(8)
        gray.set_check("posterizeNormalizationCB", True)
        gray.set_check("posterizeForceBwCB", False)

        poster = gray.params.color_common.posterization
        assert poster.enabled is True
        assert poster.level == 8
        assert poster.normalization_enabled is True
        assert poster.force_black_and_white is False

    def test_posterization_needs_segmentation_when_binarizing(self, bw: _Widget):
        assert not bw.enabled("posterizeCB")

        bw.set_check("colorSegmentationCB", True)

        assert bw.enabled("posterizeCB")
        assert not bw.enabled("posterizeLevelSB")
        bw.set_check("posterizeCB", True)
        assert bw.enabled("posterizeLevelSB")

    def test_posterization_is_available_without_binarization(self, gray: _Widget):
        assert gray.enabled("posterizeCB")
        assert not gray.enabled("posterizeLevelSB")


class TestVisibilityPerMode:
    """Mirrors OptionsWidget::updateColorsDisplay of the C++ filter."""

    @pytest.mark.parametrize(
        ("name", "bw_shown", "gray_shown", "mixed_shown"),
        [
            ("fillMarginsCB", True, True, True),
            ("fillOffcutCB", True, True, True),
            ("fillOutsidePageBoxCB", True, True, True),
            ("equalizeIlluminationCB", True, False, True),
            ("equalizeIlluminationColorCB", False, True, True),
            ("savitzkyGolaySmoothingCB", True, False, True),
            ("morphologicalSmoothingCB", True, False, True),
            ("binarizationGroup", True, False, True),
            ("segmentationGroup", True, False, True),
            ("despeckleGroup", True, False, True),
            ("fillingGroup", False, True, True),
            ("splittingGroup", False, False, True),
        ],
    )
    def test_controls_shown_per_color_mode(
        self,
        qtbot: QtBot,
        name: str,
        bw_shown: bool,
        gray_shown: bool,
        mixed_shown: bool,
    ):
        for mode, expected in (
            (ColorMode.BLACK_AND_WHITE, bw_shown),
            (ColorMode.COLOR_GRAYSCALE, gray_shown),
            (ColorMode.MIXED, mixed_shown),
        ):
            widget = _Widget(qtbot, _with_mode(mode))

            assert widget.shown(name) is expected, (name, mode)

    def test_switching_mode_updates_visibility(self, bw: _Widget):
        assert not bw.shown("splittingGroup")

        bw.click(QtWidgets.QPushButton, "mixedBtn")

        assert bw.shown("splittingGroup")

    def test_equalize_color_is_enabled_in_mixed_only_with_the_black_and_white_one(
        self, mixed: _Widget
    ):
        assert mixed.enabled("equalizeIlluminationColorCB")

        mixed.set_check("equalizeIlluminationCB", False)

        assert not mixed.enabled("equalizeIlluminationColorCB")

    def test_dewarping_tab_swaps_mode_panel_for_depth_perception(self, mixed: _Widget):
        assert mixed.shown("modeGroup")
        assert mixed.shown("despeckleGroup")
        assert not mixed.shown("depthPerceptionGroup")

        mixed.widget.set_dewarping_tab_active(True)

        assert not mixed.shown("modeGroup")
        assert not mixed.shown("despeckleGroup")
        assert mixed.shown("depthPerceptionGroup")

        mixed.widget.set_dewarping_tab_active(False)

        assert mixed.shown("modeGroup")
        assert not mixed.shown("depthPerceptionGroup")


class TestDewarping:
    @pytest.mark.parametrize(
        ("button", "mode"),
        [
            ("dewarpAutoBtn", DewarpingMode.AUTO),
            ("dewarpManualBtn", DewarpingMode.MANUAL),
            ("dewarpMarginalBtn", DewarpingMode.MARGINAL),
            ("dewarpOffBtn", DewarpingMode.OFF),
        ],
    )
    def test_mode_buttons_set_dewarping_mode(
        self, bw: _Widget, button: str, mode: DewarpingMode
    ):
        bw.click(QtWidgets.QPushButton, "dewarpAutoBtn")
        bw.changed.clear()

        bw.click(QtWidgets.QPushButton, button)

        assert bw.params.dewarping.mode == mode

    def test_depth_perception_slider_sets_depth_perception(self, bw: _Widget):
        bw.widget.set_dewarping_tab_active(True)

        bw.find(QtWidgets.QSlider, "depthPerceptionSlider").setValue(25)

        assert bw.params.depth_perception == pytest.approx(2.5)
        assert bw.changed == [PAGE]

    def test_controls_show_stored_depth_perception(self, qtbot: QtBot):
        widget = _Widget(qtbot, Params(depth_perception=1.4))

        assert widget.find(QtWidgets.QSlider, "depthPerceptionSlider").value() == 14

    def test_storing_a_distortion_model_switches_automatic_dewarping_to_manual(
        self, bw: _Widget
    ):
        bw.click(QtWidgets.QPushButton, "dewarpAutoBtn")
        bw.changed.clear()
        model = DistortionModelSpec(
            top=((0.0, 0.0), (100.0, 5.0)), bottom=((0.0, 200.0), (100.0, 200.0))
        )

        bw.widget.set_distortion_model(model)

        assert bw.params.distortion_model == model
        assert bw.params.dewarping.mode == DewarpingMode.MANUAL
        assert bw.find(QtWidgets.QPushButton, "dewarpManualBtn").isChecked()
        assert bw.changed == [PAGE]

    def test_storing_a_distortion_model_keeps_dewarping_off_when_off(self, bw: _Widget):
        model = DistortionModelSpec(
            top=((0.0, 0.0), (100.0, 0.0)), bottom=((0.0, 200.0), (100.0, 200.0))
        )

        bw.widget.set_distortion_model(model)

        assert bw.params.dewarping.mode == DewarpingMode.OFF
        assert bw.params.distortion_model == model


class TestPageState:
    def test_controls_show_the_stored_values_of_the_page(self, qtbot: QtBot):
        params = Params(
            color_mode=ColorMode.MIXED, despeckle_level=DespeckleLevel.NORMAL
        )
        params = params.with_splitting(
            params.splitting.model_copy(update={"split_output": True})
        )

        widget = _Widget(qtbot, params)

        assert widget.find(QtWidgets.QPushButton, "mixedBtn").isChecked()
        assert widget.find(QtWidgets.QCheckBox, "splittingCB").isChecked()
        assert widget.find(QtWidgets.QSlider, "despeckleSlider").value() == 2

    def test_loading_a_page_does_not_report_a_change(self, qtbot: QtBot):
        widget = _Widget(qtbot, _with_mode(ColorMode.MIXED))

        assert widget.changed == []

    def test_controls_do_nothing_without_a_page(self, qtbot: QtBot):
        widget = OptionsWidget(Filter())
        qtbot.addWidget(widget)
        seen: list[PageId] = []
        widget.params_changed.connect(seen.append)

        spinbox = widget.findChild(QtWidgets.QSpinBox, "dpiSpinBox")
        assert spinbox is not None
        spinbox.setValue(300)

        assert seen == []


@pytest.mark.parametrize("group", list(ApplyGroup))
def test_each_option_group_requests_its_own_apply_to_dialog(
    qtbot: QtBot, group: ApplyGroup
):
    widget = OptionsWidget(Filter())
    qtbot.addWidget(widget)
    button = widget.findChild(QtWidgets.QPushButton, f"apply_{group.value}_btn")
    assert button is not None

    with qtbot.waitSignal(widget.apply_group_requested) as blocker:
        button.click()

    assert blocker.args == [group]


def test_every_apply_group_has_a_dialog_title():
    assert set(OUTPUT_GROUP_TITLES) == set(ApplyGroup)
    assert all(title.strip() for title in OUTPUT_GROUP_TITLES.values())


class TestBinarizationGroupSize:
    @staticmethod
    def _group_height(qtbot: QtBot, method: BinarizationMethod) -> int:
        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=Params().binarization.with_method(method),
        )
        widget = _Widget(qtbot, params)
        widget.widget.show()
        qtbot.wait_exposed(widget.widget)
        return widget.find(QtWidgets.QWidget, "binarizationGroup").height()

    def test_group_is_shorter_with_otsu_than_with_sauvola(self, qtbot: QtBot):
        otsu = self._group_height(qtbot, BinarizationMethod.OTSU)
        sauvola = self._group_height(qtbot, BinarizationMethod.SAUVOLA)

        assert otsu < sauvola

    def test_group_is_shorter_with_sauvola_than_with_wolf(self, qtbot: QtBot):
        sauvola = self._group_height(qtbot, BinarizationMethod.SAUVOLA)
        wolf = self._group_height(qtbot, BinarizationMethod.WOLF)

        assert sauvola < wolf

    def test_group_shrinks_when_the_method_changes_to_otsu(self, qtbot: QtBot):
        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=Params().binarization.with_method(BinarizationMethod.WOLF),
        )
        widget = _Widget(qtbot, params)
        widget.widget.show()
        qtbot.wait_exposed(widget.widget)
        group = widget.find(QtWidgets.QWidget, "binarizationGroup")
        before = group.sizeHint().height()

        combo = widget.find(QtWidgets.QComboBox, "methodCombo")
        combo.setCurrentIndex(combo.findData(BinarizationMethod.OTSU))

        assert group.sizeHint().height() < before


def test_despeckle_slider_paints_its_tick_marks(qtbot: QtBot, bw: _Widget):
    slider = bw.find(QtWidgets.QSlider, "despeckleSlider")
    assert slider.tickPosition() != QtWidgets.QSlider.TickPosition.NoTicks
    bw.widget.show()
    qtbot.wait_exposed(bw.widget)

    image = slider.grab().toImage()
    background = image.pixelColor(0, image.height() - 1)
    below_groove = [
        image.pixelColor(x, y)
        for y in range(image.height() // 2, image.height())
        for x in range(image.width())
    ]

    assert any(color != background for color in below_groove)
