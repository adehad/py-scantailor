"""Tests for DefaultParams."""

import json

from scantailor.core import (
    DefaultParams,
    DeskewDefaults,
    Dpi,
    FixOrientationDefaults,
    Margins,
    OrthogonalRotation,
    OutputDefaults,
    PageLayoutDefaults,
    PageSplitDefaults,
    SelectContentDefaults,
    Units,
)
from scantailor.filters.deskew.params import AutoManualMode as DeskewMode
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.detection import (
    ContentDetectionMode,
    PageDetectionMode,
)


class TestFixOrientationDefaults:
    """Tests for FixOrientationDefaults."""

    def test_defaults(self) -> None:
        """Default rotation is 0 degrees."""
        defaults = FixOrientationDefaults()
        assert defaults.image_rotation.degrees == 0

    def test_custom_rotation(self) -> None:
        """Can create with custom rotation."""
        defaults = FixOrientationDefaults(image_rotation=OrthogonalRotation(degrees=90))
        assert defaults.image_rotation.degrees == 90


class TestDeskewDefaults:
    """Tests for DeskewDefaults."""

    def test_defaults(self) -> None:
        """Default values match C++."""
        defaults = DeskewDefaults()
        assert defaults.deskew_angle_deg == 0.0
        assert defaults.mode == DeskewMode.AUTO

    def test_to_params(self) -> None:
        """to_params creates correct filter params."""
        defaults = DeskewDefaults(deskew_angle_deg=5.0, mode=DeskewMode.MANUAL)
        params = defaults.to_params()

        assert params.deskew_angle_deg == 5.0
        assert params.mode == DeskewMode.MANUAL


class TestPageSplitDefaults:
    """Tests for PageSplitDefaults."""

    def test_defaults(self) -> None:
        """Default layout type is AUTO."""
        defaults = PageSplitDefaults()
        assert defaults.layout_type == LayoutType.AUTO_LAYOUT_TYPE

    def test_to_params(self) -> None:
        """to_params creates correct filter params."""
        defaults = PageSplitDefaults(layout_type=LayoutType.TWO_PAGES)
        params = defaults.to_params()

        assert params.layout_type == LayoutType.TWO_PAGES


class TestSelectContentDefaults:
    """Tests for SelectContentDefaults."""

    def test_defaults(self) -> None:
        """Default values match C++ (A4 page size)."""
        defaults = SelectContentDefaults()

        # A4 page size
        assert defaults.page_rect_size_mm.width_mm == 210.0
        assert defaults.page_rect_size_mm.height_mm == 297.0

        assert defaults.content_detection_mode == ContentDetectionMode.AUTO
        assert defaults.page_detection_mode == PageDetectionMode.DISABLED
        assert defaults.fine_tune_corners is False

    def test_to_params(self) -> None:
        """to_params creates correct filter params."""
        defaults = SelectContentDefaults(
            content_detection_mode=ContentDetectionMode.MANUAL,
            fine_tune_corners=True,
        )
        params = defaults.to_params()

        assert params.content_detection_mode == ContentDetectionMode.MANUAL
        assert params.fine_tune_corners is True


class TestPageLayoutDefaults:
    """Tests for PageLayoutDefaults."""

    def test_defaults(self) -> None:
        """Default margins match C++ (10, 5, 10, 5)."""
        defaults = PageLayoutDefaults()

        assert defaults.hard_margins_mm.left == 10.0
        assert defaults.hard_margins_mm.right == 5.0
        assert defaults.hard_margins_mm.top == 10.0
        assert defaults.hard_margins_mm.bottom == 5.0
        assert defaults.auto_margins is False

    def test_to_params(self) -> None:
        """to_params creates correct filter params."""
        margins = Margins(left=20.0, right=20.0, top=15.0, bottom=15.0)
        defaults = PageLayoutDefaults(hard_margins_mm=margins, auto_margins=True)
        params = defaults.to_params()

        assert params.hard_margins_mm.left == 20.0
        assert params.auto_margins is True


class TestOutputDefaults:
    """Tests for OutputDefaults."""

    def test_defaults(self) -> None:
        """Default values match C++ (600 DPI, B&W)."""
        defaults = OutputDefaults()

        assert defaults.dpi.horizontal == 600
        assert defaults.dpi.vertical == 600
        assert defaults.color_mode == ColorMode.BLACK_AND_WHITE
        assert defaults.despeckle_level == DespeckleLevel.NORMAL
        assert defaults.black_on_white is True

    def test_to_params(self) -> None:
        """to_params creates correct filter params."""
        defaults = OutputDefaults(
            dpi=Dpi.uniform(300),
            color_mode=ColorMode.MIXED,
            despeckle_level=DespeckleLevel.AGGRESSIVE,
        )
        params = defaults.to_params()

        assert params.output_dpi.horizontal == 300
        assert params.color_mode == ColorMode.MIXED
        assert params.despeckle_level == DespeckleLevel.AGGRESSIVE


class TestDefaultParams:
    """Tests for DefaultParams aggregate class."""

    def test_defaults(self) -> None:
        """All sub-defaults are created with their defaults."""
        defaults = DefaultParams()

        # Check each sub-section exists with default values
        assert defaults.fix_orientation.image_rotation.degrees == 0
        assert defaults.deskew.deskew_angle_deg == 0.0
        assert defaults.page_split.layout_type == LayoutType.AUTO_LAYOUT_TYPE
        assert (
            defaults.select_content.content_detection_mode == ContentDetectionMode.AUTO
        )
        assert defaults.page_layout.hard_margins_mm.left == 10.0
        assert defaults.output.dpi.horizontal == 600
        assert defaults.units == Units.MILLIMETERS

    def test_get_deskew_params(self) -> None:
        """get_deskew_params returns filter params."""
        defaults = DefaultParams()
        defaults.deskew = DeskewDefaults(deskew_angle_deg=3.0)

        params = defaults.get_deskew_params()
        assert params.deskew_angle_deg == 3.0

    def test_get_page_split_params(self) -> None:
        """get_page_split_params returns filter params."""
        defaults = DefaultParams()
        defaults.page_split = PageSplitDefaults(
            layout_type=LayoutType.SINGLE_PAGE_UNCUT
        )

        params = defaults.get_page_split_params()
        assert params.layout_type == LayoutType.SINGLE_PAGE_UNCUT

    def test_get_select_content_params(self) -> None:
        """get_select_content_params returns filter params."""
        defaults = DefaultParams()
        params = defaults.get_select_content_params()

        assert params.content_detection_mode == ContentDetectionMode.AUTO

    def test_get_page_layout_params(self) -> None:
        """get_page_layout_params returns filter params."""
        defaults = DefaultParams()
        params = defaults.get_page_layout_params()

        assert params.hard_margins_mm.left == 10.0

    def test_get_output_params(self) -> None:
        """get_output_params returns filter params."""
        defaults = DefaultParams()
        params = defaults.get_output_params()

        assert params.output_dpi.horizontal == 600

    def test_json_serialization(self) -> None:
        """DefaultParams serializes to JSON."""
        defaults = DefaultParams()
        json_str = defaults.model_dump_json()
        data = json.loads(json_str)

        assert "fix_orientation" in data
        assert "deskew" in data
        assert "page_split" in data
        assert "select_content" in data
        assert "page_layout" in data
        assert "output" in data
        assert data["units"] == "mm"

    def test_json_deserialization(self) -> None:
        """DefaultParams deserializes from JSON."""
        json_data = {
            "deskew": {"deskew_angle_deg": 5.0, "mode": "manual"},
            "units": "in",
        }

        defaults = DefaultParams.model_validate(json_data)

        assert defaults.deskew.deskew_angle_deg == 5.0
        assert defaults.deskew.mode == DeskewMode.MANUAL
        assert defaults.units == Units.INCHES
        # Other sections use defaults
        assert defaults.output.dpi.horizontal == 600

    def test_custom_initialization(self) -> None:
        """Can create with custom sub-defaults."""
        defaults = DefaultParams(
            deskew=DeskewDefaults(deskew_angle_deg=2.5),
            output=OutputDefaults(dpi=Dpi.uniform(1200)),
            units=Units.INCHES,
        )

        assert defaults.deskew.deskew_angle_deg == 2.5
        assert defaults.output.dpi.horizontal == 1200
        assert defaults.units == Units.INCHES
        # Others are still defaults
        assert defaults.page_split.layout_type == LayoutType.AUTO_LAYOUT_TYPE
