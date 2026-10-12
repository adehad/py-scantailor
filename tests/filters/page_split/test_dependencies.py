"""Null stored dependencies are incompatible, as in the C++ application."""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.filters.page_split.dependencies import Dependencies
from scantailor.filters.page_split.filter import Filter
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.page_split.page_layout import PageLayout, PageLayoutType
from scantailor.filters.page_split.params import AutoManualMode, Params
from scantailor.filters.page_split.xml_import import _read_params

_SIZE = (200, 100)


def make_page_id(name: str = "test.tiff") -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path(f"/{name}")), sub_page=SubPage.SINGLE_PAGE
    )


def blank_image(size: tuple[int, int] = _SIZE) -> np.ndarray:
    width, height = size
    return np.zeros((height, width), dtype=np.uint8)


def known(layout_type: LayoutType, size: tuple[int, int] = _SIZE) -> Dependencies:
    return Dependencies(image_size=size, layout_type=layout_type)


class TestIsCompatibleWith:
    def test_null_dependencies_are_incompatible_with_any_current_ones(self):
        current = known(LayoutType.AUTO_LAYOUT_TYPE)

        assert not Dependencies().is_compatible_with(current, manual_split_line=False)
        assert not Dependencies().is_compatible_with(current, manual_split_line=True)

    def test_null_dependencies_are_incompatible_even_with_matching_layout_type(self):
        current = known(LayoutType.TWO_PAGES)
        stored = Dependencies(layout_type=LayoutType.TWO_PAGES)

        assert not stored.is_compatible_with(current, manual_split_line=True)

    def test_identical_known_dependencies_are_compatible(self):
        deps = known(LayoutType.TWO_PAGES)

        assert deps.is_compatible_with(deps, manual_split_line=False)

    def test_changed_image_size_is_incompatible(self):
        stored = known(LayoutType.TWO_PAGES)
        current = known(LayoutType.TWO_PAGES, size=(300, 100))

        assert not stored.is_compatible_with(current, manual_split_line=True)

    def test_changed_rotation_is_incompatible(self):
        stored = known(LayoutType.TWO_PAGES)
        current = stored.model_copy(update={"rotation_degrees": 90})

        assert not stored.is_compatible_with(current, manual_split_line=True)

    def test_uncut_request_ignores_the_stored_layout_type(self):
        stored = known(LayoutType.TWO_PAGES)
        current = known(LayoutType.SINGLE_PAGE_UNCUT)

        assert stored.is_compatible_with(current, manual_split_line=False)

    def test_two_page_request_needs_a_manual_split_line_to_reuse_other_types(self):
        stored = known(LayoutType.PAGE_PLUS_OFFCUT)
        current = known(LayoutType.TWO_PAGES)

        assert stored.is_compatible_with(current, manual_split_line=True)
        assert not stored.is_compatible_with(current, manual_split_line=False)


class TestProcessingLayoutsWithoutDependencies:
    def test_auto_layout_without_dependencies_is_detected_again(self):
        filter_ = Filter()
        page = make_page_id()
        stale = PageLayout.two_pages(*_SIZE, 10.0)
        filter_.set_params(page, Params(page_layout=stale))

        result = filter_.process(blank_image(), page)

        assert result.dependencies.image_size == _SIZE
        assert result.page_layout != stale
        assert result.is_auto()

    def test_manual_layout_without_dependencies_is_adapted_to_the_image(self):
        filter_ = Filter()
        page = make_page_id()
        manual = PageLayout.two_pages(100, 50, 40.0)
        filter_.set_params(
            page,
            Params(
                layout_type=LayoutType.TWO_PAGES,
                page_layout=manual,
                split_line_mode=AutoManualMode.MANUAL,
            ),
        )

        result = filter_.process(blank_image(), page)

        assert result.layout_type == LayoutType.TWO_PAGES
        assert result.is_manual()
        assert result.page_layout is not None
        assert result.page_layout.outline == (0.0, 0.0, 200.0, 100.0)
        assert result.page_layout.get_split_line_x() == 40.0
        assert result.dependencies == known(LayoutType.TWO_PAGES)

    def test_layout_with_matching_dependencies_is_kept_untouched(self):
        filter_ = Filter()
        page = make_page_id()
        layout = PageLayout.two_pages(*_SIZE, 10.0)
        stored = Params(
            layout_type=LayoutType.TWO_PAGES,
            page_layout=layout,
            split_line_mode=AutoManualMode.MANUAL,
            dependencies=known(LayoutType.TWO_PAGES),
        )
        filter_.set_params(page, stored)

        assert filter_.process(blank_image(), page) == stored


class TestManualSplitSetBeforeProcessing:
    def test_manual_split_records_dependencies_and_survives_processing(self):
        filter_ = Filter()
        page = make_page_id()
        stored = filter_.set_manual_split(page, split_x=30.0, width=200, height=100)

        result = filter_.process(blank_image(), page)

        assert stored.dependencies == known(LayoutType.TWO_PAGES)
        assert result.page_layout == stored.page_layout
        assert result.page_layout is not None
        assert result.page_layout.get_split_line_x() == 30.0

    def test_manual_cutters_record_dependencies_and_survive_processing(self):
        filter_ = Filter()
        page = make_page_id()
        stored = filter_.set_manual_cutters(
            page, left_x=20.0, right_x=180.0, width=200, height=100
        )

        result = filter_.process(blank_image(), page)

        assert stored.dependencies == known(LayoutType.PAGE_PLUS_OFFCUT)
        assert result.page_layout == stored.page_layout

    def test_manual_split_records_the_rotation_it_was_set_under(self):
        filter_ = Filter()
        page = make_page_id()
        filter_.set_manual_split(
            page, split_x=30.0, width=200, height=100, rotation_degrees=90
        )

        result = filter_.process(blank_image(), page, rotation_degrees=90)

        assert result.page_layout is not None
        assert result.page_layout.get_split_line_x() == 30.0
        assert result.dependencies.rotation_degrees == 90

    def test_manual_split_keeps_dependencies_recorded_by_detection(self):
        filter_ = Filter()
        page = make_page_id()
        filter_.process(blank_image(), page, rotation_degrees=180)

        stored = filter_.set_manual_split(page, split_x=30.0, width=200, height=100)

        assert stored.dependencies.rotation_degrees == 180
        assert stored.dependencies.layout_type == LayoutType.TWO_PAGES

    def test_layout_edited_in_the_view_keeps_the_requested_type_and_survives(self):
        filter_ = Filter()
        page = make_page_id()
        filter_.set_layout_type(page, LayoutType.AUTO_LAYOUT_TYPE)
        edited = PageLayout.two_pages(*_SIZE, 70.0)

        stored = filter_.set_manual_layout(page, edited)
        result = filter_.process(blank_image(), page)

        assert stored.layout_type == LayoutType.AUTO_LAYOUT_TYPE
        assert stored.is_manual()
        assert stored.dependencies.image_size == _SIZE
        assert result.page_layout == edited

    def test_cut_copied_to_a_page_without_dependencies_survives_processing(self):
        filter_ = Filter()
        source = make_page_id("source.tiff")
        target = make_page_id("target.tiff")
        filter_.set_manual_split(source, split_x=30.0, width=200, height=100)
        filter_.set_params(
            target, Params(page_layout=PageLayout.single_page_uncut(*_SIZE))
        )

        filter_.apply_layout_type(
            source, [target], LayoutType.TWO_PAGES, apply_cut=True
        )
        result = filter_.process(blank_image(), target)

        assert result.page_layout is not None
        assert result.page_layout.layout_type == PageLayoutType.TWO_PAGES
        assert result.page_layout.get_split_line_x() == 30.0


_IMPORTED_PARAMS = """<params mode="manual">
<pages type="two-pages">
<outline><point x="0" y="0"/><point x="200" y="0"/><point x="200" y="100"/>
<point x="0" y="100"/></outline>
<cutter1><p1 x="30" y="0"/><p2 x="30" y="100"/></cutter1>
</pages>{dependencies}
</params>"""

_IMPORTED_DEPENDENCIES = """<dependencies>
<size width="200" height="100"/><rotation degrees="0"/>
<layoutType>two-pages</layoutType>
</dependencies>"""


class TestImportedLayouts:
    def _import(self, dependencies: str) -> Params:
        element = ET.fromstring(_IMPORTED_PARAMS.format(dependencies=dependencies))
        return _read_params(element, LayoutType.TWO_PAGES)

    def test_imported_dependencies_of_a_rotated_page_use_the_oriented_size(self):
        unrotated = _IMPORTED_DEPENDENCIES.replace(
            '<size width="200" height="100"/><rotation degrees="0"/>',
            '<size width="100" height="200"/><rotation degrees="90"/>',
        )

        imported = self._import(unrotated)

        assert imported.dependencies.image_size == (200, 100)
        assert imported.dependencies.rotation_degrees == 90

    def test_imported_layout_with_dependencies_is_kept_by_processing(self):
        filter_ = Filter()
        page = make_page_id()
        imported = self._import(_IMPORTED_DEPENDENCIES)
        filter_.set_params(page, imported)

        result = filter_.process(blank_image(), page)

        assert result == imported
        assert result.page_layout is not None
        assert result.page_layout.get_split_line_x() == 30.0

    def test_imported_layout_without_dependencies_is_adapted_not_kept_as_is(self):
        filter_ = Filter()
        page = make_page_id()
        imported = self._import("")
        assert imported.dependencies.is_null()
        filter_.set_params(page, imported)

        result = filter_.process(blank_image((400, 100)), page)

        assert result.page_layout is not None
        assert result.page_layout.outline == (0.0, 0.0, 400.0, 100.0)
        assert result.dependencies.image_size == (400, 100)

    def test_imported_auto_layout_without_dependencies_is_detected_again(self):
        filter_ = Filter()
        page = make_page_id()
        element = ET.fromstring(_IMPORTED_PARAMS.format(dependencies=""))
        element.set("mode", "auto")
        imported = _read_params(element, LayoutType.AUTO_LAYOUT_TYPE)
        filter_.set_params(page, imported)

        result = filter_.process(blank_image(), page)

        assert result.dependencies.image_size == _SIZE
        assert result.is_auto()


class TestStoredJson:
    def test_params_saved_without_dependencies_load_as_null_and_are_redetected(self):
        saved = Params(page_layout=PageLayout.two_pages(*_SIZE, 10.0)).model_dump(
            mode="json", exclude={"dependencies"}
        )
        filter_ = Filter()
        page = make_page_id()
        filter_.set_params(page, Params.model_validate(saved))

        result = filter_.process(blank_image(), page)

        assert result.dependencies.image_size == _SIZE
