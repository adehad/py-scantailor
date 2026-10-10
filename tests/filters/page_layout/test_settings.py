"""Tests for Page Layout settings."""

from pathlib import Path

from scantailor.core.models import ImageId, Margins, PageId, SubPage
from scantailor.filters.page_layout.params import ContentRect, ContentSize, Params
from scantailor.filters.page_layout.settings import ContentSizeUpdate, Settings


def _page(name: str) -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path(f"/{name}.tiff")),
        sub_page=SubPage.SINGLE_PAGE,
    )


def _update(
    settings: Settings, page_id: PageId, width_mm: float, height_mm: float
) -> ContentSizeUpdate:
    return settings.update_content_size_and_get_aggregate(
        page_id,
        ContentRect(x=0, y=0, width=100, height=100),
        ContentRect(x=10, y=10, width=50, height=50),
        ContentSize(width_mm=width_mm, height_mm=height_mm),
    )


class TestUpdateContentSizeAndGetAggregate:
    """The update and the aggregate reads happen under one lock."""

    def test_new_page_gets_default_layout_and_given_geometry(self) -> None:
        """An unknown page is created with default margins and the new sizes."""
        settings = Settings()
        page_id = _page("a")

        update = _update(settings, page_id, 30.0, 40.0)

        assert update.params.hard_margins_mm == Margins(
            left=10.0, top=5.0, right=10.0, bottom=5.0
        )
        assert update.params.content_size_mm == ContentSize(
            width_mm=30.0, height_mm=40.0
        )
        assert settings.get_params(page_id) == update.params

    def test_aggregate_before_and_after_bracket_the_update(self) -> None:
        """Before excludes the page's new size, after includes it."""
        settings = Settings()
        _update(settings, _page("a"), 10.0, 10.0)

        update = _update(settings, _page("b"), 50.0, 20.0)

        # Hard size is content plus 20 mm across and 10 mm down.
        assert update.aggregate_before == ContentSize(width_mm=30.0, height_mm=20.0)
        assert update.aggregate_after == ContentSize(width_mm=70.0, height_mm=30.0)
        assert update.aggregate_changed is True

    def test_unchanged_aggregate_is_reported(self) -> None:
        """A smaller page leaves the aggregate alone."""
        settings = Settings()
        _update(settings, _page("a"), 50.0, 50.0)

        update = _update(settings, _page("b"), 10.0, 10.0)

        assert update.aggregate_changed is False

    def test_existing_margins_survive(self) -> None:
        """Only geometry fields change on a known page."""
        settings = Settings()
        page_id = _page("a")
        settings.set_margins(page_id, Margins.uniform(3.0))

        update = _update(settings, page_id, 20.0, 20.0)

        assert update.params.hard_margins_mm == Margins.uniform(3.0)

    def test_auto_margins_are_adopted_only_when_enabled(self) -> None:
        """Calculated margins replace the stored ones only for auto-margin pages."""
        settings = Settings()
        manual, auto = _page("manual"), _page("auto")
        settings.set_margins(manual, Margins.uniform(3.0))
        settings.set_params(auto, settings.get_params(auto).with_auto_margins(True))
        calculated = Margins.uniform(7.0)
        page_rect = ContentRect(width=100, height=100)
        content_rect = ContentRect(width=50, height=50)
        size = ContentSize(width_mm=1.0, height_mm=1.0)

        manual_update = settings.update_content_size_and_get_aggregate(
            manual, page_rect, content_rect, size, auto_margins_mm=calculated
        )
        auto_update = settings.update_content_size_and_get_aggregate(
            auto, page_rect, content_rect, size, auto_margins_mm=calculated
        )

        assert manual_update.params.hard_margins_mm == Margins.uniform(3.0)
        assert auto_update.params.hard_margins_mm == calculated


class TestAutoMarginsAggregateBefore:
    """The aggregate before the update already includes adopted auto margins."""

    def test_auto_margin_change_alone_does_not_report_aggregate_change(self) -> None:
        """Adopting new auto margins is not part of the content size update."""
        settings = Settings()
        page_id = _page("auto")
        settings.set_params(
            page_id,
            Params(
                auto_margins=True,
                content_size_mm=ContentSize(width_mm=30.0, height_mm=40.0),
            ),
        )

        update = settings.update_content_size_and_get_aggregate(
            page_id,
            ContentRect(width=100, height=100),
            ContentRect(width=50, height=50),
            ContentSize(width_mm=30.0, height_mm=40.0),
            auto_margins_mm=Margins.uniform(20.0),
        )

        assert update.aggregate_before == update.aggregate_after
