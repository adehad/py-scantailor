"""Locking the aggregate size and copying single sides of the margins."""

from pathlib import Path

from scantailor.core.models import ImageId, Margins, PageId, SubPage
from scantailor.filters.page_layout.alignment import Alignment
from scantailor.filters.page_layout.filter import Filter
from scantailor.filters.page_layout.params import ContentRect, ContentSize, Params
from scantailor.filters.page_layout.settings import Settings


def _page(name: str) -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path(f"/{name}.tiff")),
        sub_page=SubPage.SINGLE_PAGE,
    )


def _measured(width_mm: float, height_mm: float) -> Params:
    return Params(
        hard_margins_mm=Margins.uniform(0.0),
        content_size_mm=ContentSize(width_mm=width_mm, height_mm=height_mm),
        page_rect=ContentRect(x=0, y=0, width=100, height=100),
    )


def _settings_with_two_pages() -> tuple[Settings, PageId, PageId]:
    settings = Settings()
    small, large = _page("small"), _page("large")
    settings.set_params(small, _measured(100.0, 150.0))
    settings.set_params(large, _measured(200.0, 300.0))
    return settings, small, large


class TestFrozenAggregateSize:
    def test_aggregate_is_not_frozen_by_default(self) -> None:
        settings, _, _ = _settings_with_two_pages()

        assert not settings.is_aggregate_hard_size_frozen()
        assert settings.get_aggregate_hard_size_mm() == ContentSize(
            width_mm=200.0, height_mm=300.0
        )

    def test_frozen_aggregate_keeps_the_size_it_had_when_frozen(self) -> None:
        settings, _, large = _settings_with_two_pages()

        settings.set_aggregate_hard_size_frozen(True)
        settings.set_params(large, _measured(50.0, 60.0))

        assert settings.is_aggregate_hard_size_frozen()
        assert settings.get_aggregate_hard_size_mm() == ContentSize(
            width_mm=200.0, height_mm=300.0
        )

    def test_releasing_follows_the_pages_again(self) -> None:
        settings, _, large = _settings_with_two_pages()
        settings.set_aggregate_hard_size_frozen(True)
        settings.set_params(large, _measured(50.0, 60.0))

        settings.set_aggregate_hard_size_frozen(False)

        assert not settings.is_aggregate_hard_size_frozen()
        assert settings.get_aggregate_hard_size_mm() == ContentSize(
            width_mm=100.0, height_mm=150.0
        )

    def test_freezing_twice_keeps_the_first_size(self) -> None:
        settings, _, large = _settings_with_two_pages()
        settings.set_aggregate_hard_size_frozen(True)
        settings.set_params(large, _measured(50.0, 60.0))

        settings.set_aggregate_hard_size_frozen(True)

        assert settings.get_aggregate_hard_size_mm() == ContentSize(
            width_mm=200.0, height_mm=300.0
        )

    def test_frozen_aggregate_survives_pages_leaving_size_matching(self) -> None:
        settings, _, large = _settings_with_two_pages()
        settings.set_aggregate_hard_size_frozen(True)
        settings.set_params(
            large, _measured(200.0, 300.0).with_alignment(Alignment(is_null=True))
        )

        assert settings.get_aggregate_hard_size_mm() == ContentSize(
            width_mm=200.0, height_mm=300.0
        )

    def test_update_does_not_report_a_change_while_frozen(self) -> None:
        settings, _, _ = _settings_with_two_pages()
        settings.set_aggregate_hard_size_frozen(True)

        update = settings.update_content_size_and_get_aggregate(
            _page("new"),
            ContentRect(x=0, y=0, width=100, height=100),
            ContentRect(x=0, y=0, width=100, height=100),
            ContentSize(width_mm=900.0, height_mm=900.0),
        )

        assert not update.aggregate_changed
        assert update.aggregate_after == ContentSize(width_mm=200.0, height_mm=300.0)

    def test_soft_margins_are_worked_out_against_the_frozen_size(self) -> None:
        settings, small, large = _settings_with_two_pages()
        filter_ = Filter(settings)
        settings.set_aggregate_hard_size_frozen(True)
        settings.set_params(large, _measured(20.0, 30.0))

        soft = filter_.calculate_soft_margins(small)

        assert soft.left + soft.right == 100.0
        assert soft.top + soft.bottom == 150.0

    def test_clear_releases_the_frozen_size(self) -> None:
        settings, _, _ = _settings_with_two_pages()
        settings.set_aggregate_hard_size_frozen(True)

        settings.clear()

        assert not settings.is_aggregate_hard_size_frozen()

    def test_frozen_size_can_be_saved_and_restored(self) -> None:
        settings, _, _ = _settings_with_two_pages()
        settings.set_aggregate_hard_size_frozen(True)
        saved = settings.get_frozen_aggregate_hard_size_mm()
        assert saved is not None

        restored = Settings()
        restored.restore_frozen_aggregate_hard_size_mm(
            ContentSize.model_validate(saved.model_dump(mode="json"))
        )

        assert restored.is_aggregate_hard_size_frozen()
        assert restored.get_aggregate_hard_size_mm() == ContentSize(
            width_mm=200.0, height_mm=300.0
        )

    def test_unfrozen_settings_have_nothing_to_save(self) -> None:
        settings, _, _ = _settings_with_two_pages()

        assert settings.get_frozen_aggregate_hard_size_mm() is None

    def test_restoring_nothing_releases_the_size(self) -> None:
        settings, _, _ = _settings_with_two_pages()
        settings.set_aggregate_hard_size_frozen(True)

        settings.restore_frozen_aggregate_hard_size_mm(None)

        assert not settings.is_aggregate_hard_size_frozen()


class TestApplyMarginSides:
    def test_only_the_chosen_sides_are_taken(self) -> None:
        settings = Settings()
        target = _page("target")
        settings.set_margins(target, Margins(top=1, bottom=2, left=3, right=4))

        settings.apply_margins_to_pages(
            [target], Margins(top=10, bottom=20, left=30, right=40), {"left", "top"}
        )

        assert settings.get_params(target).hard_margins_mm == Margins(
            top=10, bottom=2, left=30, right=4
        )

    def test_every_side_is_taken_by_default(self) -> None:
        settings = Settings()
        target = _page("target")

        settings.apply_margins_to_pages(
            [target], Margins(top=10, bottom=20, left=30, right=40)
        )

        assert settings.get_params(target).hard_margins_mm == Margins(
            top=10, bottom=20, left=30, right=40
        )

    def test_no_sides_leave_the_margins_alone(self) -> None:
        settings = Settings()
        target = _page("target")
        settings.set_margins(target, Margins(top=1, bottom=2, left=3, right=4))

        settings.apply_margins_to_pages([target], Margins.uniform(99.0), set())

        assert settings.get_params(target).hard_margins_mm == Margins(
            top=1, bottom=2, left=3, right=4
        )

    def test_a_page_without_parameters_starts_from_the_default_margins(self) -> None:
        settings = Settings()
        target = _page("target")

        settings.apply_margins_to_pages([target], Margins.uniform(99.0), {"right"})

        assert settings.get_params(target).hard_margins_mm == Margins(
            top=5.0, bottom=5.0, left=10.0, right=99.0
        )

    def test_choosing_sides_turns_automatic_margins_off(self) -> None:
        settings = Settings()
        target = _page("target")
        settings.set_params(target, Params().with_auto_margins())

        settings.apply_margins_to_pages([target], Margins.uniform(7.0), {"top"})

        assert not settings.get_params(target).auto_margins
