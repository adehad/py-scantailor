"""Tests for Dependencies class."""

from scantailor.filters.select_content import (
    ContentDetectionMode,
    Dependencies,
    PageDetectionMode,
)


class TestDependencies:
    """Tests for Dependencies class."""

    def test_default_values(self):
        """Default dependencies should have standard modes."""
        deps = Dependencies()
        assert deps.content_detection_mode == ContentDetectionMode.AUTO
        assert deps.page_detection_mode == PageDetectionMode.DISABLED
        assert deps.fine_tune_corners is False
        assert deps.rotated_page_outline == []

    def test_needs_content_update_mode_change(self):
        """Content update needed when mode changes."""
        deps1 = Dependencies(content_detection_mode=ContentDetectionMode.AUTO)
        deps2 = Dependencies(content_detection_mode=ContentDetectionMode.MANUAL)

        assert deps1.needs_content_update(deps2)

    def test_needs_content_update_same_mode(self):
        """No content update when mode is the same."""
        deps1 = Dependencies(content_detection_mode=ContentDetectionMode.AUTO)
        deps2 = Dependencies(content_detection_mode=ContentDetectionMode.AUTO)

        assert not deps1.needs_content_update(deps2)

    def test_needs_content_update_outline_change(self):
        """Content update needed when page outline changes."""
        deps1 = Dependencies(rotated_page_outline=[(0, 0), (100, 0)])
        deps2 = Dependencies(rotated_page_outline=[(0, 0), (200, 0)])

        assert deps1.needs_content_update(deps2)

    def test_needs_page_update_mode_change(self):
        """Page update needed when mode changes."""
        deps1 = Dependencies(page_detection_mode=PageDetectionMode.DISABLED)
        deps2 = Dependencies(page_detection_mode=PageDetectionMode.AUTO)

        assert deps1.needs_page_update(deps2)

    def test_needs_page_update_same_mode(self):
        """No page update when mode is the same."""
        deps1 = Dependencies(page_detection_mode=PageDetectionMode.DISABLED)
        deps2 = Dependencies(page_detection_mode=PageDetectionMode.DISABLED)

        assert not deps1.needs_page_update(deps2)

    def test_needs_page_update_outline_change(self):
        """Page update needed when page outline changes."""
        deps1 = Dependencies(rotated_page_outline=[(0, 0), (100, 0)])
        deps2 = Dependencies(rotated_page_outline=[(0, 0), (200, 0)])

        assert deps1.needs_page_update(deps2)

    def test_needs_page_update_fine_tune_in_auto_mode(self):
        """Page update needed when fine-tune changes in AUTO mode."""
        deps1 = Dependencies(
            page_detection_mode=PageDetectionMode.AUTO, fine_tune_corners=False
        )
        deps2 = Dependencies(
            page_detection_mode=PageDetectionMode.AUTO, fine_tune_corners=True
        )

        assert deps1.needs_page_update(deps2)

    def test_needs_page_update_fine_tune_in_disabled_mode(self):
        """No page update for fine-tune change when not in AUTO mode."""
        deps1 = Dependencies(
            page_detection_mode=PageDetectionMode.DISABLED, fine_tune_corners=False
        )
        deps2 = Dependencies(
            page_detection_mode=PageDetectionMode.DISABLED, fine_tune_corners=True
        )

        # Fine-tune only matters when page detection is AUTO
        assert not deps1.needs_page_update(deps2)

    def test_is_compatible_with_identical(self):
        """Identical dependencies should be compatible."""
        deps1 = Dependencies()
        deps2 = Dependencies()

        assert deps1.is_compatible_with(deps2)

    def test_is_compatible_with_content_change(self):
        """Content mode change makes deps incompatible."""
        deps1 = Dependencies(content_detection_mode=ContentDetectionMode.AUTO)
        deps2 = Dependencies(content_detection_mode=ContentDetectionMode.MANUAL)

        assert not deps1.is_compatible_with(deps2)

    def test_is_compatible_with_page_change(self):
        """Page mode change makes deps incompatible."""
        deps1 = Dependencies(page_detection_mode=PageDetectionMode.DISABLED)
        deps2 = Dependencies(page_detection_mode=PageDetectionMode.AUTO)

        assert not deps1.is_compatible_with(deps2)

    def test_is_compatible_fine_tune_when_irrelevant(self):
        """Fine-tune change is compatible when page detection is disabled."""
        deps1 = Dependencies(
            page_detection_mode=PageDetectionMode.DISABLED, fine_tune_corners=False
        )
        deps2 = Dependencies(
            page_detection_mode=PageDetectionMode.DISABLED, fine_tune_corners=True
        )

        assert deps1.is_compatible_with(deps2)
