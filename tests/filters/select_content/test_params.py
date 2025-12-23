"""Tests for Params class."""

from __future__ import annotations

from scantailor.filters.select_content import (
    ContentBox,
    ContentDetectionMode,
    PageBox,
    PageDetectionMode,
    Params,
)


class TestParams:
    """Tests for Params class."""

    def test_default_values(self):
        """Default params should have empty boxes and AUTO/DISABLED modes."""
        params = Params()

        assert params.content_box.is_empty()
        assert params.page_box.is_empty()
        assert params.content_detection_mode == ContentDetectionMode.AUTO
        assert params.page_detection_mode == PageDetectionMode.DISABLED
        assert params.fine_tune_corners is False

    def test_is_content_auto(self):
        """is_content_auto should return True for AUTO mode."""
        params_auto = Params(content_detection_mode=ContentDetectionMode.AUTO)
        params_manual = Params(content_detection_mode=ContentDetectionMode.MANUAL)

        assert params_auto.is_content_auto()
        assert not params_manual.is_content_auto()

    def test_is_content_manual(self):
        """is_content_manual should return True for MANUAL mode."""
        params_auto = Params(content_detection_mode=ContentDetectionMode.AUTO)
        params_manual = Params(content_detection_mode=ContentDetectionMode.MANUAL)

        assert not params_auto.is_content_manual()
        assert params_manual.is_content_manual()

    def test_is_content_disabled(self):
        """is_content_disabled should return True for DISABLED mode."""
        params_auto = Params(content_detection_mode=ContentDetectionMode.AUTO)
        params_disabled = Params(content_detection_mode=ContentDetectionMode.DISABLED)

        assert not params_auto.is_content_disabled()
        assert params_disabled.is_content_disabled()

    def test_is_page_auto(self):
        """is_page_auto should return True for AUTO mode."""
        params_auto = Params(page_detection_mode=PageDetectionMode.AUTO)
        params_disabled = Params(page_detection_mode=PageDetectionMode.DISABLED)

        assert params_auto.is_page_auto()
        assert not params_disabled.is_page_auto()

    def test_is_page_manual(self):
        """is_page_manual should return True for MANUAL mode."""
        params_manual = Params(page_detection_mode=PageDetectionMode.MANUAL)
        params_disabled = Params(page_detection_mode=PageDetectionMode.DISABLED)

        assert params_manual.is_page_manual()
        assert not params_disabled.is_page_manual()

    def test_is_page_disabled(self):
        """is_page_disabled should return True for DISABLED mode."""
        params_auto = Params(page_detection_mode=PageDetectionMode.AUTO)
        params_disabled = Params(page_detection_mode=PageDetectionMode.DISABLED)

        assert not params_auto.is_page_disabled()
        assert params_disabled.is_page_disabled()

    def test_with_content_box(self):
        """with_content_box should create new params with updated box."""
        params = Params()
        box = ContentBox(x=10, y=20, width=100, height=50)

        new_params = params.with_content_box(box)

        assert new_params.content_box == box
        assert new_params is not params

    def test_with_content_box_and_mode(self):
        """with_content_box with mode should update both."""
        params = Params(content_detection_mode=ContentDetectionMode.AUTO)
        box = ContentBox(x=10, y=20, width=100, height=50)

        new_params = params.with_content_box(box, ContentDetectionMode.MANUAL)

        assert new_params.content_box == box
        assert new_params.content_detection_mode == ContentDetectionMode.MANUAL

    def test_with_page_box(self):
        """with_page_box should create new params with updated box."""
        params = Params()
        box = PageBox(x=0, y=0, width=800, height=600)

        new_params = params.with_page_box(box)

        assert new_params.page_box == box

    def test_with_manual_content(self):
        """with_manual_content should set MANUAL mode."""
        params = Params(content_detection_mode=ContentDetectionMode.AUTO)
        box = ContentBox(x=10, y=20, width=100, height=50)

        new_params = params.with_manual_content(box)

        assert new_params.content_box == box
        assert new_params.content_detection_mode == ContentDetectionMode.MANUAL

    def test_with_manual_page(self):
        """with_manual_page should set MANUAL mode."""
        params = Params(page_detection_mode=PageDetectionMode.DISABLED)
        box = PageBox(x=0, y=0, width=800, height=600)

        new_params = params.with_manual_page(box)

        assert new_params.page_box == box
        assert new_params.page_detection_mode == PageDetectionMode.MANUAL

    def test_with_auto_content(self):
        """with_auto_content should set AUTO mode."""
        params = Params(content_detection_mode=ContentDetectionMode.MANUAL)

        new_params = params.with_auto_content()

        assert new_params.content_detection_mode == ContentDetectionMode.AUTO

    def test_with_auto_page(self):
        """with_auto_page should set AUTO mode."""
        params = Params(page_detection_mode=PageDetectionMode.DISABLED)

        new_params = params.with_auto_page()

        assert new_params.page_detection_mode == PageDetectionMode.AUTO

    def test_with_disabled_content(self):
        """with_disabled_content should set DISABLED mode."""
        params = Params(content_detection_mode=ContentDetectionMode.AUTO)

        new_params = params.with_disabled_content()

        assert new_params.content_detection_mode == ContentDetectionMode.DISABLED

    def test_with_disabled_page(self):
        """with_disabled_page should set DISABLED mode."""
        params = Params(page_detection_mode=PageDetectionMode.AUTO)

        new_params = params.with_disabled_page()

        assert new_params.page_detection_mode == PageDetectionMode.DISABLED

    def test_with_fine_tune(self):
        """with_fine_tune should set fine-tune option."""
        params = Params(fine_tune_corners=False)

        new_params = params.with_fine_tune(True)

        assert new_params.fine_tune_corners is True

    def test_ensure_content_within_page_clips(self):
        """ensure_content_within_page should clip content to page bounds."""
        params = Params(
            content_box=ContentBox(x=0, y=0, width=200, height=200),
            page_box=PageBox(x=50, y=50, width=100, height=100),
        )

        new_params = params.ensure_content_within_page()

        assert new_params.content_box.x == 50
        assert new_params.content_box.y == 50
        assert new_params.content_box.width == 100
        assert new_params.content_box.height == 100

    def test_ensure_content_within_page_outside(self):
        """Content completely outside page should become page box."""
        params = Params(
            content_box=ContentBox(x=0, y=0, width=50, height=50),
            page_box=PageBox(x=100, y=100, width=200, height=200),
        )

        new_params = params.ensure_content_within_page()

        # Content was outside, so becomes the page box
        assert new_params.content_box.x == 100
        assert new_params.content_box.y == 100
        assert new_params.content_box.width == 200
        assert new_params.content_box.height == 200

    def test_ensure_content_within_page_empty_page(self):
        """Empty page box should not modify content."""
        content = ContentBox(x=10, y=20, width=100, height=50)
        params = Params(content_box=content, page_box=PageBox())

        new_params = params.ensure_content_within_page()

        assert new_params.content_box == content

    def test_json_round_trip(self):
        """Should serialize and deserialize correctly."""
        original = Params(
            content_box=ContentBox(x=10, y=20, width=100, height=50),
            page_box=PageBox(x=0, y=0, width=800, height=600),
            content_detection_mode=ContentDetectionMode.MANUAL,
            page_detection_mode=PageDetectionMode.AUTO,
            fine_tune_corners=True,
        )

        json_str = original.model_dump_json()
        restored = Params.model_validate_json(json_str)

        assert restored.content_box == original.content_box
        assert restored.page_box == original.page_box
        assert restored.content_detection_mode == original.content_detection_mode
        assert restored.page_detection_mode == original.page_detection_mode
        assert restored.fine_tune_corners == original.fine_tune_corners
