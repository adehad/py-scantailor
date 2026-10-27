"""Tests for content detection algorithms."""

import numpy as np

from scantailor.filters.select_content.content_finder import (
    ContentDetectionResult,
    find_content_box,
)


class TestFindContentBox:
    """Tests for find_content_box function."""

    def test_returns_content_detection_result(self):
        """Should return a ContentDetectionResult."""
        image = np.zeros((100, 100), dtype=np.uint8)
        result = find_content_box(image)

        assert isinstance(result, ContentDetectionResult)
        assert hasattr(result, "content_box")
        assert hasattr(result, "confidence")

    def test_black_image_returns_an_empty_box(self):
        """A page that is all shadow holds no content."""
        image = np.zeros((100, 100), dtype=np.uint8)
        result = find_content_box(image)

        assert result.content_box.is_empty()
        assert result.confidence == 0.0

    def test_white_image_returns_an_empty_box(self):
        """A blank page holds no content."""
        image = np.ones((100, 100), dtype=np.uint8) * 255
        result = find_content_box(image)

        assert result.content_box.is_empty()
        assert result.confidence == 0.0

    def test_detects_centered_content(self):
        """Should detect content centered in image."""
        image = np.ones((300, 300), dtype=np.uint8) * 255  # White background

        # Draw text-like lines (multiple horizontal lines)
        for y in range(80, 220, 15):
            image[y : y + 3, 80:220] = 0

        result = find_content_box(image, dpi=150.0)

        # Content should be detected
        assert result.content_box.is_valid()
        # The bounding box should contain the content area
        assert result.content_box.right > 80
        assert result.content_box.bottom > 80

    def test_detects_small_content(self):
        """Should detect small content region."""
        image = np.ones((300, 300), dtype=np.uint8) * 255  # White background

        # Draw small text-like content
        image[100:120, 100:200] = 0

        result = find_content_box(image, dpi=150.0)

        assert result.content_box.is_valid()

    def test_handles_color_image(self):
        """Should handle color (BGR) images."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Draw black rectangle
        image[30:70, 30:70] = 0

        result = find_content_box(image)

        assert result.content_box.is_valid()

    def test_respects_dpi_scaling(self):
        """Higher DPI should scale processing appropriately."""
        image = np.ones((300, 300), dtype=np.uint8) * 255
        image[100:200, 100:200] = 0

        result_300 = find_content_box(image, dpi=300.0)
        result_150 = find_content_box(image, dpi=150.0)

        # Both should find content (coordinates may differ slightly due to scaling)
        assert result_300.content_box.is_valid()
        assert result_150.content_box.is_valid()


class TestContentDetectionResult:
    """Tests for ContentDetectionResult."""

    def test_is_confident_threshold(self):
        """is_confident should use 0.5 threshold."""
        from scantailor.filters.select_content.content_box import ContentBox

        high_conf = ContentDetectionResult(
            content_box=ContentBox(x=0, y=0, width=100, height=100),
            confidence=0.8,
        )
        low_conf = ContentDetectionResult(
            content_box=ContentBox(x=0, y=0, width=100, height=100),
            confidence=0.3,
        )

        assert high_conf.is_confident()
        assert not low_conf.is_confident()

    def test_boundary_confidence(self):
        """Confidence exactly at 0.5 should be confident."""
        from scantailor.filters.select_content.content_box import ContentBox

        result = ContentDetectionResult(
            content_box=ContentBox(x=0, y=0, width=100, height=100),
            confidence=0.5,
        )
        assert result.is_confident()
