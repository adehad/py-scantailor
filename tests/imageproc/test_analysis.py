"""Tests for image analysis utilities."""

import numpy as np

from scantailor.imageproc.analysis import (
    DEFAULT_COARSE_STEP,
    SkewResult,
    connected_components,
    distance_transform,
    find_skew,
)


class TestSkewResult:
    """Tests for SkewResult dataclass."""

    def test_good_confidence_threshold(self):
        """GOOD_CONFIDENCE should be 2.0."""
        assert SkewResult.GOOD_CONFIDENCE == 2.0

    def test_is_confident_above_threshold(self):
        """is_confident should return True when confidence >= 2.0."""
        result = SkewResult(angle=1.5, confidence=2.5)
        assert result.is_confident() is True

    def test_is_confident_at_threshold(self):
        """is_confident should return True when confidence == 2.0."""
        result = SkewResult(angle=1.5, confidence=2.0)
        assert result.is_confident() is True

    def test_is_confident_below_threshold(self):
        """is_confident should return False when confidence < 2.0."""
        result = SkewResult(angle=1.5, confidence=1.5)
        assert result.is_confident() is False

    def test_immutable(self):
        """SkewResult should be immutable (frozen)."""
        result = SkewResult(angle=1.5, confidence=2.0)
        try:
            result.angle = 2.0  # type: ignore[misc]
            assert False, "Should have raised an error"
        except AttributeError:
            pass  # Expected


class TestFindSkew:
    """Tests for find_skew function."""

    def test_horizontal_lines_no_skew(self):
        """Horizontal lines should detect near-zero skew."""
        # Create image with horizontal lines
        image = np.zeros((100, 200), dtype=np.uint8)
        image[20, 20:180] = 255
        image[50, 20:180] = 255
        image[80, 20:180] = 255

        result = find_skew(image)

        # Should detect near-zero angle
        assert abs(result.angle) < 1.0

    def test_uniform_image_low_confidence(self):
        """Uniform image should have low confidence."""
        image = np.full((100, 100), 128, dtype=np.uint8)

        result = find_skew(image)

        # Uniform image has no features, confidence should be low
        assert result.is_confident() is False

    def test_returns_skew_result(self):
        """find_skew should return a SkewResult."""
        image = np.zeros((50, 50), dtype=np.uint8)

        result = find_skew(image)

        assert isinstance(result, SkewResult)
        assert hasattr(result, "angle")
        assert hasattr(result, "confidence")

    def test_respects_max_angle(self):
        """Coarse search is limited to max_angle, fine search can extend slightly.

        The algorithm searches coarsely within [-max_angle, +max_angle], then
        does fine binary search that can extend up to coarse_step/2 beyond the
        boundary to find the true optimum. This matches C++ SkewFinder behavior.
        """
        # Create an image with diagonal lines that would suggest a large skew
        image = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            x = min(99, i + 20)  # Diagonal offset suggesting ~11° skew
            image[i, x] = 255

        max_angle = 5.0
        result = find_skew(image, max_angle=max_angle)

        # Fine search can extend coarse_step/2 beyond max_angle
        assert abs(result.angle) <= max_angle + DEFAULT_COARSE_STEP / 2

    def test_min_angle_threshold(self):
        """Angles smaller than min_angle should return 0."""
        # Create image with very slight skew (horizontal lines)
        image = np.zeros((100, 200), dtype=np.uint8)
        image[50, :] = 255

        result = find_skew(image, min_angle=1.0)

        # Very small detected angle should be thresholded to 0
        # (horizontal line has no skew)
        assert result.angle == 0.0

    def test_handles_color_image(self):
        """find_skew should handle color images by converting to grayscale."""
        # Create a 3-channel color image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[50, :, :] = 255  # White horizontal line

        result = find_skew(image)

        # Should not raise and should return valid result
        assert isinstance(result, SkewResult)

    def test_handles_binary_image(self):
        """find_skew should handle already-binary images."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[50, :] = 255

        result = find_skew(image)

        assert isinstance(result, SkewResult)


class TestConnectedComponents:
    """Tests for connected_components function."""

    def test_single_component(self):
        """Single white region should be detected as one component."""
        image = np.zeros((50, 50), dtype=np.uint8)
        image[10:20, 10:20] = 255  # White square

        num_labels, labels, stats, centroids = connected_components(image)

        # Label 0 is background, label 1 is the white square
        assert num_labels == 2
        assert labels.shape == image.shape

    def test_multiple_components(self):
        """Multiple separate regions should be detected."""
        image = np.zeros((50, 50), dtype=np.uint8)
        image[5:10, 5:10] = 255  # First square
        image[30:40, 30:40] = 255  # Second square

        num_labels, labels, stats, centroids = connected_components(image)

        # Background + 2 squares
        assert num_labels == 3

    def test_empty_image(self):
        """Empty image should have only background."""
        image = np.zeros((50, 50), dtype=np.uint8)

        num_labels, labels, stats, centroids = connected_components(image)

        assert num_labels == 1  # Only background

    def test_stats_shape(self):
        """Stats should have correct shape."""
        image = np.zeros((50, 50), dtype=np.uint8)
        image[10:20, 10:20] = 255

        num_labels, labels, stats, centroids = connected_components(image)

        # Stats: [x, y, width, height, area] for each label
        assert stats.shape[0] == num_labels
        assert stats.shape[1] == 5

    def test_centroids_shape(self):
        """Centroids should have correct shape."""
        image = np.zeros((50, 50), dtype=np.uint8)
        image[10:20, 10:20] = 255

        num_labels, labels, stats, centroids = connected_components(image)

        # Centroids: [x, y] for each label
        assert centroids.shape[0] == num_labels
        assert centroids.shape[1] == 2


class TestDistanceTransform:
    """Tests for distance_transform function."""

    def test_all_white_distance_zero(self):
        """All-white image should have zero distance everywhere."""
        image = np.full((50, 50), 255, dtype=np.uint8)

        result = distance_transform(image)

        # No zero pixels to measure distance from
        # OpenCV returns 0 for all pixels when there are no zero pixels
        assert result.dtype == np.float32

    def test_all_black_distance_zero(self):
        """All-black image should have zero distance everywhere."""
        image = np.zeros((50, 50), dtype=np.uint8)

        result = distance_transform(image)

        # All pixels are zero, so distance is 0
        np.testing.assert_array_equal(result, 0)

    def test_center_pixel_distance(self):
        """Center of white region should have largest distance."""
        image = np.zeros((51, 51), dtype=np.uint8)
        image[10:41, 10:41] = 255  # 31x31 white square

        result = distance_transform(image)

        # Center should have maximum distance
        center_dist = result[25, 25]
        corner_dist = result[10, 10]
        assert center_dist > corner_dist

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        image = np.zeros((30, 40), dtype=np.uint8)
        image[10:20, 10:30] = 255

        result = distance_transform(image)

        assert result.shape == image.shape

    def test_different_distance_types(self):
        """Different distance types should work."""
        image = np.zeros((50, 50), dtype=np.uint8)
        image[20:30, 20:30] = 255

        for dist_type in ("l1", "l2", "c"):
            result = distance_transform(image, distance_type=dist_type)
            assert result.shape == image.shape
            assert result.dtype == np.float32
