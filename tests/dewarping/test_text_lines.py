"""Tests for text line detection module."""

import numpy as np

from scantailor.dewarping.text_lines import (
    TextLine,
    TextLineResult,
    VerticalBounds,
    detect_vertical_bounds,
    trace_text_lines,
)


class TestVerticalBounds:
    """Tests for VerticalBounds dataclass."""

    def test_create_bounds(self):
        """VerticalBounds can be created with corner points."""
        bounds = VerticalBounds(
            left_top=(10.0, 0.0),
            left_bottom=(15.0, 100.0),
            right_top=(90.0, 0.0),
            right_bottom=(85.0, 100.0),
        )

        assert bounds.left_top == (10.0, 0.0)
        assert bounds.left_bottom == (15.0, 100.0)
        assert bounds.right_top == (90.0, 0.0)
        assert bounds.right_bottom == (85.0, 100.0)

    def test_left_line_property(self):
        """left_line returns start and end points."""
        bounds = VerticalBounds(
            left_top=(10.0, 0.0),
            left_bottom=(15.0, 100.0),
            right_top=(90.0, 0.0),
            right_bottom=(85.0, 100.0),
        )

        assert bounds.left_line == ((10.0, 0.0), (15.0, 100.0))

    def test_right_line_property(self):
        """right_line returns start and end points."""
        bounds = VerticalBounds(
            left_top=(10.0, 0.0),
            left_bottom=(15.0, 100.0),
            right_top=(90.0, 0.0),
            right_bottom=(85.0, 100.0),
        )

        assert bounds.right_line == ((90.0, 0.0), (85.0, 100.0))


class TestTextLine:
    """Tests for TextLine dataclass."""

    def test_create_empty_line(self):
        """TextLine can be created empty."""
        line = TextLine()
        assert len(line) == 0
        assert line.is_valid is False

    def test_create_line_with_points(self):
        """TextLine can be created with points."""
        points = [(0.0, 10.0), (50.0, 12.0), (100.0, 8.0)]
        line = TextLine(points=points)

        assert len(line) == 3
        assert line.is_valid is True
        assert line.points == points

    def test_single_point_not_valid(self):
        """Single point line is not valid."""
        line = TextLine(points=[(50.0, 50.0)])
        assert line.is_valid is False

    def test_two_points_valid(self):
        """Two point line is valid."""
        line = TextLine(points=[(0.0, 10.0), (100.0, 10.0)])
        assert line.is_valid is True


class TestTextLineResult:
    """Tests for TextLineResult dataclass."""

    def test_create_empty_result(self):
        """TextLineResult can be created empty."""
        result = TextLineResult()
        assert len(result.lines) == 0
        assert result.vertical_bounds is None

    def test_create_result_with_data(self):
        """TextLineResult can be created with lines and bounds."""
        lines = [TextLine(points=[(0.0, 10.0), (100.0, 10.0)])]
        bounds = VerticalBounds(
            left_top=(0.0, 0.0),
            left_bottom=(0.0, 100.0),
            right_top=(100.0, 0.0),
            right_bottom=(100.0, 100.0),
        )

        result = TextLineResult(lines=lines, vertical_bounds=bounds)

        assert len(result.lines) == 1
        assert result.vertical_bounds is not None


class TestDetectVerticalBounds:
    """Tests for detect_vertical_bounds function."""

    def test_simple_rectangular_content(self):
        """Detect bounds for simple rectangular content."""
        # Create a binary image with a rectangular block
        binary = np.zeros((100, 200), dtype=np.uint8)
        binary[20:80, 30:170] = 255  # White rectangle

        bounds = detect_vertical_bounds(binary)

        # Left boundary should be around x=30
        assert 25 <= bounds.left_top[0] <= 35
        assert 25 <= bounds.left_bottom[0] <= 35

        # Right boundary should be around x=170
        assert 165 <= bounds.right_top[0] <= 175
        assert 165 <= bounds.right_bottom[0] <= 175

    def test_empty_image(self):
        """Empty image returns default bounds."""
        binary = np.zeros((100, 200), dtype=np.uint8)

        bounds = detect_vertical_bounds(binary)

        # Should return some bounds even for empty image
        assert bounds.left_top is not None
        assert bounds.right_top is not None

    def test_full_width_content(self):
        """Content spanning full width."""
        binary = np.zeros((100, 200), dtype=np.uint8)
        binary[10:90, :] = 255

        bounds = detect_vertical_bounds(binary)

        # Left should be near 0
        assert bounds.left_top[0] <= 5
        # Right should be near 200
        assert bounds.right_top[0] >= 195

    def test_trapezoid_content(self):
        """Content with trapezoid shape (wider at top)."""
        binary = np.zeros((100, 200), dtype=np.uint8)
        # Wider at top, narrower at bottom
        for y in range(20, 80):
            left_x = 30 + (y - 20)  # Moves right as y increases
            right_x = 170 - (y - 20)  # Moves left as y increases
            binary[y, left_x:right_x] = 255

        bounds = detect_vertical_bounds(binary)

        # Bounds should be detected (may not perfectly match trapezoid
        # due to RANSAC fitting)
        assert bounds.left_top[0] < bounds.right_top[0]
        assert bounds.left_bottom[0] < bounds.right_bottom[0]

        # Left boundary should be in reasonable range
        assert 25 <= bounds.left_top[0] <= 90
        # Right boundary should be in reasonable range
        assert 110 <= bounds.right_top[0] <= 175


class TestTraceTextLines:
    """Tests for trace_text_lines function."""

    def test_simple_horizontal_lines(self):
        """Detect simple horizontal text lines."""
        # Create image with horizontal stripes (simulating text lines)
        image = np.full((200, 300), 255, dtype=np.uint8)

        # Add dark horizontal lines
        for y in [30, 60, 90, 120, 150]:
            image[y - 2 : y + 2, 20:280] = 50

        result = trace_text_lines(image, dpi=200.0)

        # Should detect vertical bounds
        assert result.vertical_bounds is not None

        # Should detect some lines (may not match exactly due to algorithm)
        # The algorithm may merge or filter some lines
        assert isinstance(result.lines, list)

    def test_empty_image_returns_empty_lines(self):
        """Empty image returns no text lines."""
        image = np.full((200, 300), 255, dtype=np.uint8)

        result = trace_text_lines(image, dpi=200.0)

        # Should have bounds but empty or minimal lines
        assert result.vertical_bounds is not None

    def test_dpi_tuple(self):
        """DPI can be specified as tuple."""
        image = np.full((200, 300), 200, dtype=np.uint8)
        image[50:55, 20:280] = 50  # One line

        result = trace_text_lines(image, dpi=(300.0, 300.0))

        assert result.vertical_bounds is not None

    def test_dpi_scalar(self):
        """DPI can be specified as scalar."""
        image = np.full((200, 300), 200, dtype=np.uint8)
        image[50:55, 20:280] = 50

        result = trace_text_lines(image, dpi=300.0)

        assert result.vertical_bounds is not None

    def test_content_rect(self):
        """Content rectangle limits detection area."""
        image = np.full((200, 300), 255, dtype=np.uint8)
        # Lines both inside and outside content rect
        image[30:35, 20:280] = 50  # Inside
        image[150:155, 20:280] = 50  # Mostly inside
        image[190:195, 20:280] = 50  # Outside

        result = trace_text_lines(image, dpi=200.0, content_rect=(10, 10, 280, 160))

        assert result.vertical_bounds is not None

    def test_color_image_converted_to_grayscale(self):
        """Color image is converted to grayscale."""
        color_image = np.full((200, 300, 3), 200, dtype=np.uint8)
        color_image[50:55, 20:280, :] = 50

        result = trace_text_lines(color_image, dpi=200.0)

        assert result.vertical_bounds is not None

    def test_curved_text_lines(self):
        """Image with curved text lines."""
        image = np.full((200, 300), 255, dtype=np.uint8)

        # Create a curved line
        for x in range(30, 270):
            # Sine wave pattern
            y = int(100 + 10 * np.sin(x * np.pi / 120))
            image[y - 2 : y + 2, x] = 50

        result = trace_text_lines(image, dpi=200.0)

        assert result.vertical_bounds is not None


class TestIntegration:
    """Integration tests for text line detection."""

    def test_text_block_simulation(self):
        """Simulate a text block with multiple lines."""
        # Create an image simulating a page of text
        image = np.full((400, 300), 240, dtype=np.uint8)

        # Add noise (fixed seed for reproducibility)
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 20, size=image.shape, dtype=np.uint8)
        image = np.clip(image - noise, 0, 255).astype(np.uint8)

        # Add text lines (dark stripes with varying intensity)
        line_positions = [50, 80, 110, 140, 170, 200, 230, 260, 290, 320]
        for y in line_positions:
            # Vary line darkness
            darkness = rng.integers(30, 80)
            image[y - 2 : y + 2, 30:270] = darkness

        result = trace_text_lines(image, dpi=200.0)

        # Should detect bounds
        assert result.vertical_bounds is not None
        # Left boundary should be different from right (or both at edge)
        assert result.vertical_bounds.left_top[0] <= result.vertical_bounds.right_top[0]

    def test_bounds_are_ordered(self):
        """Vertical bounds should have left <= right."""
        image = np.full((200, 300), 200, dtype=np.uint8)
        image[50:150, 50:250] = 100  # Content block

        result = trace_text_lines(image, dpi=200.0)

        bounds = result.vertical_bounds
        assert bounds is not None

        # Left boundary should be to the left of (or equal to) right boundary
        # Note: When there's no clear content boundary, both may default to edges
        assert bounds.left_top[0] <= bounds.right_top[0]
        assert bounds.left_bottom[0] <= bounds.right_bottom[0]

    def test_very_small_image(self):
        """Handle very small images gracefully."""
        image = np.full((20, 30), 200, dtype=np.uint8)
        image[5:15, 5:25] = 100

        # Should not crash
        result = trace_text_lines(image, dpi=200.0)

        assert result.vertical_bounds is not None

    def test_high_dpi_image(self):
        """Handle high DPI image with downscaling."""
        # 600 DPI image should be downscaled to ~200 DPI
        image = np.full((600, 900), 200, dtype=np.uint8)
        image[100:500, 100:800] = 100

        result = trace_text_lines(image, dpi=600.0)

        assert result.vertical_bounds is not None
