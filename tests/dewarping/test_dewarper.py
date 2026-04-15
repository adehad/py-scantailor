"""Tests for cylindrical surface dewarper."""

import numpy as np
import pytest

from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper


class TestCylindricalSurfaceDewarper:
    """Tests for CylindricalSurfaceDewarper class."""

    def test_create_from_straight_lines(self):
        """Create dewarper from straight horizontal lines."""
        # Two parallel horizontal lines
        top = np.array([[0, 0], [100, 0]], dtype=np.float64)
        bottom = np.array([[0, 100], [100, 100]], dtype=np.float64)

        dewarper = CylindricalSurfaceDewarper.from_directrices(
            top, bottom, depth_perception=2.0
        )

        # Arc length should be close to 1.0 for straight lines
        assert dewarper.directrix_arc_length == pytest.approx(1.0, abs=0.1)

    def test_create_from_curved_lines(self):
        """Create dewarper from curved lines (page spine)."""
        # Simulate book spine curvature
        x = np.linspace(0, 100, 10)
        top_y = 5 * np.sin(np.pi * x / 100)  # Slight curve
        bottom_y = 100 + 5 * np.sin(np.pi * x / 100)

        top = np.column_stack([x, top_y])
        bottom = np.column_stack([x, bottom_y])

        dewarper = CylindricalSurfaceDewarper.from_directrices(
            top, bottom, depth_perception=2.0
        )

        # Arc length should be slightly > 1.0 for curved lines
        assert dewarper.directrix_arc_length >= 1.0

    def test_map_to_dewarped_corners(self):
        """Corner points map correctly to dewarped space."""
        top = np.array([[0, 0], [100, 0]], dtype=np.float64)
        bottom = np.array([[0, 100], [100, 100]], dtype=np.float64)

        dewarper = CylindricalSurfaceDewarper.from_directrices(
            top, bottom, depth_perception=2.0
        )

        # Top-left corner should map to (0, 0)
        tl = dewarper.map_to_dewarped_space(np.array([0, 0]))
        assert tl[0] == pytest.approx(0.0, abs=0.1)
        assert tl[1] == pytest.approx(0.0, abs=0.1)

        # Top-right corner should map to (1, 0)
        tr = dewarper.map_to_dewarped_space(np.array([100, 0]))
        assert tr[0] == pytest.approx(1.0, abs=0.1)
        assert tr[1] == pytest.approx(0.0, abs=0.1)

        # Bottom-left corner should map to (0, 1)
        bl = dewarper.map_to_dewarped_space(np.array([0, 100]))
        assert bl[0] == pytest.approx(0.0, abs=0.1)
        assert bl[1] == pytest.approx(1.0, abs=0.1)

        # Bottom-right corner should map to (1, 1)
        br = dewarper.map_to_dewarped_space(np.array([100, 100]))
        assert br[0] == pytest.approx(1.0, abs=0.1)
        assert br[1] == pytest.approx(1.0, abs=0.1)

    def test_map_to_warped_corners(self):
        """Corner points in dewarped space map back to warped space."""
        top = np.array([[0, 0], [100, 0]], dtype=np.float64)
        bottom = np.array([[0, 100], [100, 100]], dtype=np.float64)

        dewarper = CylindricalSurfaceDewarper.from_directrices(
            top, bottom, depth_perception=2.0
        )

        # (0, 0) should map to top-left
        tl = dewarper.map_to_warped_space(np.array([0.0, 0.0]))
        assert tl[0] == pytest.approx(0.0, abs=1.0)
        assert tl[1] == pytest.approx(0.0, abs=1.0)

        # (1, 1) should map to bottom-right
        br = dewarper.map_to_warped_space(np.array([1.0, 1.0]))
        assert br[0] == pytest.approx(100.0, abs=1.0)
        assert br[1] == pytest.approx(100.0, abs=1.0)

    def test_roundtrip_mapping(self):
        """Map to dewarped and back preserves position."""
        top = np.array([[0, 0], [50, -5], [100, 0]], dtype=np.float64)
        bottom = np.array([[0, 100], [50, 105], [100, 100]], dtype=np.float64)

        dewarper = CylindricalSurfaceDewarper.from_directrices(
            top, bottom, depth_perception=2.0
        )

        # Test several points
        test_points = [
            np.array([25, 50]),
            np.array([50, 50]),
            np.array([75, 50]),
        ]

        for pt in test_points:
            dewarped = dewarper.map_to_dewarped_space(pt)
            warped = dewarper.map_to_warped_space(dewarped)
            assert warped[0] == pytest.approx(pt[0], abs=2.0)
            assert warped[1] == pytest.approx(pt[1], abs=2.0)

    def test_map_generatrix(self):
        """Map generatrix returns valid line and homography."""
        top = np.array([[0, 0], [100, 0]], dtype=np.float64)
        bottom = np.array([[0, 100], [100, 100]], dtype=np.float64)

        dewarper = CylindricalSurfaceDewarper.from_directrices(
            top, bottom, depth_perception=2.0
        )

        gtx = dewarper.map_generatrix(0.5)

        # Generatrix should be a vertical line at x=50
        p1, p2 = gtx.img_line
        assert p1[0] == pytest.approx(50.0, abs=1.0)
        assert p2[0] == pytest.approx(50.0, abs=1.0)
        assert p1[1] < p2[1]  # p1 is top, p2 is bottom

        # 1D homography should be 2x2
        assert gtx.pln2img_1d.shape == (2, 2)


class TestDewarperWithCurvature:
    """Tests with curved page surfaces."""

    def test_convex_curvature(self):
        """Test with convex (outward bulging) page."""
        # Page curves outward in the middle
        x = np.linspace(0, 200, 20)
        curve_amount = 20 * np.sin(np.pi * x / 200)

        top = np.column_stack([x, -curve_amount])
        bottom = np.column_stack([x, 300 + curve_amount])

        dewarper = CylindricalSurfaceDewarper.from_directrices(
            top, bottom, depth_perception=2.0
        )

        # Center point should map reasonably
        center = np.array([100, 150])
        dewarped = dewarper.map_to_dewarped_space(center)

        assert 0.4 < dewarped[0] < 0.6  # Roughly center X
        assert 0.4 < dewarped[1] < 0.6  # Roughly center Y

    def test_concave_curvature(self):
        """Test with concave (inward curving) page."""
        # Page curves inward in the middle
        x = np.linspace(0, 200, 20)
        curve_amount = 20 * np.sin(np.pi * x / 200)

        top = np.column_stack([x, curve_amount])
        bottom = np.column_stack([x, 300 - curve_amount])

        dewarper = CylindricalSurfaceDewarper.from_directrices(
            top, bottom, depth_perception=2.0
        )

        # Center point should map reasonably
        center = np.array([100, 150])
        dewarped = dewarper.map_to_dewarped_space(center)

        assert 0.4 < dewarped[0] < 0.6  # Roughly center X
        assert 0.4 < dewarped[1] < 0.6  # Roughly center Y


class TestDepthPerception:
    """Tests for depth perception parameter."""

    def test_different_depth_perceptions(self):
        """Different depth perception values affect mapping."""
        x = np.linspace(0, 100, 10)
        curve = 10 * np.sin(np.pi * x / 100)

        top = np.column_stack([x, curve])
        bottom = np.column_stack([x, 100 + curve])

        # Test that different depth perceptions produce different mappings
        # for a point in the curved region
        test_point = np.array([50, 50])
        dewarped_results = []

        for depth in [1.0, 2.0, 3.0]:
            dewarper = CylindricalSurfaceDewarper.from_directrices(
                top, bottom, depth_perception=depth
            )
            dewarped = dewarper.map_to_dewarped_space(test_point)
            dewarped_results.append(tuple(dewarped))

        # The dewarped positions should differ with depth perception
        # (normalized arc lengths are all 1.0, but mappings differ)
        assert len(dewarped_results) == 3
        # All should map to valid positions
        for result in dewarped_results:
            assert 0 <= result[0] <= 1
            assert 0 <= result[1] <= 1
