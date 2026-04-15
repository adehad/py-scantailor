"""Cylindrical surface dewarping.

This module provides the CylindricalSurfaceDewarper class that corrects
perspective distortion from curved page surfaces (like book spines).

SIMPLIFICATIONS FROM C++ IMPLEMENTATION
========================================

The original C++ CylindricalSurfaceDewarper
(src/dewarping/CylindricalSurfaceDewarper.cpp):
- Uses custom HomographicTransform<1,D> and
  HomographicTransform<2,D> templates
- Has a CoupledPolylinesIterator for synchronizing traversal of two polylines
- Uses ToLineProjector for projecting points onto lines
- Custom matrix solving via MatrixCalc

PYTHON SIMPLIFICATION:
- Uses existing Homography class from scantailor.math
- CoupledPolylinesIterator replaced with simpler sequential iteration
- numpy.linalg.solve for matrix operations
- Direct line projection calculations

COORDINATE SYSTEMS:
- img: Warped image coordinates (input)
- pln: Plane coordinates where corner points map to unit
  square: top-left -> (0,0), top-right -> (1,0),
  bottom-left -> (0,1), bottom-right -> (1,1)
- crv: Dewarped normalized coordinates (output)

The dewarping model assumes the page surface is a cylindrical section,
with the top and bottom curves (directrices) defining the cylinder shape.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import NDArray

from scantailor.dewarping.arc_length_mapper import ArcLengthMapper
from scantailor.dewarping.polyline_intersector import (
    PolylineIntersector,
    project_point_to_line,
)
from scantailor.math.homography import Homography


@dataclass
class Generatrix:
    """A vertical line (generatrix) mapped between coordinate systems.

    Attributes:
        img_line: The line in image coordinates as (p1, p2).
        pln2img_1d: 1D homography mapping plane Y to image line parameter.
    """

    img_line: tuple[NDArray[np.floating], NDArray[np.floating]]
    pln2img_1d: NDArray[np.float64]  # Shape (2, 2) for 1D homography


def _compute_1d_homography(pairs: list[tuple[float, float]]) -> NDArray[np.float64]:
    """Compute a 1D homography from three point correspondences.

    The 1D homography maps x -> (a*x + b) / (c*x + 1).

    Args:
        pairs: List of 3 (from, to) value pairs.

    Returns:
        2x2 matrix representing the 1D homography.
    """
    assert len(pairs) == 3

    # Set up system Ax = b for homography coefficients
    # For each pair (from, to): -from * a - b + from * to * c = -to
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)

    for i, (from_val, to_val) in enumerate(pairs):
        A[i, 0] = -from_val
        A[i, 1] = -1
        A[i, 2] = from_val * to_val
        b[i] = -to_val

    # Solve for [a, b, c] where homography is [[a, b], [c, 1]]
    coeffs = np.linalg.solve(A, b)

    return np.array([[coeffs[0], coeffs[1]], [coeffs[2], 1.0]], dtype=np.float64)


def _apply_1d_homography(H: NDArray[np.float64], x: float) -> float:
    """Apply a 1D homography to a scalar value.

    Args:
        H: 2x2 homography matrix.
        x: Input value.

    Returns:
        Transformed value.
    """
    num = H[0, 0] * x + H[0, 1]
    denom = H[1, 0] * x + H[1, 1]
    if abs(denom) < 1e-10:
        return 0.0
    return num / denom


@dataclass
class CylindricalSurfaceDewarper:
    """Corrects perspective distortion from cylindrical page surfaces.

    This class models a page as a section of a cylinder, defined by two
    curves (directrices) at the top and bottom edges. It provides mappings
    between warped image coordinates and dewarped normalized coordinates.

    Attributes:
        directrix_arc_length: Arc length of the directrix assuming unit chord.
    """

    # Required fields (no defaults)
    _pln2img: Homography = field(repr=False)
    _img2pln: Homography = field(repr=False)
    _depth_perception: float
    _pln_straight_line_y: float = field(repr=False)
    _img_directrix1_intersector: PolylineIntersector = field(repr=False)
    _img_directrix2_intersector: PolylineIntersector = field(repr=False)
    # Optional fields (have defaults)
    _directrix_arc_length: float = field(default=1.0)
    _arc_length_mapper: ArcLengthMapper = field(
        default_factory=ArcLengthMapper, repr=False
    )

    @classmethod
    def from_directrices(
        cls,
        img_directrix1: NDArray[np.float64],
        img_directrix2: NDArray[np.float64],
        depth_perception: float = 2.0,
    ) -> Self:
        """Create a dewarper from two directrix curves.

        Args:
            img_directrix1: Top curve points, shape (N, 2).
            img_directrix2: Bottom curve points, shape (M, 2).
            depth_perception: Distance from camera to the plane formed by
                the outer generatrices. Values 1-3 work well.

        Returns:
            Configured CylindricalSurfaceDewarper.
        """
        img_directrix1 = np.asarray(img_directrix1, dtype=np.float64)
        img_directrix2 = np.asarray(img_directrix2, dtype=np.float64)

        # Compute homography mapping plane coords to image coords
        pln2img = cls._calc_pln_to_img_homography(img_directrix1, img_directrix2)
        img2pln = pln2img.inverse()

        # Find the Y value in plane coords where the surface is flat
        pln_straight_line_y = cls._calc_pln_straight_line_y(
            img_directrix1, img_directrix2, pln2img, img2pln
        )

        # Create intersectors for the two directrices
        intersector1 = PolylineIntersector(img_directrix1)
        intersector2 = PolylineIntersector(img_directrix2)

        dewarper = cls(
            _pln2img=pln2img,
            _img2pln=img2pln,
            _depth_perception=depth_perception,
            _pln_straight_line_y=pln_straight_line_y,
            _img_directrix1_intersector=intersector1,
            _img_directrix2_intersector=intersector2,
        )

        # Initialize arc length mapper
        dewarper._init_arc_length_mapper(img_directrix1, img_directrix2)

        return dewarper

    @property
    def directrix_arc_length(self) -> float:
        """Arc length of a directrix assuming chord length is one."""
        return self._directrix_arc_length

    def map_generatrix(self, crv_x: float) -> Generatrix:
        """Map a generatrix from dewarped to image coordinates.

        A generatrix is a vertical line in the dewarped space.

        Args:
            crv_x: X coordinate in dewarped space [0, 1].

        Returns:
            Generatrix with image line and 1D homography for Y mapping.
        """
        # Convert dewarped X to plane X via arc length mapping
        pln_x = self._arc_length_mapper.arc_len_to_x(crv_x)

        # Get the generatrix line in image coordinates
        pln_top = np.array([pln_x, 0.0])
        pln_bottom = np.array([pln_x, 1.0])
        img_top = self._pln2img.apply_to_point(pln_top)
        img_bottom = self._pln2img.apply_to_point(pln_bottom)
        img_generatrix = (img_top, img_bottom)

        # Find intersections with the directrices
        img_directrix1_pt = self._img_directrix1_intersector.intersect(
            img_top, img_bottom
        )
        img_directrix2_pt = self._img_directrix2_intersector.intersect(
            img_top, img_bottom
        )

        # Project points onto the generatrix line
        _, proj1 = project_point_to_line(img_directrix1_pt, img_top, img_bottom)
        _, proj2 = project_point_to_line(img_directrix2_pt, img_top, img_bottom)

        # Straight line point
        pln_straight = np.array([pln_x, self._pln_straight_line_y])
        img_straight = self._pln2img.apply_to_point(pln_straight)
        _, proj_straight = project_point_to_line(img_straight, img_top, img_bottom)

        # Build 1D homography mapping plane Y to image projection
        pairs: list[tuple[float, float]]
        if (
            abs(self._pln_straight_line_y) < 0.05
            or abs(self._pln_straight_line_y - 1.0) < 0.05
        ):
            # Straight line is near an edge, use midpoint instead
            pairs = [
                (0.0, proj1),
                (1.0, proj2),
                (0.5, 0.5 * (proj1 + proj2)),
            ]
        else:
            pairs = [
                (0.0, proj1),
                (1.0, proj2),
                (self._pln_straight_line_y, proj_straight),
            ]

        pln2img_1d = _compute_1d_homography(pairs)

        return Generatrix(img_line=img_generatrix, pln2img_1d=pln2img_1d)

    def map_to_dewarped_space(self, img_pt: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform a point from warped image to dewarped coordinates.

        Args:
            img_pt: Point in warped image coordinates [x, y].

        Returns:
            Point in dewarped normalized coordinates [x, y].
        """
        img_pt = np.asarray(img_pt, dtype=np.float64)

        # Get plane X coordinate
        pln_pt = self._img2pln.apply_to_point(img_pt)
        pln_x = pln_pt[0]

        # Convert to dewarped X via arc length
        crv_x = self._arc_length_mapper.x_to_arc_len(pln_x)

        # Get the generatrix at this X
        pln_top = np.array([pln_x, 0.0])
        pln_bottom = np.array([pln_x, 1.0])
        img_top = self._pln2img.apply_to_point(pln_top)
        img_bottom = self._pln2img.apply_to_point(pln_bottom)

        # Find intersections with the directrices
        img_directrix1_pt = self._img_directrix1_intersector.intersect(
            img_top, img_bottom
        )
        img_directrix2_pt = self._img_directrix2_intersector.intersect(
            img_top, img_bottom
        )

        # Project points onto the generatrix line
        _, proj1 = project_point_to_line(img_directrix1_pt, img_top, img_bottom)
        _, proj2 = project_point_to_line(img_directrix2_pt, img_top, img_bottom)

        # Straight line point
        pln_straight = np.array([pln_x, self._pln_straight_line_y])
        img_straight = self._pln2img.apply_to_point(pln_straight)
        _, proj_straight = project_point_to_line(img_straight, img_top, img_bottom)

        # Build inverse 1D homography (image projection -> plane Y -> crv Y)
        pairs: list[tuple[float, float]]
        if (
            abs(self._pln_straight_line_y) < 0.05
            or abs(self._pln_straight_line_y - 1.0) < 0.05
        ):
            pairs = [
                (proj1, 0.0),
                (proj2, 1.0),
                (0.5 * (proj1 + proj2), 0.5),
            ]
        else:
            pairs = [
                (proj1, 0.0),
                (proj2, 1.0),
                (proj_straight, self._pln_straight_line_y),
            ]

        img2crv_1d = _compute_1d_homography(pairs)

        # Project input point onto generatrix and transform Y
        _, img_proj = project_point_to_line(img_pt, img_top, img_bottom)
        crv_y = _apply_1d_homography(img2crv_1d, img_proj)

        return np.array([crv_x, crv_y], dtype=np.float64)

    def map_to_warped_space(self, crv_pt: NDArray[np.float64]) -> NDArray[np.floating]:
        """Transform a point from dewarped to warped image coordinates.

        Args:
            crv_pt: Point in dewarped normalized coordinates [x, y].

        Returns:
            Point in warped image coordinates [x, y].
        """
        crv_pt = np.asarray(crv_pt, dtype=np.float64)

        gtx = self.map_generatrix(crv_pt[0])
        img_y = _apply_1d_homography(gtx.pln2img_1d, crv_pt[1])

        # Interpolate along the generatrix line
        img_p1, img_p2 = gtx.img_line
        return img_p1 + img_y * (img_p2 - img_p1)

    @staticmethod
    def _calc_pln_to_img_homography(
        img_directrix1: NDArray[np.float64],
        img_directrix2: NDArray[np.float64],
    ) -> Homography:
        """Compute homography mapping plane to image coordinates.

        Maps: (0,0)->top-left, (1,0)->top-right, (0,1)->bottom-left, (1,1)->bottom-right
        """
        src_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        dst_points = np.array(
            [
                img_directrix1[0],
                img_directrix1[-1],
                img_directrix2[0],
                img_directrix2[-1],
            ],
            dtype=np.float32,
        )
        return Homography.from_four_points(src_points, dst_points)

    @classmethod
    def _calc_pln_straight_line_y(
        cls,
        img_directrix1: NDArray[np.float64],
        img_directrix2: NDArray[np.float64],
        pln2img: Homography,
        img2pln: Homography,
    ) -> float:
        """Calculate the Y value in plane coords where surface is flat.

        This finds where the page surface would appear as a straight line
        if viewed from the camera position.
        """
        pln_y_accum = 0.0
        weight_accum = 0.0

        # Iterate through both polylines together
        for img_curve1_pt, img_curve2_pt, pln_x in cls._coupled_polylines_iter(
            img_directrix1, img_directrix2, pln2img, img2pln
        ):
            img_line1_pt = pln2img.apply_to_point(np.array([pln_x, 0.0]))
            img_line2_pt = pln2img.apply_to_point(np.array([pln_x, 1.0]))

            # Project points onto the curve-defined generatrix
            _, p1 = project_point_to_line(img_curve1_pt, img_curve1_pt, img_curve2_pt)
            _, p2 = project_point_to_line(img_line1_pt, img_curve1_pt, img_curve2_pt)
            _, p3 = project_point_to_line(img_line2_pt, img_curve1_pt, img_curve2_pt)
            _, p4 = project_point_to_line(img_curve2_pt, img_curve1_pt, img_curve2_pt)

            dp1 = p2 - p1  # 0
            dp2 = p4 - p3  # 1 - p3

            weight = abs(dp1 + dp2)
            if weight < 0.01:
                continue

            # Find the point where the curved and straight lines would coincide
            p0 = (p3 * dp1 + p2 * dp2) / (dp1 + dp2)

            # Get the image point at this projection
            img_pt = img_curve1_pt + p0 * (img_curve2_pt - img_curve1_pt)
            pln_pt = img2pln.apply_to_point(img_pt)

            pln_y_accum += pln_pt[1] * weight
            weight_accum += weight

        return 0.5 if weight_accum == 0 else pln_y_accum / weight_accum

    @staticmethod
    def _coupled_polylines_iter(
        img_directrix1: NDArray[np.float64],
        img_directrix2: NDArray[np.float64],
        pln2img: Homography,
        img2pln: Homography,
    ):
        """Iterate through both polylines, yielding corresponding points.

        Yields tuples of (curve1_pt, curve2_pt, pln_x) where pln_x increases
        monotonically.
        """
        i1, i2 = 0, 0
        n1, n2 = len(img_directrix1), len(img_directrix2)

        prev_pt1 = img_directrix1[0]
        prev_pt2 = img_directrix2[0]
        next_pln_x1 = 0.0
        next_pln_x2 = 0.0

        while i1 < n1 or i2 < n2:
            if i1 < n1 and (i2 >= n2 or next_pln_x1 <= next_pln_x2):
                pt1 = img_directrix1[i1]
                pln_pt1 = img2pln.apply_to_point(pt1)
                pln_x = pln_pt1[0]

                # Find corresponding point on directrix2
                pln_ptx = np.array([pln_x, pln_pt1[1] + 1])
                img_ptx = pln2img.apply_to_point(pln_ptx)

                # Intersect line (pt1, img_ptx) with segment (prev_pt2, next_pt2)
                pt2 = _line_segment_intersect(
                    pt1, img_ptx, prev_pt2, img_directrix2[min(i2, n2 - 1)]
                )
                if pt2 is None:
                    pt2 = img_directrix2[min(i2, n2 - 1)]

                yield pt1, pt2, pln_x

                prev_pt1 = pt1
                i1 += 1
                if i1 < n1:
                    next_pln_x1 = img2pln.apply_to_point(img_directrix1[i1])[0]
            else:
                pt2 = img_directrix2[i2]
                pln_pt2 = img2pln.apply_to_point(pt2)
                pln_x = pln_pt2[0]

                # Find corresponding point on directrix1
                pln_ptx = np.array([pln_x, pln_pt2[1] + 1])
                img_ptx = pln2img.apply_to_point(pln_ptx)

                # Intersect line (pt2, img_ptx) with segment (prev_pt1, next_pt1)
                pt1 = _line_segment_intersect(
                    pt2, img_ptx, prev_pt1, img_directrix1[min(i1, n1 - 1)]
                )
                if pt1 is None:
                    pt1 = img_directrix1[min(i1, n1 - 1)]

                yield pt1, pt2, pln_x

                prev_pt2 = pt2
                i2 += 1
                if i2 < n2:
                    next_pln_x2 = img2pln.apply_to_point(img_directrix2[i2])[0]

    def _init_arc_length_mapper(
        self,
        img_directrix1: NDArray[np.float64],
        img_directrix2: NDArray[np.float64],
    ) -> None:
        """Initialize the arc length mapper from directrices."""
        prev_pln_x = -np.inf

        for img_curve1_pt, img_curve2_pt, pln_x in self._coupled_polylines_iter(
            img_directrix1, img_directrix2, self._pln2img, self._img2pln
        ):
            if pln_x <= prev_pln_x:
                # S-shaped surface - skip these points
                continue

            img_line1_pt = self._pln2img.apply_to_point(np.array([pln_x, 0.0]))
            img_line2_pt = self._pln2img.apply_to_point(np.array([pln_x, 1.0]))

            # Project onto curve-defined generatrix
            _, y1 = project_point_to_line(img_line1_pt, img_curve1_pt, img_curve2_pt)
            _, y2 = project_point_to_line(img_line2_pt, img_curve1_pt, img_curve2_pt)

            # Compute elevation based on depth perception
            elevation = self._depth_perception * (1.0 - (y2 - y1))
            elevation = max(-0.5, min(0.5, elevation))

            self._arc_length_mapper.add_sample(pln_x, elevation)
            prev_pln_x = pln_x

        # Store directrix arc length and normalize
        self._directrix_arc_length = self._arc_length_mapper.total_arc_length()
        self._arc_length_mapper.normalize_range(1.0)


def _line_segment_intersect(
    line_p1: NDArray[np.floating],
    line_p2: NDArray[np.floating],
    seg_p1: NDArray[np.floating],
    seg_p2: NDArray[np.floating],
) -> NDArray[np.float64] | None:
    """Find intersection of infinite line with line segment.

    Args:
        line_p1: First point defining the infinite line.
        line_p2: Second point defining the infinite line.
        seg_p1: First endpoint of the line segment.
        seg_p2: Second endpoint of the line segment.

    Returns:
        Intersection point, or None if no intersection.
    """
    x1, y1 = line_p1
    x2, y2 = line_p2
    x3, y3 = seg_p1
    x4, y4 = seg_p2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Check if intersection is within segment (u in [0, 1])
    if u < 0 or u > 1:
        return None

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return np.array([x, y], dtype=np.float64)
