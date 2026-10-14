"""Cylindrical surface dewarping: maps a curved page onto a flat rectangle.

The page is modelled as a piece of a cylinder. Two curves, the directrices, run
along the top and the bottom of the page. A generatrix joins a point on the top
directrix to the point at the same position across the page on the bottom one.
In the dewarped output every generatrix is a vertical line, so text that follows
the curve comes out straight.

Coordinates are in three systems:

- ``img``: the warped image, which is the input.
- ``pln``: a plane in which the four corners of the surface are the unit square.
  The top-left corner is (0, 0), top-right (1, 0), bottom-left (0, 1) and
  bottom-right (1, 1).
- ``crv``: the dewarped output, normalized to the unit square.

The corner homography maps ``pln`` to ``img``. Its inverse is taken with the C++
solver's rules for entries that are almost zero.

The C++ class, ``CylindricalSurfaceDewarper`` in
``src/dewarping/CylindricalSurfaceDewarper.cpp``, is transcribed step by step
where it decides the result: the corner homography and its inverse, the walk over
both directrices in order of plane x, the height of the straight line and the arc
length mapping. The distortion model builder scores candidate curves with this
mapping, so its results match the C++ builder only as long as these steps match.

The C++ code maps one point at a time. Here many points or generatrices go
through NumPy together, and the one-dimensional homographies of the generatrices
are solved with ``numpy.linalg.solve``.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import NDArray

from scantailor.dewarping.arc_length_mapper import ArcLengthMapper
from scantailor.dewarping.line_geometry import (
    line_intersection,
    point_at,
    projection_scalar,
)
from scantailor.dewarping.polyline_intersector import PolylineIntersector
from scantailor.dewarping.spline_fit import lu_solve
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


@dataclass(frozen=True)
class GeneratrixBatch:
    """Many generatrices mapped to image coordinates.

    Attributes:
        start: First point of each image line, shape (N, 2).
        end: Second point of each image line, shape (N, 2).
        pln2img_1d: 1D homography of each line, shape (N, 2, 2).
    """

    start: NDArray[np.float64]
    end: NDArray[np.float64]
    pln2img_1d: NDArray[np.float64]


@dataclass(frozen=True)
class _DirectrixProjections:
    """Where the directrices and the straight line cross each generatrix.

    Attributes:
        img_top: Image point of each generatrix at plane y = 0, shape (N, 2).
        img_bottom: Image point at plane y = 1, shape (N, 2).
        directrix1: Position of the first directrix along each generatrix, where
            0 is ``img_top`` and 1 is ``img_bottom``; shape (N,).
        directrix2: Same for the second directrix.
        straight: Same for the straight line.
        straight_y: Plane y of the straight line, or 0.5 when it lies too close
            to an edge to pin the mapping.
        straight_near_edge: True when the midpoint stands in for the straight line.
    """

    img_top: NDArray[np.float64]
    img_bottom: NDArray[np.float64]
    directrix1: NDArray[np.float64]
    directrix2: NDArray[np.float64]
    straight: NDArray[np.float64]
    straight_y: float
    straight_near_edge: bool

    def correspondences(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Three matched (plane y, position along the generatrix) pairs per line.

        Returns:
            tuple[numpy.ndarray of numpy.float64, numpy.ndarray of numpy.float64]:
                Plane y values and
                positions along the generatrix, both of shape (N, 3).
        """
        count = len(self.directrix1)
        plane_y = np.column_stack(
            [np.zeros(count), np.ones(count), np.full(count, self.straight_y)]
        )
        third = (
            0.5 * (self.directrix1 + self.directrix2)
            if self.straight_near_edge
            else self.straight
        )
        return plane_y, np.column_stack([self.directrix1, self.directrix2, third])


def _apply_homography(
    matrix: NDArray[np.floating], points: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply a 3x3 homography to points of shape (N, 2) in double precision.

    Raises:
        ValueError: If a point maps to infinity.
    """
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-10):
        raise ValueError("Point maps to infinity")
    return np.asarray(homogeneous[:, :2] / homogeneous[:, 2:3], dtype=np.float64)


def _solve_1d_homographies(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Fit the 1D homography x -> (a*x + b) / (c*x + 1) through three pairs each.

    Args:
        source: Input values, shape (N, 3).
        target: Output values, shape (N, 3).

    Returns:
        numpy.ndarray of numpy.float64: Matrices [[a, b], [c, 1]], shape (N, 2, 2).

    Raises:
        numpy.linalg.LinAlgError: If any of the systems is singular.
    """
    system = np.stack([-source, -np.ones_like(source), source * target], axis=2)
    coefficients = np.linalg.solve(system, -target[:, :, None])[:, :, 0]
    matrices = np.empty((len(source), 2, 2), dtype=np.float64)
    matrices[:, 0, 0] = coefficients[:, 0]
    matrices[:, 0, 1] = coefficients[:, 1]
    matrices[:, 1, 0] = coefficients[:, 2]
    matrices[:, 1, 1] = 1.0
    return matrices


def _apply_1d_homographies(
    matrices: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply one 1D homography to each value; a vanishing denominator gives 0."""
    numerator = matrices[:, 0, 0] * values + matrices[:, 0, 1]
    denominator = matrices[:, 1, 0] * values + matrices[:, 1, 1]
    degenerate = np.abs(denominator) < 1e-10
    return np.where(degenerate, 0.0, numerator / np.where(degenerate, 1.0, denominator))


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
        img2pln = _invert(pln2img)

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

    def directrix_bounding_box(self) -> tuple[float, float, float, float]:
        """Bounding box of both directrices in image coordinates.

        Returns:
            tuple[float, float, float, float]: (left, top, right, bottom).
        """
        points = np.vstack(
            [
                self._img_directrix1_intersector.polyline,
                self._img_directrix2_intersector.polyline,
            ]
        )
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        return float(left), float(top), float(right), float(bottom)

    def map_generatrix(self, crv_x: float) -> Generatrix:
        """Map a generatrix from dewarped to image coordinates.

        A generatrix is a vertical line in the dewarped space.

        Args:
            crv_x: X coordinate in dewarped space [0, 1].

        Returns:
            Generatrix with image line and 1D homography for Y mapping.
        """
        batch = self.map_generatrices(np.array([crv_x], dtype=np.float64))
        return Generatrix(
            img_line=(batch.start[0], batch.end[0]),
            pln2img_1d=batch.pln2img_1d[0],
        )

    def map_generatrices(self, crv_xs: NDArray[np.floating]) -> GeneratrixBatch:
        """Map many generatrices from dewarped to image coordinates.

        Args:
            crv_xs: X coordinates in dewarped space, shape (N,).

        Returns:
            GeneratrixBatch of N image lines with their Y homographies.

        Raises:
            numpy.linalg.LinAlgError: If a generatrix has no valid homography.
        """
        crv_xs = np.asarray(crv_xs, dtype=np.float64)
        pln_x = self._arc_length_mapper.arc_len_to_x_many(crv_xs)
        projections = self._project_directrices(pln_x)
        plane_y, image_y = projections.correspondences()
        return GeneratrixBatch(
            start=projections.img_top,
            end=projections.img_bottom,
            pln2img_1d=_solve_1d_homographies(plane_y, image_y),
        )

    def map_to_dewarped_space(self, img_pt: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform a point from warped image coordinates to dewarped coordinates.

        Args:
            img_pt: Point in warped image coordinates [x, y].

        Returns:
            Point in dewarped normalized coordinates [x, y].
        """
        point = np.asarray(img_pt, dtype=np.float64)
        return self.map_points_to_dewarped_space(point[None, :])[0]

    def map_points_to_dewarped_space(
        self, img_pts: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transform many points from warped image to dewarped coordinates.

        Args:
            img_pts: Points in warped image coordinates, shape (N, 2).

        Returns:
            Points in dewarped normalized coordinates, shape (N, 2).
        """
        img_pts = np.asarray(img_pts, dtype=np.float64)
        pln_x = _apply_homography(self._img2pln.matrix, img_pts)[:, 0]
        crv_x = self._arc_length_mapper.x_to_arc_len_many(pln_x)

        projections = self._project_directrices(pln_x)
        plane_y, image_y = projections.correspondences()
        img_to_crv = _solve_1d_homographies(image_y, plane_y)

        axis = projections.img_bottom - projections.img_top
        axis_len_sq = np.sum(axis * axis, axis=1)
        along = np.sum((img_pts - projections.img_top) * axis, axis=1) / np.where(
            axis_len_sq < 1e-10, 1.0, axis_len_sq
        )
        along = np.where(axis_len_sq < 1e-10, 0.0, along)
        crv_y = _apply_1d_homographies(img_to_crv, along)
        return np.column_stack([crv_x, crv_y])

    def _project_directrices(self, pln_x: NDArray[np.float64]) -> _DirectrixProjections:
        """Intersect the generatrices at plane X positions with both directrices.

        Args:
            pln_x: Plane X coordinates, shape (N,).

        Returns:
            The generatrix end points and where the directrices and the
            straight line cross each generatrix, as positions along it.
        """
        count = len(pln_x)
        zeros = np.zeros(count)
        img_top = _apply_homography(
            self._pln2img.matrix, np.column_stack([pln_x, zeros])
        )
        img_bottom = _apply_homography(
            self._pln2img.matrix, np.column_stack([pln_x, zeros + 1.0])
        )
        img_straight = _apply_homography(
            self._pln2img.matrix,
            np.column_stack([pln_x, zeros + self._pln_straight_line_y]),
        )
        axis = img_bottom - img_top
        axis_len_sq = np.sum(axis * axis, axis=1)
        safe_len_sq = np.where(axis_len_sq < 1e-10, 1.0, axis_len_sq)

        def along(points: NDArray[np.float64]) -> NDArray[np.float64]:
            scalar = np.sum((points - img_top) * axis, axis=1) / safe_len_sq
            return np.where(axis_len_sq < 1e-10, 0.0, scalar)

        directrix1 = self._img_directrix1_intersector.intersect_many(
            img_top, img_bottom
        )
        directrix2 = self._img_directrix2_intersector.intersect_many(
            img_top, img_bottom
        )
        straight_near_edge = (
            abs(self._pln_straight_line_y) < 0.05
            or abs(self._pln_straight_line_y - 1.0) < 0.05
        )
        return _DirectrixProjections(
            img_top=img_top,
            img_bottom=img_bottom,
            directrix1=along(directrix1),
            directrix2=along(directrix2),
            straight=along(img_straight),
            straight_y=0.5 if straight_near_edge else self._pln_straight_line_y,
            straight_near_edge=straight_near_edge,
        )

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
        plane = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        image = np.array(
            [
                img_directrix1[0],
                img_directrix1[-1],
                img_directrix2[0],
                img_directrix2[-1],
            ],
            dtype=np.float64,
        )
        return _four_point_homography(plane, image)

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
        if viewed from the camera position. Every generatrix through a point
        of either directrix votes with the weight of its perspective shift.
        """
        pln_y_accum = 0.0
        weight_accum = 0.0

        for img_curve1_pt, img_curve2_pt, pln_x in _coupled_polylines(
            img_directrix1, img_directrix2, pln2img, img2pln
        ):
            generatrix = (img_curve1_pt, img_curve2_pt)
            img_line1_pt = _apply(pln2img, (pln_x, 0.0))
            img_line2_pt = _apply(pln2img, (pln_x, 1.0))
            p1 = 0.0
            p2 = projection_scalar(generatrix, img_line1_pt)
            p3 = projection_scalar(generatrix, img_line2_pt)
            p4 = 1.0
            dp1 = p2 - p1
            dp2 = p4 - p3
            weight = abs(dp1 + dp2)
            if weight < 0.01:
                continue

            p0 = (p3 * dp1 + p2 * dp2) / (dp1 + dp2)
            pln_y_accum += _apply(img2pln, point_at(generatrix, p0))[1] * weight
            weight_accum += weight

        return 0.5 if weight_accum == 0 else pln_y_accum / weight_accum

    def _init_arc_length_mapper(
        self,
        img_directrix1: NDArray[np.float64],
        img_directrix2: NDArray[np.float64],
    ) -> None:
        """Initialize the arc length mapper from directrices."""
        prev_pln_x = -np.inf

        for img_curve1_pt, img_curve2_pt, pln_x in _coupled_polylines(
            img_directrix1, img_directrix2, self._pln2img, self._img2pln
        ):
            if pln_x <= prev_pln_x:
                # S-shaped surface - skip these points
                continue

            generatrix = (img_curve1_pt, img_curve2_pt)
            y1 = projection_scalar(generatrix, _apply(self._pln2img, (pln_x, 0.0)))
            y2 = projection_scalar(generatrix, _apply(self._pln2img, (pln_x, 1.0)))

            # Compute elevation based on depth perception
            elevation = self._depth_perception * (1.0 - (y2 - y1))
            elevation = max(-0.5, min(0.5, elevation))

            self._arc_length_mapper.add_sample(pln_x, elevation)
            prev_pln_x = pln_x

        # Store directrix arc length and normalize
        self._directrix_arc_length = self._arc_length_mapper.total_arc_length()
        self._arc_length_mapper.normalize_range(1.0)


Point = tuple[float, float]


def _apply(homography: Homography, point: Point) -> Point:
    """Apply a homography to one point in double precision."""
    matrix = homography.matrix
    x, y = point
    w = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return (
        (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / w,
        (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / w,
    )


def _four_point_homography(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> Homography:
    """Homography taking four points to four points, solved like the C++ code.

    Args:
        source (numpy.ndarray of numpy.float64): Points, shape ``(4, 2)``.
        target (numpy.ndarray of numpy.float64): Their images, shape ``(4, 2)``.

    Returns:
        Homography: The transformation with its bottom right element 1.

    Raises:
        numpy.linalg.LinAlgError: If the points do not determine a homography.
    """
    system = np.zeros((8, 8))
    rhs = np.zeros(8)
    for i, ((x, y), (u, v)) in enumerate(zip(source, target, strict=True)):
        system[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y]
        rhs[2 * i] = -u
        system[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y]
        rhs[2 * i + 1] = -v
    solution = lu_solve(system, rhs)
    if solution is None:
        raise np.linalg.LinAlgError("The four points do not determine a homography")
    matrix = np.append(solution, 1.0).reshape(3, 3)
    return Homography(matrix=matrix)


def _invert(homography: Homography) -> Homography:
    """Invert a homography with the C++ solver's rules for near-zero entries."""
    matrix = homography.matrix
    columns = []
    for i in range(3):
        unit = np.zeros(3)
        unit[i] = 1.0
        column = lu_solve(np.asarray(matrix, dtype=np.float64), unit)
        if column is None:
            raise np.linalg.LinAlgError("The homography is singular")
        columns.append(column)
    return Homography(matrix=np.column_stack(columns))


def _coupled_polylines(
    directrix1: NDArray[np.float64],
    directrix2: NDArray[np.float64],
    pln2img: Homography,
    img2pln: Homography,
):
    """Walk two polylines together, in order of their plane x.

    For every vertex of either polyline the point of the other polyline on the
    same generatrix is found, extending the segment before it if need be.

    Yields:
        tuple: ``(point on the first, point on the second, plane x)``.
    """
    n1 = len(directrix1)
    n2 = len(directrix2)
    i1 = i2 = 0
    prev1 = next1 = (float(directrix1[0, 0]), float(directrix1[0, 1]))
    prev2 = next2 = (float(directrix2[0, 0]), float(directrix2[0, 1]))
    next_x1 = next_x2 = 0.0

    def vertex(polyline: NDArray[np.float64], index: int) -> Point:
        return (float(polyline[index, 0]), float(polyline[index, 1]))

    while True:
        if next_x1 < next_x2 and i1 != n1:
            pln = _apply(img2pln, next1)
            point1 = next1
            across = _apply(pln2img, (pln[0], pln[1] + 1))
            crossing = line_intersection((point1, across), (next2, prev2))
            point2 = next2 if crossing is None else crossing
            yield point1, point2, pln[0]

            i1 += 1
            if i1 != n1:
                prev1, next1 = next1, vertex(directrix1, i1)
                next_x1 = _apply(img2pln, next1)[0]
            if (
                i2 != n2
                and (next2[0] - point2[0]) ** 2 + (next2[1] - point2[1]) ** 2 < 1
            ):
                i2 += 1
                if i2 != n2:
                    prev2, next2 = next2, vertex(directrix2, i2)
                    next_x2 = _apply(img2pln, next2)[0]
        elif i2 != n2:
            pln = _apply(img2pln, next2)
            point2 = next2
            across = _apply(pln2img, (pln[0], pln[1] + 1))
            crossing = line_intersection((point2, across), (next1, prev1))
            point1 = next1 if crossing is None else crossing
            yield point1, point2, pln[0]

            i2 += 1
            if i2 != n2:
                prev2, next2 = next2, vertex(directrix2, i2)
                next_x2 = _apply(img2pln, next2)[0]
            if (
                i1 != n1
                and (next1[0] - point1[0]) ** 2 + (next1[1] - point1[1]) ** 2 < 1
            ):
                i1 += 1
                if i1 != n1:
                    prev1, next1 = next1, vertex(directrix1, i1)
                    next_x1 = _apply(img2pln, next1)[0]
        else:
            return
