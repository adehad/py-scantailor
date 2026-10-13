"""Carry points between a warped page and the dewarped output the generator makes."""

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from scantailor.dewarping.page_dewarper import dewarper_from_polylines
from scantailor.dewarping.raster_dewarper import output_model_domain


class DewarpingPointMapper:
    """Maps points between the page as shot and the output dewarped from it.

    The mapping reproduces what ``compute_dewarp_maps`` followed by the
    post-dewarp rotation does to pixels, so a shape drawn on the page lands on
    the same pixels of the output that it covers when the output is generated.
    """

    def __init__(
        self,
        top: Sequence[tuple[float, float]],
        bottom: Sequence[tuple[float, float]],
        depth_perception: float,
        output_size: tuple[int, int],
        post_deskew_angle: float = 0.0,
    ) -> None:
        """Build the mapper of a distortion model.

        Args:
            top (Sequence[tuple[float, float]]): Top edge of the model in
                the pixels of the warped page the output is generated from.
            bottom (Sequence[tuple[float, float]]): Bottom edge, same
                space.
            depth_perception (float): The depth perception of the output.
            output_size (tuple[int, int]): Width and height of the dewarped
                output in pixels.
            post_deskew_angle (float): Rotation in degrees, positive turning the
                page clockwise, applied about the output's centre after
                dewarping.

        Raises:
            ValueError: If the edges do not form a usable surface.
        """
        dewarper = dewarper_from_polylines(top, bottom, depth_perception)
        if dewarper is None:
            raise ValueError("The distortion model does not form a surface")
        width, height = output_size
        domain_left, domain_top, domain_right, domain_bottom = output_model_domain(
            dewarper, output_size
        )
        self._dewarper = dewarper
        self._origin = np.array([domain_left, domain_top], dtype=np.float64)
        self._model_per_pixel = np.array(
            [
                (domain_right - domain_left) / width,
                (domain_bottom - domain_top) / height,
            ],
            dtype=np.float64,
        )
        radians = math.radians(post_deskew_angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        # Clockwise on a y-down grid.
        self._rotation = np.array([[cosine, -sine], [sine, cosine]])
        self._centre = np.array([width / 2.0, height / 2.0])

    def to_dewarped(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map points of the warped page to output pixels.

        Args:
            points (NDArray[np.float64]): Warped page pixels, shape (N, 2).

        Returns:
            NDArray[np.float64]: The same points in the dewarped output, (N, 2).
        """
        model = self._dewarper.map_points_to_dewarped_space(
            np.asarray(points, dtype=np.float64)
        )
        unrotated = (model - self._origin) / self._model_per_pixel
        return (unrotated - self._centre) @ self._rotation.T + self._centre

    def to_warped(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map points of the dewarped output back to the warped page.

        Args:
            points (NDArray[np.float64]): Output pixels, shape (N, 2).

        Returns:
            NDArray[np.float64]: The same points on the warped page, (N, 2).
        """
        unrotated = (
            np.asarray(points, dtype=np.float64) - self._centre
        ) @ self._rotation + self._centre
        model = self._origin + unrotated * self._model_per_pixel
        return np.array(
            [self._dewarper.map_to_warped_space(point) for point in model],
            dtype=np.float64,
        ).reshape(-1, 2)
