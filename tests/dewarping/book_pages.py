"""Generated book pages: justified text on a page bent as a cylinder of known shape.

A flat page of text is mapped onto the surface between a top and a bottom
directrix with the cylindrical surface dewarper's own mapping, so the exact
position of every flat point in the warped image is known. Nothing here uses
scans.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from scantailor.dewarping.dewarper import CylindricalSurfaceDewarper

WORDS = (
    "the quick brown fox jumps over lazy dog and keeps running far away from "
    "every single tall tree while others watch quietly near old stone wall"
).split()
DEFAULT_PAPER = 235
DEFAULT_INK = 25
DEFAULT_BACKGROUND = 45
# The cylindrical dewarper needs only a rough depth perception here.
DEPTH_PERCEPTION = 2.0
# The maps from the warped image to the flat page are sampled this coarsely and
# interpolated; they are smooth.
_MAP_STEP = 8


def justified_text_page(
    width: int = 1000,
    height: int = 1300,
    margin: int = 70,
    spacing: int = 46,
    scale: float = 1.0,
    thickness: int = 2,
    seed: int = 0,
    paper: int = DEFAULT_PAPER,
    ink: int = DEFAULT_INK,
) -> tuple[NDArray[np.uint8], list[int]]:
    """A flat page of justified lines of text.

    Args:
        width (int): Page width in pixels.
        height (int): Page height in pixels.
        margin (int): Blank margin on every side.
        spacing (int): Distance between baselines.
        scale (float): Font scale of ``cv2.putText``.
        thickness (int): Stroke thickness.
        seed (int): Seed of the word choice.
        paper (int): Gray level of the paper.
        ink (int): Gray level of the text.

    Returns:
        tuple[NDArray[np.uint8], list[int]]: The page and the baseline of every
            text line.
    """
    rng = np.random.default_rng(seed)
    page = np.full((height, width), paper, dtype=np.uint8)
    baselines = []
    text_width = width - 2 * margin
    y = margin + int(20 * scale)
    while y < height - margin:
        words: list[tuple[str, int]] = []
        used = 0
        while True:
            word = WORDS[int(rng.integers(len(WORDS)))]
            word_width = cv2.getTextSize(
                word, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
            )[0][0]
            if used + word_width + 14 * len(words) > text_width:
                break
            words.append((word, word_width))
            used += word_width
        gap = (text_width - used) / max(1, len(words) - 1)
        x = float(margin)
        for word, word_width in words:
            cv2.putText(
                page,
                word,
                (round(x), y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                ink,
                thickness,
                cv2.LINE_AA,
            )
            x += word_width + gap
        baselines.append(y)
        y += spacing
    return page, baselines


@dataclass(frozen=True)
class BentPage:
    """A flat page bent onto a known cylinder.

    Attributes:
        image (NDArray[np.uint8]): The warped image, with dark ground around the
            page.
        dewarper (CylindricalSurfaceDewarper): The mapping that flattens it.
        flat_size (tuple[int, int]): Width and height of the flat page.
        top (NDArray[np.float64]): The page's top edge in the warped image.
        bottom (NDArray[np.float64]): The page's bottom edge.
        baselines (list[int]): Baselines of the text lines on the flat page.
        margin (int): Margin of the flat page.
    """

    image: NDArray[np.uint8]
    dewarper: CylindricalSurfaceDewarper
    flat_size: tuple[int, int]
    top: NDArray[np.float64]
    bottom: NDArray[np.float64]
    baselines: list[int]
    margin: int

    def flat_to_warped(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Warped-image position of points of the flat page.

        Args:
            points (NDArray[np.float64]): Flat page pixels, shape ``(n, 2)``.

        Returns:
            NDArray[np.float64]: The same points in the warped image.
        """
        width, height = self.flat_size
        return np.array(
            [
                self.dewarper.map_to_warped_space(
                    np.array([x / width, y / height], dtype=np.float64)
                )
                for x, y in points
            ]
        )

    def warped_to_flat_height(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Height on the flat page, in pixels, of warped-image points.

        Args:
            points (NDArray[np.float64]): Warped image pixels, shape ``(n, 2)``.

        Returns:
            NDArray[np.float64]: Flat page y of every point.
        """
        return (
            self.dewarper.map_points_to_dewarped_space(points)[:, 1] * self.flat_size[1]
        )


def bend_page(
    flat: NDArray[np.uint8],
    baselines: list[int],
    margin: int,
    top: NDArray[np.float64],
    bottom: NDArray[np.float64],
    image_size: tuple[int, int],
    background: int = DEFAULT_BACKGROUND,
) -> BentPage:
    """Bend a flat page so that its top and bottom edges follow two curves.

    Args:
        flat (NDArray[np.uint8]): The flat page.
        baselines (list[int]): Baselines of its text lines.
        margin (int): Its margin.
        top (NDArray[np.float64]): Warped top edge, shape ``(n, 2)``, left to right.
        bottom (NDArray[np.float64]): Warped bottom edge, left to right.
        image_size (tuple[int, int]): Width and height of the warped image.
        background (int): Gray level around the page.

    Returns:
        BentPage: The warped page and its dewarper.
    """
    dewarper = CylindricalSurfaceDewarper.from_directrices(
        top, bottom, DEPTH_PERCEPTION
    )
    image_width, image_height = image_size
    flat_height, flat_width = flat.shape

    # Coarse samples sit at the centres of the blocks they will be stretched over.
    columns = (
        np.arange(image_width // _MAP_STEP + 2, dtype=np.float64) * _MAP_STEP
        + (_MAP_STEP - 1) / 2
    )
    rows = (
        np.arange(image_height // _MAP_STEP + 2, dtype=np.float64) * _MAP_STEP
        + (_MAP_STEP - 1) / 2
    )
    grid_x, grid_y = np.meshgrid(columns, rows)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    crv = dewarper.map_points_to_dewarped_space(points)
    map_x = (crv[:, 0] * flat_width).reshape(grid_x.shape).astype(np.float32)
    map_y = (crv[:, 1] * flat_height).reshape(grid_x.shape).astype(np.float32)
    full_x = cv2.resize(map_x, (columns.size * _MAP_STEP, rows.size * _MAP_STEP))
    full_y = cv2.resize(map_y, (columns.size * _MAP_STEP, rows.size * _MAP_STEP))
    image = cv2.remap(
        flat,
        full_x[:image_height, :image_width],
        full_y[:image_height, :image_width],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )
    return BentPage(
        image=np.asarray(image, dtype=np.uint8),
        dewarper=dewarper,
        flat_size=(flat_width, flat_height),
        top=top,
        bottom=bottom,
        baselines=baselines,
        margin=margin,
    )


def sagging_edges(
    left: float,
    right: float,
    top_y: float,
    bottom_y: float,
    top_bow: float,
    bottom_bow: float,
    count: int = 41,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Top and bottom page edges that bulge down by half a sine wave.

    Args:
        left (float): X of the page's left side.
        right (float): X of its right side.
        top_y (float): Height of the top edge at the page's sides.
        bottom_y (float): Height of the bottom edge at the sides.
        top_bow (float): How far the top edge sags in the middle; negative
            bulges it upwards.
        bottom_bow (float): The same for the bottom edge.
        count (int): Points per edge.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: The two edges.
    """
    xs = np.linspace(left, right, count)
    wave = np.sin(np.pi * (xs - left) / (right - left))
    return (
        np.column_stack([xs, top_y + top_bow * wave]),
        np.column_stack([xs, bottom_y + bottom_bow * wave]),
    )


def bent_book_page(
    bow: float = 45.0,
    squeeze: float = -25.0,
    seed: int = 0,
) -> BentPage:
    """The standard test page: 1000x1300 of text bent into a 1200x1600 scan.

    Args:
        bow (float): Sag of the top edge in the middle of the page.
        squeeze (float): Sag of the bottom edge in the middle; the page narrows
            vertically when it differs from ``bow``.
        seed (int): Seed of the text.

    Returns:
        BentPage: The page.
    """
    flat, baselines = justified_text_page(seed=seed)
    top, bottom = sagging_edges(100.0, 1100.0, 140.0, 1450.0, bow, squeeze)
    return bend_page(flat, baselines, 70, top, bottom, (1200, 1600))
