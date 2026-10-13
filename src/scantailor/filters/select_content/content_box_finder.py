"""Content box detection on a 150 dpi gray page.

The page is binarised and the shadows along its edges are split off as garbage.
What is left is grouped into blocks of content separated by whitespace, text
lines are recognised among the blocks, and the box is finally trimmed from each
side while the area cut off is closer to garbage than to the remaining content.

All images in this module are boolean arrays where True is ink ("black").
Rectangles have inclusive right and bottom edges.
"""

from dataclasses import dataclass, replace

import cv2
import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy import ndimage

from scantailor.imageproc.binarize import binarize_wolf
from scantailor.imageproc.despeckle import despeckle

from .flood import limited_flood
from .max_whitespace import WhitespaceRect, WhitespaceSearch

PROCESSING_DPI = 150

_WOLF_WINDOW = 51
_WOLF_LOWER_BOUND = 50
_WOLF_COEF = 0.3

_HORIZONTAL_SHADOW_SEED = (200, 14)
_VERTICAL_SHADOW_SEED = (14, 300)
_GARBAGE_SEGMENT_LENGTH = 200

_NORMAL_DESPECKLE_LEVEL = 2.0

_MAX_HORIZONTAL_BLOCKS = 80
_MAX_VERTICAL_BLOCKS = 40
_MAX_EXTRA_BLOCKS = 10
_EXTRA_MIN_SIZE = 4
_BLOCK_TRIM_FRACTION = 5

_MIN_TEXT_HEIGHT = 6
_MIN_FILL_FACTOR = 0.22
_MAX_FILL_FACTOR = 0.65
_MIN_ASPECT = 0.6
_UEPS_PER_ASPECT = 0.4

_MAX_REMOVED_TO_KEPT_AREA = 0.3
_MIN_REMOVED_SIDE = 6
_MIN_TEXT_INFLUENCE = 0.2
_MAX_TEXT_INFLUENCE = 1.0
_TEXT_INFLUENCE_PIXELS = 5000
# Distance reported when there is nothing to measure the distance to.
_INFINITE_DISTANCE = 65536.0
_MIN_CONTENT_SIDE = 8
_MAX_THIN_ASPECT = 20
_THIN_CONTENT_WIDTH = 30


@dataclass(frozen=True)
class Rect:
    """Pixel rectangle with inclusive edges.

    Attributes:
        left (int): First column.
        top (int): First row.
        right (int): Last column.
        bottom (int): Last row.
    """

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        """Return the number of columns."""
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        """Return the number of rows."""
        return self.bottom - self.top + 1

    @property
    def is_empty(self) -> bool:
        """Return True when the rectangle covers no pixel."""
        return self.right < self.left or self.bottom < self.top

    @property
    def area(self) -> int:
        """Return the number of pixels covered, 0 for an empty rectangle."""
        return 0 if self.is_empty else self.width * self.height

    def window(self) -> tuple[slice, slice]:
        """Return the (rows, columns) slices selecting this rectangle."""
        return slice(self.top, self.bottom + 1), slice(self.left, self.right + 1)


_EMPTY = Rect(0, 0, -1, -1)


def find_content_rect(
    gray: NDArray[np.uint8],
    page_rect: Rect | None = None,
) -> Rect | None:
    """Find the content of a page scanned at 150 dpi.

    Args:
        gray (NDArray[np.uint8]): 2-D gray image, dark ink on a light page.
            Not modified.
        page_rect (Rect | None): Area of the page; everything outside it is
            treated as shadow.

    Returns:
        Rect | None: Box around the content in pixels of ``gray`` with one pixel
            of slack on every side, or None when no content was found.
    """
    ink = _binarize(gray)
    if page_rect is not None:
        outside = np.ones_like(ink)
        outside[page_rect.window()] = False
        ink |= outside
    if not ink.any():
        return None

    garbage = _find_shadows(ink)
    content = ink & ~garbage
    despeckled = _despeckle(content)

    blocks = _content_blocks(content, despeckled)
    blocks = _remove_blocks_touching_borders(blocks)
    text = _estimate_text_mask(content, blocks) & content

    rect = _bounding_box(blocks)
    vertical_garbage, horizontal_garbage = _segment_garbage(garbage)
    garbage_by_axis = (
        _Garbage(horizontal_garbage, True),
        _Garbage(vertical_garbage, False),
    )
    rect = _trim_until_stable(content, blocks, text, rect, garbage_by_axis)
    if rect.is_empty:
        return None
    return Rect(rect.left - 1, rect.top - 1, rect.right + 1, rect.bottom + 1)


def _binarize(gray: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Return the ink of a page; pixels darker than the lower bound are always ink."""
    black_on_white = binarize_wolf(
        gray, _WOLF_WINDOW, k=_WOLF_COEF, lower_bound=_WOLF_LOWER_BOUND
    )
    return black_on_white == 0


def _fill_from_seeds(
    seeds: NDArray[np.bool_],
    mask: NDArray[np.bool_],
    connectivity: int,
) -> NDArray[np.bool_]:
    """Return the connected parts of ``mask`` that contain a seed pixel."""
    count, labels = cv2.connectedComponents(
        mask.view(np.uint8), connectivity=connectivity
    )
    hit = np.zeros(count, dtype=bool)
    hit[labels[seeds & mask]] = True
    hit[0] = False
    return hit[labels]


def _open(
    image: NDArray[np.bool_],
    size: tuple[int, int],
    surroundings_are_ink: bool,
) -> NDArray[np.bool_]:
    """Morphological opening of the ink by a (width, height) brick.

    Args:
        image (NDArray[np.bool_]): Ink mask.
        size (tuple[int, int]): Brick width and height.
        surroundings_are_ink (bool): Whether pixels outside the image count as
            ink. When True, ink touching the border is not eaten away by it.

    Returns:
        NDArray[np.bool_]: New mask.
    """
    brick = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    source = image.view(np.uint8)
    if surroundings_are_ink:
        # The default borders of erosion and dilation ignore the outside.
        opened = cv2.dilate(cv2.erode(source, brick), brick)
    else:
        opened = cv2.morphologyEx(
            source,
            cv2.MORPH_OPEN,
            brick,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    opened = np.asarray(opened, dtype=np.uint8)
    return opened.view(bool)


def _find_shadows(ink: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return the long dark structures of a page, which are not content."""
    seeds = _open(ink, _HORIZONTAL_SHADOW_SEED, True) | _open(
        ink, _VERTICAL_SHADOW_SEED, True
    )
    dilated = cv2.dilate(ink.view(np.uint8), np.ones((3, 3), np.uint8)).view(bool)
    shadows = _fill_from_seeds(seeds, dilated, 8) & ink
    return _filter_shadows(shadows)


def _filter_shadows(shadows: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Give back shadows that are really text, such as white letters on black.

    Shadows touching the border of the page are genuine and always kept.
    """
    frame = np.zeros_like(shadows)
    frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = True
    touching = _fill_from_seeds(frame, shadows, 8)
    inner = shadows & ~touching
    if inner.any():
        inverse = ~inner
        outside = _fill_from_seeds(frame, inverse, 8)
        solid = ~outside | inner
        text = _estimate_text_mask(inverse, solid)
        inner &= ~_fill_from_seeds(text, inner, 8)
    return inner | touching


def _despeckle(content: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Remove the small components that are not near substantial ones."""
    black_on_white = np.where(content, 0, 255).astype(np.uint8)
    cleaned = despeckle(black_on_white, PROCESSING_DPI, _NORMAL_DESPECKLE_LEVEL)
    return cleaned == 0


def _content_blocks(
    content: NDArray[np.bool_],
    despeckled: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Group content into blocks by cutting out the largest whitespace rectangles.

    Args:
        content (NDArray[np.bool_]): Ink without shadows.
        despeckled (NDArray[np.bool_]): ``content`` without small specks.

    Returns:
        NDArray[np.bool_]: Mask of the blocks, each trimmed to its ink.
    """
    height, width = content.shape
    area_threshold = min(width, height)
    blocks = np.ones_like(content)

    _cut_whitespace(
        blocks, despeckled, (2, 1), _MAX_HORIZONTAL_BLOCKS, area_threshold, "height"
    )
    _cut_whitespace(
        blocks, despeckled, (1, 2), _MAX_VERTICAL_BLOCKS, area_threshold, "width"
    )
    blocks = _trim_blocks_to_content(despeckled, blocks)

    obstacles = content | ~blocks
    _cut_whitespace(
        blocks,
        obstacles,
        (1, 1),
        _MAX_EXTRA_BLOCKS,
        area_threshold,
        None,
        min_size=_EXTRA_MIN_SIZE,
    )
    return _trim_blocks_to_content(despeckled, blocks)


def _cut_whitespace(
    blocks: NDArray[np.bool_],
    obstacles: NDArray[np.bool_],
    powers: tuple[int, int],
    max_count: int,
    area_threshold: int,
    shrink: str | None,
    min_size: int = 1,
) -> None:
    """Clear the largest whitespace rectangles from ``blocks`` in place.

    Args:
        blocks (NDArray[np.bool_]): Block mask; modified.
        obstacles (NDArray[np.bool_]): Ink the rectangles must avoid. Not
            modified.
        powers (tuple[int, int]): Width and height exponents of the quality.
        max_count (int): Most rectangles to cut.
        area_threshold (int): Rectangles with a smaller area end the search.
        shrink (str | None): "height" or "width" shrinks the obstacle added
            after each cut by a fifth of that dimension on both sides, so later
            rectangles may overlap it. None adds the whole rectangle.
        min_size (int): Smallest width and height of a rectangle.
    """
    search = WhitespaceSearch(obstacles, *powers, min_size, min_size)
    for _ in range(max_count):
        found = search.largest()
        if found is None or found.area < area_threshold:
            return
        blocks[
            found.top : found.top + found.height, found.left : found.left + found.width
        ] = False
        _block_found(search, found, shrink)


def _block_found(
    search: WhitespaceSearch,
    found: WhitespaceRect,
    shrink: str | None,
) -> None:
    """Block a found rectangle, optionally shrunk along one axis, in the search."""
    top, left = found.top, found.left
    bottom = found.top + found.height
    right = found.left + found.width
    if shrink == "height":
        margin = found.height // _BLOCK_TRIM_FRACTION
        top, bottom = top + margin, bottom - margin
    elif shrink == "width":
        margin = found.width // _BLOCK_TRIM_FRACTION
        left, right = left + margin, right - margin
    search.block(top, left, bottom, right)


def _trim_blocks_to_content(
    content: NDArray[np.bool_],
    blocks: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Shrink every block to the bounding box of the content inside it."""
    count, labels = cv2.connectedComponents(blocks.view(np.uint8), connectivity=4)
    if count == 1:
        return np.zeros_like(blocks)
    boxes = ndimage.find_objects(np.where(content, labels, 0), max_label=count - 1)
    empty = (np.iinfo(np.int32).max, -1)
    rows = np.array([(b[0].start, b[0].stop - 1) if b else empty for b in boxes])
    cols = np.array([(b[1].start, b[1].stop - 1) if b else empty for b in boxes])
    rows = np.vstack([[empty], rows])
    cols = np.vstack([[empty], cols])

    return _keep_inside_label_boxes(np.asarray(labels, dtype=np.int32), rows, cols)


@njit(cache=True)
def _keep_inside_label_boxes(
    labels: NDArray[np.int32],
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
) -> NDArray[np.bool_]:
    """Keep the pixels of each label that lie inside that label's box.

    Args:
        labels (NDArray[np.int32]): Component labels, 0 for background.
        rows (NDArray[np.int64]): First and last row of every label's box.
        cols (NDArray[np.int64]): First and last column of every label's box.

    Returns:
        NDArray[np.bool_]: Mask of the pixels kept.
    """
    height, width = labels.shape
    kept = np.zeros((height, width), dtype=np.bool_)
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if (
                label > 0
                and rows[label, 0] <= y <= rows[label, 1]
                and cols[label, 0] <= x <= cols[label, 1]
            ):
                kept[y, x] = True
    return kept


def _remove_blocks_touching_borders(blocks: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Remove the parts of blocks that lie close to the border and connect to it.

    A plain flood from the border could remove too much, so the flood spreads
    only a quarter of the smaller image side.
    """
    height, width = blocks.shape
    reach = min(width, height) // 4
    if reach < 2:
        return blocks
    return blocks & ~limited_flood(blocks, reach - 1)


def _bounding_box(mask: NDArray[np.bool_]) -> Rect:
    """Return the bounding box of the True pixels, or an empty rectangle."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return _EMPTY
    cols = np.flatnonzero(mask.any(axis=0))
    return Rect(int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))


def _distance_to_ink(
    ink: NDArray[np.bool_],
    left: bool = False,
    right: bool = False,
    top: bool = False,
    bottom: bool = False,
) -> NDArray[np.float32]:
    """Distance from every pixel to the nearest ink pixel or selected border.

    The selected borders lie one pixel outside the image.

    Returns:
        NDArray[np.float32]: Euclidean distances; a large constant when there
            is nothing to measure to.
    """
    padded = np.zeros((ink.shape[0] + 2, ink.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = ink
    padded[0, :] = top
    padded[-1, :] = bottom
    padded[:, 0] |= left
    padded[:, -1] |= right
    if not padded.any():
        return np.full(ink.shape, _INFINITE_DISTANCE, dtype=np.float32)
    return _distance_to_zero(~padded)[1:-1, 1:-1]


def _distance_to_zero(mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Exact Euclidean distance from every True pixel to the nearest False one."""
    distance = cv2.distanceTransform(
        mask.view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    return np.asarray(distance, dtype=np.float32)


def _ultimate_eroded_points(content: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return the peaks of the distance from every pixel to the nearest ink.

    A peak is a plateau of equal distance, connected through its 8 neighbours,
    that has no higher neighbour.
    """
    padded = np.zeros((content.shape[0] + 2, content.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = content
    if not padded.any():
        return np.zeros_like(content)
    squared = np.rint(_distance_to_zero(~padded) ** 2).astype(np.float32)

    brick = np.ones((3, 3), np.uint8)
    candidates = squared == cv2.dilate(squared, brick)
    candidates[0, :] = candidates[-1, :] = candidates[:, 0] = candidates[:, -1] = False

    # A candidate beside an equal-valued non-candidate sits on a ridge, not a peak.
    other_values = np.where(candidates, -1.0, squared).astype(np.float32)
    on_ridge = candidates & (cv2.dilate(other_values, brick) == squared)
    count, labels = cv2.connectedComponents(candidates.view(np.uint8), connectivity=8)
    ridge = np.zeros(count, dtype=bool)
    ridge[labels[on_ridge]] = True
    ridge[0] = False
    return (candidates & ~ridge[labels])[1:-1, 1:-1]


def _split_text_lines(rows: NDArray[np.int64]) -> list[tuple[int, int]]:
    """Split a row histogram at its deep, balanced valleys.

    Args:
        rows (NDArray[np.int64]): Ink pixels per row of a block.

    Returns:
        list[tuple[int, int]]: Inclusive row ranges, each at least as tall as the
            shortest text line.
    """
    splittable = [(0, len(rows) - 1)]
    ranges: list[tuple[int, int]] = []
    while splittable:
        first, last = splittable.pop()
        if last - first < _MIN_TEXT_HEIGHT - 1:
            continue
        segment = rows[first : last + 1]
        valley = _best_valley(segment)
        if valley is None:
            ranges.append((first, last))
        else:
            splittable.append((first, first + valley - 1))
            splittable.append((first + valley + 1, last))
    return ranges


def _best_valley(segment: NDArray[np.int64]) -> int | None:
    """Return the index of the best split row inside a histogram segment."""
    inner = segment[1:-1]
    peak_before = np.maximum.accumulate(segment)[:-2]
    peak_after = np.maximum.accumulate(segment[::-1])[::-1][2:]
    shoulder_before = peak_before - inner
    shoulder_after = peak_after - inner
    deep = inner * 3.5 <= 0.5 * (peak_before + peak_after)
    balanced = np.minimum(shoulder_before, shoulder_after) * 20 >= np.maximum(
        shoulder_before, shoulder_after
    )
    valid = deep & (shoulder_before > 0) & (shoulder_after > 0) & balanced
    if not valid.any():
        return None
    magnitude = np.where(valid, shoulder_before + shoulder_after, -1)
    return int(np.argmax(magnitude)) + 1


def _estimate_text_mask(
    content: NDArray[np.bool_],
    blocks: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Mark the parts of blocks that look like lines of text.

    A text line is a run of rows with a moderate fill factor, as wide as it is
    tall at least, and with enough ultimate eroded points to tell it from a
    slightly skewed straight line.

    Args:
        content (NDArray[np.bool_]): Ink.
        blocks (NDArray[np.bool_]): Candidate blocks.

    Returns:
        NDArray[np.bool_]: Mask of the block rows that are text.
    """
    text = np.zeros_like(blocks)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        blocks.view(np.uint8), connectivity=4
    )
    if count == 1:
        return text
    peaks = _ultimate_eroded_points(content)
    for label in range(1, count):
        x, y, width, height = (int(v) for v in stats[label, :4])
        block = labels[y : y + height, x : x + width] == label
        ink_rows = (content[y : y + height, x : x + width] & block).sum(axis=1)
        block_rows = block.sum(axis=1)
        for first, last in _split_text_lines(ink_rows.astype(np.int64)):
            line = _text_line(ink_rows, block_rows, first, last)
            if line is None:
                continue
            top, bottom = line
            columns = slice(x, x + width)
            rows = slice(y + top, y + bottom + 1)
            if not _has_enough_peaks(blocks[rows, columns], peaks[rows, columns]):
                continue
            text[rows, columns] |= block[top : bottom + 1]
    return text


def _text_line(
    ink_rows: NDArray[np.int64],
    block_rows: NDArray[np.int64],
    first: int,
    last: int,
) -> tuple[int, int] | None:
    """Grow a text line from the ink centre of a row range, or reject the range.

    Returns:
        tuple[int, int] | None: First and last row of the line, or None.
    """
    weights = ink_rows[first : last + 1]
    total = int(weights.sum())
    if total == 0:
        return None
    weighted = int((weights * np.arange(first, last + 1)).sum())
    center = (weighted + total // 2) // total
    top = center - _MIN_TEXT_HEIGHT // 2
    bottom = top + _MIN_TEXT_HEIGHT - 1
    if top < first or bottom > last:
        return None

    black = int(ink_rows[top : bottom + 1].sum())
    whole = int(block_rows[top : bottom + 1].sum())
    widest = int(block_rows[top : bottom + 1].max())
    if not _MIN_FILL_FACTOR * whole <= black <= _MAX_FILL_FACTOR * whole:
        return None

    while (top > first or bottom < last) and abs(
        (center - top) - (bottom - center)
    ) <= 1:
        new_top = top - 1 if top > first else top
        new_bottom = bottom + 1 if bottom < last else bottom
        black += int(ink_rows[new_top] + ink_rows[new_bottom])
        whole += int(block_rows[new_top] + block_rows[new_bottom])
        if black < whole * _MIN_FILL_FACTOR:
            break
        widest = max(widest, int(block_rows[new_top]), int(block_rows[new_bottom]))
        top, bottom = new_top, new_bottom

    if black > whole * _MAX_FILL_FACTOR:
        return None
    if widest < (bottom - top + 1) * _MIN_ASPECT:
        return None
    return top, bottom


def _has_enough_peaks(
    line_blocks: NDArray[np.bool_],
    line_peaks: NDArray[np.bool_],
) -> bool:
    """Check that a line has as many separate peaks as its width calls for."""
    height, width = line_blocks.shape
    needed = int(_UEPS_PER_ASPECT * width / height)
    if needed == 0:
        return True
    found, _ = cv2.connectedComponents(
        (line_blocks & line_peaks).view(np.uint8), connectivity=4
    )
    return found - 1 >= needed


def _segment_garbage(
    garbage: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Split garbage into the parts attached to vertical and horizontal edges.

    Long straight runs of garbage seed each class and the rest of a connected
    piece of garbage joins the nearest seed. Garbage that touches no seed
    belongs to both classes.

    Returns:
        tuple[NDArray[np.bool_], NDArray[np.bool_]]: Vertical and horizontal
            garbage.
    """
    horizontal = _open(garbage, (_GARBAGE_SEGMENT_LENGTH, 1), False)
    horizontal[0, :] |= garbage[0, :]
    horizontal[-1, :] |= garbage[-1, :]
    vertical = _open(garbage, (1, _GARBAGE_SEGMENT_LENGTH), False)
    vertical[:, 0] |= garbage[:, 0]
    vertical[:, -1] |= garbage[:, -1]

    seeds = vertical | horizontal
    seed_class = np.zeros(garbage.shape, dtype=np.uint8)
    seed_class[vertical] = 1
    seed_class[horizontal] = 2
    reached = _fill_from_seeds(seeds, garbage, 8)

    nearest_row, nearest_col = ndimage.distance_transform_edt(
        ~seeds, return_distances=False, return_indices=True
    )
    nearest_class = seed_class[nearest_row, nearest_col]
    unconnected = garbage & ~reached
    return (
        (reached & (nearest_class == 1)) | unconnected,
        (reached & (nearest_class == 2)) | unconnected,
    )


class _Garbage:
    """Garbage along one axis, with the distance of every pixel to it.

    Attributes:
        image (NDArray[np.bool_]): The garbage pixels.
    """

    def __init__(self, image: NDArray[np.bool_], horizontal_edges: bool) -> None:
        self.image = image.copy()
        self._horizontal_edges = horizontal_edges
        self._distance: NDArray[np.float32] | None = None

    def add(self, content: NDArray[np.bool_], rect: Rect) -> None:
        """Add the content inside ``rect`` to the garbage."""
        window = rect.window()
        self.image[window] |= content[window]
        self._distance = None

    def distance(self) -> NDArray[np.float32]:
        """Return the distance to the garbage or to the edges it is attached to."""
        if self._distance is None:
            edges = self._horizontal_edges
            self._distance = _distance_to_ink(
                self.image,
                left=not edges,
                right=not edges,
                top=edges,
                bottom=edges,
            )
        return self._distance


def _trim_until_stable(
    content: NDArray[np.bool_],
    blocks: NDArray[np.bool_],
    text: NDArray[np.bool_],
    rect: Rect,
    garbage: tuple[_Garbage, _Garbage],
) -> Rect:
    """Trim each side in turn until no side changes any more.

    Args:
        content (NDArray[np.bool_]): Ink without shadows.
        blocks (NDArray[np.bool_]): Content blocks.
        text (NDArray[np.bool_]): Ink that belongs to text lines.
        rect (Rect): Starting rectangle.
        garbage (tuple[_Garbage, _Garbage]): Horizontal and vertical garbage;
            grows while trimming.

    Returns:
        Rect: Trimmed rectangle, empty when nothing credible remains.
    """
    horizontal_garbage, vertical_garbage = garbage
    sides = {"left", "right", "top", "bottom"}
    # Trimming one side makes the others, except the opposite one, worth a retry.
    retry_after = {
        "left": ("left", "top", "bottom"),
        "right": ("right", "top", "bottom"),
        "top": ("top", "left", "right"),
        "bottom": ("bottom", "left", "right"),
    }
    while sides and not rect.is_empty:
        for side in ("left", "right", "top", "bottom"):
            if side not in sides:
                continue
            sides.discard(side)
            used = vertical_garbage if side in ("left", "right") else horizontal_garbage
            trimmed = _trim_side(side, content, blocks, text, rect, used)
            if trimmed.is_empty:
                return _EMPTY
            if trimmed != rect:
                sides.update(retry_after[side])
            rect = trimmed
        if rect.width < _MIN_CONTENT_SIDE or rect.height < _MIN_CONTENT_SIDE:
            return _EMPTY
        if (
            rect.width < _THIN_CONTENT_WIDTH
            and rect.height > rect.width * _MAX_THIN_ASPECT
        ):
            return _EMPTY
    return rect


def _trim_side(
    side: str,
    content: NDArray[np.bool_],
    blocks: NDArray[np.bool_],
    text: NDArray[np.bool_],
    area: Rect,
    garbage: _Garbage,
) -> Rect:
    """Cut whitespace and garbage-like blocks off one side of ``area``.

    Each run of blocks followed by whitespace is a candidate for removal. When
    the candidate is kept, trimming may still be retried past it.

    Returns:
        Rect: The reduced rectangle, or ``area`` when nothing could be removed.
    """
    by_columns = side in ("left", "right")
    forward = side in ("left", "top")
    histogram = blocks[area.window()].sum(axis=0 if by_columns else 1)
    origin = area.left if by_columns else area.top
    size = len(histogram)

    start = 0 if forward else size - 1
    while 0 <= start < size:
        if forward:
            gap = _first_index(histogram[start:] == 0, start, size)
            after = _first_index(histogram[gap:] != 0, gap, size)
        else:
            gap = _last_index(histogram[: start + 1] == 0)
            after = _last_index(histogram[: gap + 1] != 0) if gap >= 0 else -1
        kept, removed = _split_area(area, side, origin + after, origin + gap)
        if kept.is_empty:
            return area
        if removed.is_empty:
            return kept
        result, retry = _trim(content, blocks, text, area, kept, removed, garbage)
        if not retry:
            return result
        start = after
    return area


def _first_index(condition: NDArray[np.bool_], offset: int, default: int) -> int:
    """Return the offset of the first True, or ``default`` when there is none."""
    hits = np.flatnonzero(condition)
    return offset + int(hits[0]) if hits.size else default


def _last_index(condition: NDArray[np.bool_]) -> int:
    """Return the index of the last True, or -1 when there is none."""
    hits = np.flatnonzero(condition)
    return int(hits[-1]) if hits.size else -1


def _split_area(
    area: Rect, side: str, kept_edge: int, gap_edge: int
) -> tuple[Rect, Rect]:
    """Split ``area`` at a whitespace gap into the part kept and the part removed.

    Args:
        area (Rect): Rectangle to split.
        side (str): Side being trimmed.
        kept_edge (int): First kept row or column for left and top, last kept
            for right and bottom.
        gap_edge (int): First whitespace row or column for left and top, last
            whitespace for right and bottom.
    """
    if side == "left":
        return replace(area, left=kept_edge), replace(area, right=gap_edge - 1)
    if side == "right":
        return replace(area, right=kept_edge), replace(area, left=gap_edge + 1)
    if side == "top":
        return replace(area, top=kept_edge), replace(area, bottom=gap_edge - 1)
    return replace(area, bottom=kept_edge), replace(area, top=gap_edge + 1)


def _relative(part: slice, whole: slice) -> slice:
    """Express ``part`` of an axis relative to the start of ``whole``."""
    return slice(part.start - whole.start, part.stop - whole.start)


def _trim(
    content: NDArray[np.bool_],
    blocks: NDArray[np.bool_],
    text: NDArray[np.bool_],
    area: Rect,
    kept: Rect,
    removed: Rect,
    garbage: _Garbage,
) -> tuple[Rect, bool]:
    """Decide whether ``removed`` is garbage, by comparing distances.

    Pixels of removed blocks are weighed by their distance to the garbage and to
    the content that would remain; text in the removed area, and cuts across the
    short direction, shift the weights towards keeping it.

    Returns:
        tuple[Rect, bool]: The rectangle after the decision and whether trimming
            may be retried further in.
    """
    too_much = removed.area > _MAX_REMOVED_TO_KEPT_AREA * kept.area
    if (
        too_much
        and removed.width >= _MIN_REMOVED_SIDE
        and removed.height >= _MIN_REMOVED_SIDE
    ):
        return area, False

    vertical_cut = kept.top == area.top and kept.bottom == area.bottom
    bias = 0.5 if vertical_cut else 0.65
    text_pixels = int(text[removed.window()].sum())
    if text_pixels == 0:
        bias = 0.4 if vertical_cut else 0.5
    else:
        total = int(content[removed.window()].sum())
        total += int(garbage.image[removed.window()].sum()) + 1
        influence = _MAX_TEXT_INFLUENCE
        if text_pixels < _TEXT_INFLUENCE_PIXELS:
            influence = _MIN_TEXT_INFLUENCE + (
                _MAX_TEXT_INFLUENCE - _MIN_TEXT_INFLUENCE
            ) * np.log(text_pixels) / np.log(_TEXT_INFLUENCE_PIXELS)
        bias += (1.0 - bias) * influence * text_pixels / total
        bias = min(max(bias, 0.0), 1.0)

    # Only ``area`` is measured: the remaining ink and the pixels asked about
    # both lie inside it, and a distance to ink does not depend on the rest.
    area_rows, area_columns = area.window()
    remaining = np.zeros((area.height, area.width), dtype=bool)
    kept_rows, kept_columns = kept.window()
    remaining[
        _relative(kept_rows, area_rows), _relative(kept_columns, area_columns)
    ] = content[kept_rows, kept_columns] & blocks[kept_rows, kept_columns]
    to_others = _distance_to_ink(remaining)

    removed_window = removed.window()
    in_blocks = blocks[removed_window]
    to_garbage_sum = float(garbage.distance()[removed_window][in_blocks].sum())
    removed_rows, removed_columns = removed_window
    to_others_removed = to_others[
        _relative(removed_rows, area_rows), _relative(removed_columns, area_columns)
    ]
    to_others_sum = float(to_others_removed[in_blocks].sum())
    if to_garbage_sum * bias < to_others_sum * (1.0 - bias):
        garbage.add(content, removed)
        return kept, False
    return area, bias < 0.85
