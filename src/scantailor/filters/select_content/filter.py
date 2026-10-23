"""Select Content filter implementation."""

from collections.abc import Callable, Hashable, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
from scantailor.core.models import PageId

from .content_box import ContentBox, PageBox, PhysicalSize
from .content_finder import ContentDetectionResult, find_content_box
from .dependencies import Dependencies
from .detection import ContentDetectionMode, PageDetectionMode
from .page_finder import find_page_box
from .params import Params
from .settings import Settings

_VIEW_TOLERANCE = 1e-3


def _view_change(
    stored: Sequence[Sequence[float]], current: NDArray[np.float64] | None
) -> NDArray[np.float64] | None:
    """Map from the view the boxes were drawn in to the current view.

    Args:
        stored (Sequence[Sequence[float]]): 3x3 source-to-view matrix recorded
            with the boxes; empty when unknown.
        current (NDArray[np.float64] | None): 3x3 source-to-view matrix of the
            image being processed.

    Returns:
        NDArray[np.float64] | None: 3x3 matrix taking old view pixels to new
        view pixels, or None when either view is unknown or they are the same.
    """
    if not stored or current is None:
        return None
    old = np.asarray(stored, dtype=np.float64)
    new = np.asarray(current, dtype=np.float64)
    if old.shape != (3, 3) or new.shape != (3, 3):
        return None
    if np.allclose(old, new, atol=_VIEW_TOLERANCE):
        return None
    return np.asarray(new @ np.linalg.inv(old), dtype=np.float64)


def _mapped_box(box: ContentBox, to_new_view: NDArray[np.float64]) -> ContentBox:
    """Bounding box of ``box`` after its corners are mapped into the new view."""
    corners = np.array(
        [
            [box.x, box.y, 1.0],
            [box.right, box.y, 1.0],
            [box.right, box.bottom, 1.0],
            [box.x, box.bottom, 1.0],
        ]
    )
    mapped = corners @ to_new_view.T
    xs, ys = mapped[:, 0], mapped[:, 1]
    return ContentBox(
        x=float(xs.min()),
        y=float(ys.min()),
        width=float(xs.max() - xs.min()),
        height=float(ys.max() - ys.min()),
    )


def _mapped_page_box(
    box: PageBox, to_new_view: NDArray[np.float64], width: int, height: int
) -> PageBox:
    """Map a page box into the new view and clip it to the image.

    A box that lands entirely outside the image is replaced by the image.
    """
    image = ContentBox(x=0.0, y=0.0, width=float(width), height=float(height))
    mapped = _mapped_box(box.to_content_box(), to_new_view).intersection(image)
    if mapped.is_empty():
        return PageBox.from_image_size(width, height)
    return PageBox(x=mapped.x, y=mapped.y, width=mapped.width, height=mapped.height)


def _made(
    image: NDArray[np.uint8] | Callable[[], NDArray[np.uint8]],
) -> NDArray[np.uint8]:
    """Return the image, making it first when it is given as a function."""
    return image() if callable(image) else image


def _content_within_page(params: Params) -> bool:
    """Tell whether the stored content box lies inside the stored page box."""
    if params.content_box.is_empty() or params.page_box.is_empty():
        return True
    return params.content_box.intersection(params.page_box.to_content_box()) == (
        params.content_box
    )


class Settled(NamedTuple):
    """What ``Filter.settle`` left the page with.

    Attributes:
        params (Params): The page's parameters.
        page_detected (bool): Whether the page finder ran.
        content_detected (bool): Whether the content finder ran.
    """

    params: Params
    page_detected: bool
    content_detected: bool


class Filter:
    """Select Content filter for detecting content and page boundaries.

    This filter detects the actual content area within a scanned page,
    distinguishing it from margins, shadows, and other artifacts.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the filter.

        Args:
            settings: Optional settings instance. Creates new if not provided.
        """
        self._settings = settings if settings is not None else Settings()

    @property
    def name(self) -> str:
        """Return the filter name."""
        return "Select Content"

    @property
    def settings(self) -> Settings:
        """Return the settings instance."""
        return self._settings

    def get_params(self, page_id: PageId) -> Params:
        """Get parameters for a page."""
        return self._settings.get_params(page_id)

    def set_params(self, page_id: PageId, params: Params) -> None:
        """Set parameters for a page."""
        self._settings.set_params(page_id, params)

    def is_params_set(self, page_id: PageId) -> bool:
        """Check if parameters have been set for a page."""
        return self._settings.is_params_set(page_id)

    def detect_content(
        self,
        image: NDArray[np.uint8],
        page_id: PageId,
        dpi: float = 300.0,
        store_result: bool = True,
        outside: NDArray[np.bool_] | Callable[[], NDArray[np.bool_]] | None = None,
        page_box: ContentBox | None = None,
        outside_key: Hashable | None = None,
        dbg: DebugImages | None = None,
    ) -> ContentDetectionResult:
        """Detect content boundaries in an image.

        Args:
            image: Grayscale or color image.
            page_id: Page identifier.
            dpi: Image resolution.
            store_result: Whether to store the result in settings, unless the
                page's parameters changed while it was being detected.
            outside: Mask of the pixels that have no source pixel behind them
                (corners exposed by a rotation); they are treated as shadow.
            page_box: Area of the page in pixels of ``image``.
            outside_key: Identifies a callable ``outside`` so that it is only
                called when the detection is not already remembered.
            dbg (DebugImages | None): Receives the content finder's
                intermediate images.

        Returns:
            ContentDetectionResult with detected box and confidence.
        """
        params = self.get_params(page_id)
        result = find_content_box(
            image,
            dpi=dpi,
            page_box=page_box,
            outside=outside,
            outside_key=outside_key,
            dbg=dbg,
        )

        if store_result:
            new_params = params.with_content_box(
                result.content_box, ContentDetectionMode.AUTO
            )
            self._settings.replace_params(page_id, params, new_params)

        return result

    def detect_page(
        self,
        image: NDArray[np.uint8],
        page_id: PageId,
        dpi: float = 300.0,
        store_result: bool = True,
        outside: NDArray[np.bool_] | Callable[[], NDArray[np.bool_]] | None = None,
        dbg: DebugImages | None = None,
    ) -> PageBox:
        """Detect page boundaries in an image.

        Uses the page's fine tuning setting and the expected page size and
        tolerance from the filter settings.

        Args:
            image (numpy.ndarray of numpy.uint8): Grayscale or color image.
            page_id (PageId): Page identifier.
            dpi (float): Image resolution.
            store_result (bool): Whether to store the result in settings,
                unless the page's parameters changed while it was being detected.
            outside (numpy.ndarray of numpy.bool | None): Pixels of ``image``
                with no source pixel behind them, or a function making them;
                they count as background around the paper.
            dbg (DebugImages | None): Receives the page finder's intermediate
                images.

        Returns:
            PageBox: The detected page, or the whole image when none was found.
        """
        params = self.get_params(page_id)
        page_box = find_page_box(
            image,
            dpi=dpi,
            fine_tune=params.fine_tune_corners,
            expected_size_mm=self._settings.get_page_detection_box(),
            tolerance=self._settings.get_page_detection_tolerance(),
            outside=outside() if callable(outside) else outside,
            dbg=dbg,
        )
        if page_box is None:
            h, w = image.shape[:2]
            page_box = PageBox.from_image_size(w, h)

        if store_result:
            new_params = params.with_page_box(page_box, PageDetectionMode.AUTO)
            self._settings.replace_params(page_id, params, new_params)

        return page_box

    def update_needs(
        self,
        page_id: PageId,
        page_outline: Sequence[tuple[float, float]] | None,
    ) -> tuple[bool, bool]:
        """Tell which detections the stored boxes no longer match.

        The stored boxes stay valid while the page outline and the detection
        modes they were made under are the ones now; a mode switched since,
        or an outline that moved (the split, the rotation or the page's
        trim changed), makes them stale.

        Args:
            page_id (PageId): The page.
            page_outline (Sequence[tuple[float, float]] | None): Outline of the
                page in the coordinates of the image being processed; None
                when unknown, which makes every box stale.

        Returns:
            tuple[bool, bool]: Whether the content box and whether the page box
                must be made again.
        """
        if page_outline is None or not self.is_params_set(page_id):
            return True, True
        params = self.get_params(page_id)
        current = Dependencies(
            content_detection_mode=params.content_detection_mode,
            page_detection_mode=params.page_detection_mode,
            fine_tune_corners=params.fine_tune_corners,
            rotated_page_outline=list(page_outline),
        )
        stored = params.dependencies
        return (
            stored.needs_content_update(current),
            stored.needs_page_update(current),
        )

    def process(
        self,
        image: NDArray[np.uint8] | Callable[[], NDArray[np.uint8]],
        page_id: PageId,
        dpi: float = 300.0,
        outside: NDArray[np.bool_] | Callable[[], NDArray[np.bool_]] | None = None,
        outside_key: Hashable | None = None,
        page_outline: Sequence[tuple[float, float]] | None = None,
        view_transform: NDArray[np.float64] | None = None,
        image_size: tuple[int, int] | None = None,
        dbg: DebugImages | None = None,
    ) -> Params:
        """Detect content and page boundaries where the modes ask for it.

        Takes the arguments of ``settle`` and returns only its parameters.

        Returns:
            Params: The updated parameters.
        """
        return self.settle(
            image,
            page_id,
            dpi,
            outside,
            outside_key,
            page_outline,
            view_transform,
            image_size,
            dbg,
        ).params

    def settle(
        self,
        image: NDArray[np.uint8] | Callable[[], NDArray[np.uint8]],
        page_id: PageId,
        dpi: float = 300.0,
        outside: NDArray[np.bool_] | Callable[[], NDArray[np.bool_]] | None = None,
        outside_key: Hashable | None = None,
        page_outline: Sequence[tuple[float, float]] | None = None,
        view_transform: NDArray[np.float64] | None = None,
        image_size: tuple[int, int] | None = None,
        dbg: DebugImages | None = None,
    ) -> Settled:
        """Detect content and page boundaries and tell which finders ran.

        Respects detection modes - only runs auto-detection when mode is AUTO.
        When ``page_outline`` is given, detection also runs only when the
        stored boxes no longer match the outline and modes they were made
        under (see ``update_needs``); otherwise it runs every time.
        Hand-drawn boxes stay on the content they were drawn around: when the
        view has changed since they were drawn (a split cutter or the deskew
        angle moved), each is mapped from the old view into the new one and
        then clipped.

        Args:
            image (NDArray[np.uint8] | Callable[[], NDArray[np.uint8]]):
                Grayscale or color image, or a function making it. A function
                is only called when detection must run.
            page_id (PageId): Page identifier.
            dpi (float): Image resolution.
            outside (NDArray[np.bool_] | Callable[[], NDArray[np.bool_]] | None):
                Mask of the pixels of ``image`` that have no source pixel
                behind them, or a function making it; content detection
                treats them as shadow.
            outside_key (Hashable | None): Identifies the mask a function
                ``outside`` makes, so that it is only made when detection must
                run.
            page_outline (Sequence[tuple[float, float]] | None): Outline of
                the page in the coordinates of ``image``; remembered so that a
                later change of it can be told.
            view_transform (NDArray[np.float64] | None): 3x3 matrix from the
                source image's pixels to those of ``image``; remembered so that
                hand-drawn boxes can follow a later change of view.
            image_size (tuple[int, int] | None): (width, height) of the image a
                function ``image`` makes; required with a function, ignored
                with an array.
            dbg (DebugImages | None): Receives the finders' intermediate
                images when they run.

        Returns:
            Settled: Updated parameters with detected or stored boxes, and
                which finders ran. When the page's parameters changed while it
                was processed, they are kept and returned instead of the result.

        Raises:
            ValueError: If ``image`` is a function and ``image_size`` is None.
        """
        started = self.get_params(page_id)
        params = started
        if callable(image):
            if image_size is None:
                msg = "image_size is required when image is a function"
                raise ValueError(msg)
            w, h = image_size
        else:
            h, w = image.shape[:2]
        outline = [] if page_outline is None else list(page_outline)
        to_new_view = _view_change(params.dependencies.view_transform, view_transform)
        update_content, update_page = self.update_needs(page_id, page_outline)
        page_detected = False
        content_detected = False

        # Handle page detection
        if params.is_page_auto():
            if update_page:
                page_detected = True
                page_box = self.detect_page(
                    _made(image),
                    page_id,
                    dpi=dpi,
                    store_result=False,
                    outside=outside,
                    dbg=dbg,
                )
                params = params.with_page_box(page_box, PageDetectionMode.AUTO)
                update_content = update_content or not _content_within_page(params)
        elif params.is_page_disabled():
            page_box = PageBox.from_image_size(w, h)
            params = params.with_page_box(page_box, PageDetectionMode.DISABLED)
        elif to_new_view is not None and not params.page_box.is_empty():
            page_box = _mapped_page_box(params.page_box, to_new_view, w, h)
            params = params.with_page_box(page_box)

        if (
            to_new_view is not None
            and params.is_content_manual()
            and not params.content_box.is_empty()
        ):
            params = params.with_content_box(
                _mapped_box(params.content_box, to_new_view)
            )

        # Handle content detection
        if params.is_content_auto():
            if update_content:
                content_detected = True
                result = self.detect_content(
                    _made(image),
                    page_id,
                    dpi=dpi,
                    store_result=False,
                    outside=outside,
                    page_box=params.page_box.to_content_box(),
                    outside_key=outside_key,
                    dbg=dbg,
                )
                params = params.with_content_box(
                    result.content_box, ContentDetectionMode.AUTO
                )
        elif params.is_content_disabled():
            # Use page box as content
            content_box = params.page_box.to_content_box()
            params = params.with_content_box(content_box, ContentDetectionMode.DISABLED)

        # Ensure content is within page
        params = params.ensure_content_within_page()

        # Calculate physical size
        content_size_mm = PhysicalSize.from_pixels(
            params.content_box.width, params.content_box.height, dpi
        )
        params = params.model_copy(
            update={
                "content_size_mm": content_size_mm,
                "dependencies": params.dependencies.model_copy(
                    update={
                        "content_detection_mode": params.content_detection_mode,
                        "page_detection_mode": params.page_detection_mode,
                        "fine_tune_corners": params.fine_tune_corners,
                        "rotated_page_outline": outline,
                        "view_transform": (
                            []
                            if view_transform is None
                            else np.asarray(view_transform, dtype=np.float64).tolist()
                        ),
                    }
                ),
            }
        )

        if not self._settings.replace_params(page_id, started, params):
            params = self.get_params(page_id)
        return Settled(params, page_detected, content_detected)

    def set_manual_content(self, page_id: PageId, content_box: ContentBox) -> Params:
        """Set content box manually.

        Args:
            page_id: Page identifier.
            content_box: Manual content box.

        Returns:
            Updated parameters.
        """
        params = self.get_params(page_id).with_manual_content(content_box)
        self.set_params(page_id, params)
        return params

    def set_manual_page(self, page_id: PageId, page_box: PageBox) -> Params:
        """Set page box manually.

        Args:
            page_id: Page identifier.
            page_box: Manual page box.

        Returns:
            Updated parameters.
        """
        params = self.get_params(page_id).with_manual_page(page_box)
        self.set_params(page_id, params)
        return params

    def reset_content_to_auto(self, page_id: PageId) -> None:
        """Reset content detection to automatic mode."""
        params = self.get_params(page_id).with_auto_content()
        self.set_params(page_id, params)

    def reset_page_to_auto(self, page_id: PageId) -> None:
        """Reset page detection to automatic mode."""
        params = self.get_params(page_id).with_auto_page()
        self.set_params(page_id, params)

    def disable_content_detection(self, page_id: PageId) -> None:
        """Disable content detection (use full page as content)."""
        params = self.get_params(page_id).with_disabled_content()
        self.set_params(page_id, params)

    def disable_page_detection(self, page_id: PageId) -> None:
        """Disable page detection (use full image as page)."""
        params = self.get_params(page_id).with_disabled_page()
        self.set_params(page_id, params)

    def get_content_box(self, page_id: PageId) -> ContentBox:
        """Get the content box for a page."""
        return self.get_params(page_id).content_box

    def get_page_box(self, page_id: PageId) -> PageBox:
        """Get the page box for a page."""
        return self.get_params(page_id).page_box

    def apply_to_pages(self, page_ids: list[PageId], params: Params) -> None:
        """Apply parameters to multiple pages."""
        self._settings.apply_params_to_pages(page_ids, params)

    def set_fine_tune(self, page_id: PageId, enabled: bool) -> None:
        """Set fine-tune corners option."""
        params = self.get_params(page_id).with_fine_tune(enabled)
        self.set_params(page_id, params)
