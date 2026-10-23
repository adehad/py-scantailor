"""Page Split filter implementation.

This filter is the second in the processing pipeline (after Fix Orientation).
It splits scanned images into individual pages, handling cases like:
- Single pages (no split needed)
- Two-page spreads (split down the middle)
- Pages with offcut garbage (cut off edges)
"""

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from scantailor.core.debug_images import DebugImages
from scantailor.core.models import Dpi, PageId

from .dependencies import Dependencies
from .layout_type import LayoutType
from .page_layout import PageLayout, PageLayoutType
from .page_layout_adapter import adapt_page_layout, correct_page_layout_type
from .params import Params
from .settings import Settings
from .split_finder import SplitResult, detect_split


class Settled(NamedTuple):
    """What ``Filter.settle`` left the page with.

    Attributes:
        params (Params): The page's parameters.
        detected (bool): Whether the layout estimator ran.
    """

    params: Params
    detected: bool


class Filter:
    """Page Split filter.

    This filter handles splitting scanned images into logical pages.
    It can operate in automatic mode (detecting the split point) or
    manual mode (using user-specified split lines).

    Example:
        >>> from scantailor.filters.page_split import Filter, Settings
        >>> from scantailor.core import ImageId, PageId, SubPage
        >>> from pathlib import Path
        >>> import numpy as np
        >>>
        >>> settings = Settings()
        >>> filter = Filter(settings)
        >>>
        >>> # Auto-detect split for an image
        >>> image = np.zeros((100, 200), dtype=np.uint8)
        >>> page_id = PageId(
        ...     image_id=ImageId(file_path=Path("/scan.tiff")),
        ...     sub_page=SubPage.SINGLE_PAGE,
        ... )
        >>> result = filter.detect_split(image, page_id)
    """

    name: str = "Page Split"

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the filter.

        Args:
            settings: Settings storage for per-page parameters.
                     If None, a new Settings instance is created.
        """
        self._settings = settings if settings is not None else Settings()

    @property
    def settings(self) -> Settings:
        """Get the settings storage."""
        return self._settings

    def get_params(self, page_id: PageId) -> Params:
        """Get the current parameters for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The stored parameters or defaults if not set.
        """
        return self._settings.get_params(page_id)

    def set_params(self, page_id: PageId, params: Params) -> None:
        """Set the parameters for a page.

        Args:
            page_id: The page to set parameters for.
            params: The parameters to store.
        """
        self._settings.set_params(page_id, params)

    def is_params_set(self, page_id: PageId) -> bool:
        """Check if parameters have been set for a page."""
        return self._settings.is_params_set(page_id)

    def get_layout_type(self, page_id: PageId) -> LayoutType:
        """Get the layout type for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The layout type for this page.
        """
        return self.get_params(page_id).layout_type

    def set_layout_type(self, page_id: PageId, layout_type: LayoutType) -> Params:
        """Request a layout type for a page.

        A stored layout of another type is dropped so that the next run detects
        a layout of the requested type; any other stored layout is kept.

        Args:
            page_id: The page to update.
            layout_type: The new layout type.

        Returns:
            The updated Params.
        """
        params = self.get_params(page_id)
        if params.page_layout is not None and _conflicts(
            layout_type, params.page_layout
        ):
            params = Params(layout_type=layout_type)
        else:
            params = params.with_layout_type(layout_type)
        self._settings.set_params(page_id, params)
        return params

    def retype_layout(self, page_id: PageId, layout_type: LayoutType) -> Params:
        """Change the type of the stored layout in place, keeping its cutters.

        For a single uncut page or for two pages when the split line was placed
        by hand, where detecting again would throw the line away.

        Args:
            page_id: The page to update.
            layout_type: The single-page or two-page type to switch to.

        Returns:
            The updated Params; only the type is requested when no layout is stored.
        """
        params = self.get_params(page_id)
        if params.page_layout is None:
            return self.set_layout_type(page_id, layout_type)
        layout = params.page_layout.with_type(_LAYOUT_TYPES[layout_type])
        params = params.model_copy(
            update={"layout_type": layout_type, "page_layout": layout}
        )
        self._settings.set_params(page_id, params)
        return params

    def get_page_layout(self, page_id: PageId) -> PageLayout | None:
        """Get the page layout for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The page layout or None if not detected yet.
        """
        return self.get_params(page_id).page_layout

    def detect_split(
        self,
        image: NDArray[np.uint8],
        page_id: PageId,
        dpi: Dpi | float = 300.0,
        store_result: bool = True,
        rotation_degrees: int = 0,
        bw_threshold: int | None = None,
        trim_area: NDArray[np.float64] | None = None,
        dbg: DebugImages | None = None,
    ) -> SplitResult:
        """Detect how to split an image.

        Args:
            image: The image to analyze (grayscale or color).
            page_id: The page ID to associate with the result.
            dpi: Image resolution in dots per inch; the axes may differ.
            store_result: If True, store the result in settings, unless the
                page's parameters changed while it was being detected.
            rotation_degrees: Orthogonal rotation already applied to ``image``;
                recorded so a later rotation change invalidates the result.
            bw_threshold: Gray level below which a pixel is black, when the page
                has a stored one; otherwise Otsu's threshold of the analysed part.
            trim_area: Corners, in pixels of ``image``, of the region a manual
                trim keeps. Only that region is analysed and a change of it
                invalidates the result.
            dbg (DebugImages | None): Receives the estimator's intermediate
                images.

        Returns:
            SplitResult with the detected layout, in the pixels of ``image``.
        """
        params = self.get_params(page_id)
        layout_type = params.layout_type
        trim = _trim_rect(trim_area, image.shape)

        result = detect_split(image, layout_type, dpi, bw_threshold, trim, dbg=dbg)

        if store_result:
            dependencies = self._dependencies_of(
                params.layout_type, image.shape, rotation_degrees, trim
            )
            new_params = params.with_auto_layout(result.layout).model_copy(
                update={"dependencies": dependencies}
            )
            self._settings.replace_params(page_id, params, new_params)

        return result

    def set_manual_split(
        self,
        page_id: PageId,
        split_x: float,
        width: float,
        height: float,
        rotation_degrees: int = 0,
    ) -> Params:
        """Set a manual split line for two-page layout.

        Args:
            page_id: The page to update.
            split_x: X coordinate of the split line.
            width: Width of the oriented image.
            height: Height of the oriented image.
            rotation_degrees: Orthogonal rotation applied to the image; recorded
                with the layout so processing keeps it.

        Returns:
            The updated Params.
        """
        layout = PageLayout.two_pages(width, height, split_x)
        params = self._with_manual_layout_of_type(
            page_id, layout, LayoutType.TWO_PAGES, rotation_degrees
        )
        self._settings.set_params(page_id, params)
        return params

    def set_manual_cutters(
        self,
        page_id: PageId,
        left_x: float,
        right_x: float,
        width: float,
        height: float,
        rotation_degrees: int = 0,
    ) -> Params:
        """Set manual cutter lines for single-page-cut layout.

        Args:
            page_id: The page to update.
            left_x: X coordinate of the left cutter.
            right_x: X coordinate of the right cutter.
            width: Width of the oriented image.
            height: Height of the oriented image.
            rotation_degrees: Orthogonal rotation applied to the image; recorded
                with the layout so processing keeps it.

        Returns:
            The updated Params.
        """
        layout = PageLayout.single_page_cut(width, height, left_x, right_x)
        params = self._with_manual_layout_of_type(
            page_id, layout, LayoutType.PAGE_PLUS_OFFCUT, rotation_degrees
        )
        self._settings.set_params(page_id, params)
        return params

    def set_manual_layout(
        self, page_id: PageId, layout: PageLayout, rotation_degrees: int = 0
    ) -> Params:
        """Store a layout placed by hand, keeping the page's requested type.

        Args:
            page_id: The page to update.
            layout: The layout as edited; its outline is the oriented image.
            rotation_degrees: Orthogonal rotation applied to the image.

        Returns:
            The updated Params.
        """
        params = self.get_params(page_id)
        stored = params.with_manual_layout(layout).model_copy(
            update={
                "dependencies": _dependencies_for(
                    params.dependencies, layout, params.layout_type, rotation_degrees
                )
            }
        )
        self._settings.set_params(page_id, stored)
        return stored

    def _with_manual_layout_of_type(
        self,
        page_id: PageId,
        layout: PageLayout,
        layout_type: LayoutType,
        rotation_degrees: int,
    ) -> Params:
        """Params holding a hand-placed layout, recorded as made for its own type."""
        params = self.get_params(page_id)
        dependencies = _dependencies_for(
            params.dependencies, layout, layout_type, rotation_degrees
        )
        return (
            params.with_manual_layout(layout)
            .with_layout_type(layout_type)
            .model_copy(update={"dependencies": dependencies})
        )

    def reset_to_auto(self, page_id: PageId) -> None:
        """Reset a page to auto mode.

        This clears any manual settings and reverts to auto-detection.

        Args:
            page_id: The page to reset.
        """
        params = Params(layout_type=LayoutType.AUTO_LAYOUT_TYPE)
        self._settings.set_params(page_id, params)

    def apply_layout_type(
        self,
        source: PageId,
        targets: list[PageId],
        layout_type: LayoutType,
        *,
        apply_cut: bool,
    ) -> None:
        """Give pages a layout type, optionally with the source page's cutters.

        With ``apply_cut`` the source's layout goes to every target that has one,
        adapted to that target's own outline rather than copied as coordinates.
        A target without a layout receives the source's layout as it is and
        adapts it when it is processed.

        Args:
            source: The page whose layout is copied.
            targets: The pages that receive the layout type.
            layout_type: The type to request; ``AUTO_LAYOUT_TYPE`` makes every
                target detect its own layout.
            apply_cut: Copy the source's cutters, except for ``AUTO_LAYOUT_TYPE``
                and ``SINGLE_PAGE_UNCUT`` which have none to copy.
        """
        source_params = self.get_params(source)
        copy_layout = (
            apply_cut
            and layout_type
            not in (LayoutType.AUTO_LAYOUT_TYPE, LayoutType.SINGLE_PAGE_UNCUT)
            and source_params.page_layout is not None
        )
        for target in targets:
            old = self.get_params(target)
            self.set_layout_type(target, layout_type)
            if copy_layout:
                self._settings.set_params(
                    target, self._with_copied_layout(source_params, old, layout_type)
                )

    @staticmethod
    def _with_copied_layout(
        source: Params, old: Params, layout_type: LayoutType
    ) -> Params:
        """Params of a target after the source's layout was applied to it."""
        assert source.page_layout is not None
        if old.page_layout is None:
            if _conflicts(layout_type, source.page_layout):
                return Params(layout_type=layout_type)
            return source.model_copy(update={"layout_type": layout_type})
        layout = adapt_page_layout(source.page_layout, old.page_layout.outline)
        adapted_type = layout.to_layout_type()
        return source.model_copy(
            update={
                "layout_type": adapted_type,
                "page_layout": layout,
                "dependencies": _dependencies_for(
                    old.dependencies, layout, adapted_type, 0
                ),
            }
        )

    @staticmethod
    def _dependencies_of(
        layout_type: LayoutType,
        image_shape: tuple[int, ...],
        rotation_degrees: int,
        trim: tuple[int, int, int, int] | None = None,
    ) -> Dependencies:
        height, width = image_shape[:2]
        return Dependencies(
            image_size=(width, height),
            rotation_degrees=rotation_degrees,
            layout_type=layout_type,
            trim=trim,
        )

    def clear_detection(self, page_id: PageId) -> None:
        """Forget the layout and split line mode but keep the requested type.

        The next run detects the layout again.

        Args:
            page_id: The page to reset.
        """
        layout_type = self.get_params(page_id).layout_type
        self._settings.set_params(page_id, Params(layout_type=layout_type))

    def process(
        self,
        image: NDArray[np.uint8],
        page_id: PageId,
        dpi: Dpi | float = 300.0,
        rotation_degrees: int = 0,
        bw_threshold: int | None = None,
        trim_area: NDArray[np.float64] | None = None,
        dbg: DebugImages | None = None,
    ) -> Params:
        """Make the stored layout fit the image, detecting only when it must.

        Takes the arguments of ``settle`` and returns only its parameters.

        Returns:
            Params: The updated parameters.
        """
        return self.settle(
            image, page_id, dpi, rotation_degrees, bw_threshold, trim_area, dbg
        ).params

    def settle(
        self,
        image: NDArray[np.uint8],
        page_id: PageId,
        dpi: Dpi | float = 300.0,
        rotation_degrees: int = 0,
        bw_threshold: int | None = None,
        trim_area: NDArray[np.float64] | None = None,
        dbg: DebugImages | None = None,
    ) -> Settled:
        """Make the stored layout fit the image and tell whether it detected.

        A stored layout is kept while its dependencies hold. When the image size
        or rotation changed, or the requested type differs from the one the
        layout was made for, a layout of an explicitly requested type is adapted
        to the new outline; only an automatic request detects again. A layout of
        another type than the requested one, or none, is detected.

        Args:
            image (NDArray[np.uint8]): The oriented image to process.
            page_id (PageId): The page ID.
            dpi (Dpi | float): Image resolution; the axes may differ.
            rotation_degrees (int): Orthogonal rotation applied to ``image``.
            bw_threshold (int | None): The page's stored black and white
                threshold, if any.
            trim_area (NDArray[np.float64] | None): Corners, in pixels of
                ``image``, of the region a manual trim keeps; None when there
                is no trim.
            dbg (DebugImages | None): Receives the estimator's intermediate
                images when it runs.

        Returns:
            Settled: The updated parameters and whether the estimator ran. When
                the page's parameters changed while it was processed, they are
                kept and returned instead of the result.
        """
        params = self.get_params(page_id)
        layout = params.page_layout
        requested = params.layout_type

        def detect() -> Settled:
            return Settled(
                self._detect(
                    image,
                    page_id,
                    dpi,
                    rotation_degrees,
                    bw_threshold,
                    trim_area,
                    dbg,
                ),
                detected=True,
            )

        if layout is None or _conflicts(requested, layout):
            return detect()

        current = self._dependencies_of(
            requested, image.shape, rotation_degrees, _trim_rect(trim_area, image.shape)
        )
        if params.dependencies.is_compatible_with(
            current, manual_split_line=params.is_manual()
        ):
            new_layout = correct_page_layout_type(layout)
            if new_layout.layout_type == layout.layout_type:
                return Settled(params, detected=False)
        elif requested == LayoutType.AUTO_LAYOUT_TYPE:
            return detect()
        else:
            height, width = image.shape[:2]
            outline = (0.0, 0.0, float(width), float(height))
            new_layout = adapt_page_layout(layout, outline)

        new_type = new_layout.to_layout_type()
        new_params = Params(
            layout_type=new_type,
            page_layout=new_layout,
            split_line_mode=params.split_line_mode,
            dependencies=current.model_copy(update={"layout_type": new_type}),
        )
        if self._settings.replace_params(page_id, params, new_params):
            return Settled(new_params, detected=False)
        return Settled(self.get_params(page_id), detected=False)

    def _detect(
        self,
        image: NDArray[np.uint8],
        page_id: PageId,
        dpi: Dpi | float,
        rotation_degrees: int,
        bw_threshold: int | None,
        trim_area: NDArray[np.float64] | None,
        dbg: DebugImages | None,
    ) -> Params:
        self.detect_split(
            image,
            page_id,
            dpi=dpi,
            rotation_degrees=rotation_degrees,
            bw_threshold=bw_threshold,
            trim_area=trim_area,
            dbg=dbg,
        )
        return self.get_params(page_id)

    def get_sub_page_count(self, page_id: PageId) -> int:
        """Get the number of sub-pages for an image.

        Args:
            page_id: The page to check.

        Returns:
            1 or 2 depending on the layout.
        """
        params = self.get_params(page_id)
        if params.page_layout is None:
            return 1
        return params.page_layout.num_sub_pages


def _dependencies_for(
    stored: Dependencies,
    layout: PageLayout,
    layout_type: LayoutType,
    rotation_degrees: int,
) -> Dependencies:
    """Dependencies of a layout that was not detected from pixels.

    Known dependencies keep their image size and rotation. Unknown ones are
    taken from the layout's outline, because a layout stored without them is
    thrown away on the next processing.
    """
    if not stored.is_null():
        return stored.model_copy(update={"layout_type": layout_type})
    _x, _y, width, height = layout.outline
    return Dependencies(
        image_size=(round(width), round(height)),
        rotation_degrees=rotation_degrees,
        layout_type=layout_type,
    )


def _trim_rect(
    trim_area: NDArray[np.float64] | None, image_shape: tuple[int, ...]
) -> tuple[int, int, int, int] | None:
    """Bounding box of a trim polygon as ``(x, y, width, height)`` in pixels.

    Returns:
        The box clipped to the image; None when there is no trim or it keeps
        the whole image.
    """
    if trim_area is None:
        return None
    height, width = image_shape[:2]
    left = max(0, math.floor(float(trim_area[:, 0].min())))
    top = max(0, math.floor(float(trim_area[:, 1].min())))
    right = min(width, math.ceil(float(trim_area[:, 0].max())))
    bottom = min(height, math.ceil(float(trim_area[:, 1].max())))
    if right <= left or bottom <= top:
        return None
    rect = (left, top, right - left, bottom - top)
    return None if rect == (0, 0, width, height) else rect


_LAYOUT_TYPES = {
    LayoutType.SINGLE_PAGE_UNCUT: PageLayoutType.SINGLE_PAGE_UNCUT,
    LayoutType.PAGE_PLUS_OFFCUT: PageLayoutType.SINGLE_PAGE_CUT,
    LayoutType.TWO_PAGES: PageLayoutType.TWO_PAGES,
}


def _conflicts(requested: LayoutType, layout: PageLayout) -> bool:
    """Whether a layout cannot satisfy a request; any layout satisfies auto."""
    return (
        requested != LayoutType.AUTO_LAYOUT_TYPE
        and _LAYOUT_TYPES[requested] != layout.layout_type
    )
