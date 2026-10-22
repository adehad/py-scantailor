"""The options widget and image view that belong to a filter stage.

The view of every stage but Output shows the oriented source image through the
transformation the earlier stages define, with the stage's own geometry drawn
on top. Dragging a handle stores the new geometry as manual parameters of the
current page and asks for a re-render. The views need unprocessed pixels that
the renderer's stage results no longer hold, so the source image is taken from
the decoded sources the renderer shares, decoding it off the GUI thread when it
is not there yet.
"""

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import shiboken6
from loguru import logger
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui, QtWidgets

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.app.ui.image_views.deskew import DeskewImageView
from scantailor.app.ui.image_views.fix_orientation import FixOrientationImageView
from scantailor.app.ui.image_views.output import (
    ImageViewTab,
    OutputImageView,
    ViewZone,
    ZoneKind,
)
from scantailor.app.ui.image_views.page_layout import PageLayoutImageView
from scantailor.app.ui.image_views.page_layout_state import PageLayoutState
from scantailor.app.ui.image_views.page_split import PageSplitImageView
from scantailor.app.ui.image_views.select_content import SelectContentImageView
from scantailor.app.ui.options_widgets import (
    deskew,
    fix_orientation,
    output,
    page_layout,
    page_split,
    select_content,
)
from scantailor.app.ui.source_cache import SourceCache
from scantailor.app.ui.widgets.help_icon import install_help_icons
from scantailor.core.black_on_white_estimator import is_black_on_white
from scantailor.core.content_mask import ContentMask, compute_content_mask
from scantailor.core.image_io import ImageLoadError
from scantailor.core.models import (
    Dpi,
    ImageId,
    Margins,
    OrthogonalRotation,
    PageId,
    SubPage,
)
from scantailor.core.page_geometry import (
    build_transformation,
    content_rect_of,
    transformation_dpi,
    view_dpi,
)
from scantailor.core.page_source import DpiOverrides, load_page
from scantailor.core.pipeline import OutputExtras
from scantailor.core.settings import ApplicationSettings
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation, Rect
from scantailor.core.zones import (
    FillZoneProperties,
    PictureLayer,
    PictureZoneProperties,
)
from scantailor.dewarping.point_mapper import DewarpingPointMapper
from scantailor.filters.output.despeckle import DespeckleLevel, DespeckleStrength
from scantailor.filters.output.dewarping_options import DewarpingMode
from scantailor.filters.output.generator import applied_post_deskew_angle
from scantailor.filters.output.params import (
    ColorMode,
    DistortionModelSpec,
    FillZoneSpec,
    PictureZoneSpec,
)
from scantailor.filters.output.params import Params as OutputParams
from scantailor.filters.page_layout.guide import Guide as StoredGuide
from scantailor.filters.page_layout.page_rect import adapt_content_rect, rect_size_mm
from scantailor.filters.page_split.page_layout import PageLayout
from scantailor.filters.select_content.content_box import ContentBox, PageBox
from scantailor.imageproc.geometry import rotate_orthogonal

_VIEW_CLASSES: dict[FilterStage, type[ImageViewBase]] = {
    FilterStage.FIX_ORIENTATION: FixOrientationImageView,
    FilterStage.PAGE_SPLIT: PageSplitImageView,
    FilterStage.DESKEW: DeskewImageView,
    FilterStage.SELECT_CONTENT: SelectContentImageView,
    FilterStage.PAGE_LAYOUT: PageLayoutImageView,
    FilterStage.OUTPUT: OutputImageView,
}

# Layer a newly drawn picture zone gets; it adds the zone to the picture mask.
_NEW_PICTURE_LAYER = PictureLayer.PAINTER2

_NO_DESPECKLING = cast(
    str,
    QtCore.QT_TRANSLATE_NOOP(
        "output::Task::UiUpdater",
        "Despeckling can't be done in Color / Grayscale mode.",
    ),
)


@dataclass
class _ViewState:
    """How much of a page a view shows and where, independent of its pixel size.

    Attributes:
        zoom: Widget pixels per virtual pixel.
        page_width: Width of the displayed area in virtual pixels.
        scroll: Horizontal and vertical scroll position as fractions of the
            scroll range, None for an axis that cannot scroll.
    """

    zoom: float
    page_width: float
    scroll: tuple[float | None, float | None]

    @classmethod
    def of(cls, view: ImageViewBase) -> _ViewState:
        """Capture the zoom and scroll position of ``view``."""
        return cls(
            view.zoom_level,
            view.virtual_display_area.width(),
            (
                _scroll_fraction(view.horizontalScrollBar()),
                _scroll_fraction(view.verticalScrollBar()),
            ),
        )

    def apply_to(self, view: ImageViewBase) -> None:
        """Show as much of the page, at the same place, in a view of another size."""
        width = view.virtual_display_area.width()
        if width <= 0 or self.page_width <= 0:
            return
        view.set_zoom(self.zoom * self.page_width / width)
        for bar, fraction in zip(
            (view.horizontalScrollBar(), view.verticalScrollBar()), self.scroll
        ):
            if fraction is not None and bar.maximum() > bar.minimum():
                bar.setValue(
                    bar.minimum() + round(fraction * (bar.maximum() - bar.minimum()))
                )


def _scroll_fraction(bar: QtWidgets.QScrollBar) -> float | None:
    extent = bar.maximum() - bar.minimum()
    return (bar.value() - bar.minimum()) / extent if extent > 0 else None


@dataclass
class _Source:
    """Decoded source image of a page, BGR or gray as the loader returns it."""

    image_id: ImageId
    pixels: NDArray[np.uint8]
    dpi: Dpi


class _SourceSignals(QtCore.QObject):
    loaded = QtCore.Signal(int, object, object, object)
    failed = QtCore.Signal(int, object, str)


class _SourceTask(QtCore.QRunnable):
    """Decodes one source image; touches no widget."""

    def __init__(
        self,
        generation: int,
        image_id: ImageId,
        dpi_overrides: DpiOverrides,
        signals: _SourceSignals,
        sources: SourceCache | None = None,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._generation = generation
        self._image_id = image_id
        self._dpi_overrides = dpi_overrides
        self._signals = signals
        self._sources = sources

    def run(self) -> None:
        """Decode the image, or take it from the shared sources, and report it."""
        try:
            if self._sources is None:
                page = load_page(self._image_id, self._dpi_overrides)
            else:
                page = self._sources.load(self._image_id, self._dpi_overrides)
        except (ImageLoadError, OSError) as error:
            self._report(self._signals.failed, str(error))
            return
        self._report(self._signals.loaded, page.pixels, page.resolution())

    def _report(self, signal: QtCore.SignalInstance, *args: object) -> None:
        try:
            signal.emit(self._generation, self._image_id, *args)
        except RuntimeError:
            # The panels were deleted while the image was being decoded.
            logger.debug(f"Nobody is waiting for {self._image_id.file_path} any more")


def _to_display(pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return loader pixels (BGR or gray) as RGB or gray for display."""
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        return np.ascontiguousarray(pixels[:, :, ::-1])
    return pixels


_MAPPER_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError, IndexError)


def _fill_zone_mapper(
    params: OutputParams,
    xform: ImageTransformation,
    scale: NDArray[np.float64],
    output: NDArray[np.uint8],
) -> DewarpingPointMapper | None:
    """Make the mapper between the page and its dewarped output, if it is dewarped."""
    if not params.needs_dewarping() or not params.distortion_model.is_defined():
        return None
    model = _model_in_view_pixels(params.distortion_model, xform, scale)
    try:
        return DewarpingPointMapper(
            model.top,
            model.bottom,
            params.depth_perception,
            (output.shape[1], output.shape[0]),
            applied_post_deskew_angle(params),
        )
    except _MAPPER_ERRORS as error:
        logger.warning(
            "Fill zones are shown on the page, the model is unusable: {}", error
        )
        return None


def _model_in_view_pixels(
    model: DistortionModelSpec,
    xform: ImageTransformation,
    scale: NDArray[np.float64],
) -> DistortionModelSpec:
    """Map a stored distortion model into the pixels of the Output view."""
    return _mapped_edges(model, lambda points: xform.map_polygon(points) * scale)


def _model_in_page_pixels(
    model: DistortionModelSpec,
    xform: ImageTransformation,
    scale: NDArray[np.float64],
) -> DistortionModelSpec:
    """Map an edited model from the Output view's pixels back to stored ones."""
    return _mapped_edges(model, lambda points: xform.map_polygon_back(points / scale))


def _mapped_edges(
    model: DistortionModelSpec,
    mapping: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> DistortionModelSpec:
    def edge(
        points: tuple[tuple[float, float], ...],
    ) -> tuple[tuple[float, float], ...]:
        if not points:
            return ()
        mapped = mapping(np.asarray(points, dtype=np.float64))
        return tuple((float(x), float(y)) for x, y in mapped)

    return DistortionModelSpec(top=edge(model.top), bottom=edge(model.bottom))


def _qrect(box: ContentBox | PageBox) -> QtCore.QRectF:
    return QtCore.QRectF(box.x, box.y, box.width, box.height)


class StagePanels(QtCore.QObject):
    """Options widget and image view of one stage, bound to a stage sequence.

    The options widgets write their edits to the stage's settings themselves;
    edits made in the image view are written here. Either way this class
    reports that an edit happened.

    Signals:
        settings_changed: The user edited the current page.
        page_size_changed: The view shows another page or size. Args: size
            (QSizeF | None) in pixels at the given resolution, None when no page
            is shown; dpi (float).
        cursor_moved: The cursor moved over the view. Args: position (QPointF | None)
            in pixels from the page's top left corner, None when it left the view.
        unremove_page_requested: The user asked in the Page Split view for a
            removed half back. Args: page (PageId) of the removed half.
        source_failed: The current page's source image could not be decoded.
            Args: image_id (ImageId) of that image.
        source_decoded: The current page's source image was decoded. Args:
            image_id (ImageId), width (int) and height (int) of the decoded
            pixels, before Fix Orientation.
    """

    settings_changed = QtCore.Signal()
    page_size_changed = QtCore.Signal(object, float)
    cursor_moved = QtCore.Signal(object)
    unremove_page_requested = QtCore.Signal(object)
    source_failed = QtCore.Signal(object)
    source_decoded = QtCore.Signal(object, int, int)

    def __init__(
        self,
        stage: FilterStage,
        sequence: StageSequence,
        parent: QtCore.QObject | None = None,
    ) -> None:
        """Build the widgets of ``stage``.

        Args:
            stage (FilterStage): The stage to show.
            sequence (StageSequence): Stages whose settings the options edit.
            parent (QtCore.QObject | None): Owner of this object.
        """
        super().__init__(parent)
        self.stage = stage
        self.sequence = sequence
        self.options: QtWidgets.QWidget = self._build_options(stage, sequence)
        self.view: ImageViewBase = _VIEW_CLASSES[stage]()

        self._page: PageId | None = None
        self._source: _Source | None = None
        self._stage_image: NDArray[np.uint8] | None = None
        self._view_key: tuple[object, ...] | None = None
        self._shown_objects: tuple[object, ...] = ()
        self._output_transformation: ImageTransformation | None = None
        # Output pixels per transformation pixel; the Output stage resamples.
        self._output_scale = np.ones(2)
        # Carries fill zones between the page and the dewarped output they are
        # drawn on; None when the output is not dewarped.
        self._fill_zone_mapper: DewarpingPointMapper | None = None
        self._extras: OutputExtras | None = None
        self._fed_extras: OutputExtras | None = None
        self._fed_model: tuple[DistortionModelSpec, float] | None = None

        self._source_pool = QtCore.QThreadPool(self)
        self._source_pool.setMaxThreadCount(1)
        self._source_signals = _SourceSignals(self)
        self._source_signals.loaded.connect(self._on_source_loaded)
        self._source_signals.failed.connect(self._on_source_failed)
        self._source_generation = 0
        self._dpi_overrides: DpiOverrides = {}
        self._sources: SourceCache | None = None
        self._removed_pages: Callable[[ImageId], tuple[bool, bool]] | None = None
        self._content_mask: tuple[tuple[object, ...], ContentMask] | None = None

        self._connect_view(stage)
        self._viewport = self.view.viewport()
        self._viewport.installEventFilter(self)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Report the cursor position over the view without consuming events.

        Args:
            watched (QtCore.QObject): The filtered object, the view's viewport.
            event (QtCore.QEvent): The event.

        Returns:
            bool: Always False; the view handles the event as usual.
        """
        if watched is self._viewport:
            if event.type() == QtCore.QEvent.Type.MouseMove:
                moved = cast(QtGui.QMouseEvent, event)
                self.cursor_moved.emit(self._page_position(moved.position()))
            elif event.type() == QtCore.QEvent.Type.Leave:
                self.cursor_moved.emit(None)
        return False

    def _page_scale(self) -> NDArray[np.float64]:
        """Return view pixels per page pixel along x and y."""
        if isinstance(self.view, OutputImageView):
            return self._output_scale
        return np.ones(2)

    def _page_position(self, widget_position: QtCore.QPointF) -> QtCore.QPointF:
        """Map a viewport position to pixels from the page's top left corner."""
        view = self.view
        origin = view._virtual_crop_area.boundingRect().topLeft()
        offset = view._widget_to_virtual.map(widget_position) - origin
        scale_x, scale_y = self._page_scale()
        return QtCore.QPointF(offset.x() / scale_x, offset.y() / scale_y)

    def _announce_page_size(self, dpi: float) -> None:
        size = self.view._virtual_crop_area.boundingRect().size()
        scale_x, scale_y = self._page_scale()
        self.page_size_changed.emit(
            QtCore.QSizeF(size.width() / scale_x, size.height() / scale_y), dpi
        )

    def set_source_cache(self, sources: SourceCache) -> None:
        """Take decoded sources from a cache that other users share.

        Args:
            sources (SourceCache): The cache; applies to sources requested from
                now on.
        """
        self._sources = sources

    def set_removed_pages_provider(
        self, provider: Callable[[ImageId], tuple[bool, bool]] | None
    ) -> None:
        """Tell the Page Split view which halves of an image the project dropped.

        Args:
            provider (Callable[[ImageId], tuple[bool, bool]] | None): Returns
                whether the left and the right half of an image were removed.
        """
        self._removed_pages = provider

    def set_dpi_overrides(self, dpi_overrides: DpiOverrides) -> None:
        """Use the project's resolutions for sources decoded from now on.

        Args:
            dpi_overrides (DpiOverrides): Snapshot that is not mutated afterwards,
                because source decoding reads it on a worker thread.
        """
        if dpi_overrides != self._dpi_overrides:
            self._source = None
        self._dpi_overrides = dpi_overrides

    def _build_options(
        self, stage: FilterStage, sequence: StageSequence
    ) -> QtWidgets.QWidget:
        forward = self._on_option_edited
        match stage:
            case FilterStage.FIX_ORIENTATION:
                widget = fix_orientation.OptionsWidget(sequence.fix_orientation_filter)
                widget.rotation_changed.connect(forward)
                widget.trim_changed.connect(forward)
            case FilterStage.PAGE_SPLIT:
                widget = page_split.OptionsWidget(sequence.page_split_filter)
                widget.layout_type_changed.connect(forward)
                widget.split_mode_changed.connect(forward)
            case FilterStage.DESKEW:
                widget = deskew.OptionsWidget(sequence.deskew_filter)
                widget.angle_changed.connect(forward)
                widget.oblique_changed.connect(forward)
            case FilterStage.SELECT_CONTENT:
                widget = select_content.OptionsWidget(sequence.select_content_filter)
                widget.page_mode_changed.connect(forward)
                widget.content_mode_changed.connect(forward)
                widget.fine_tune_changed.connect(forward)
                widget.page_box_changed.connect(forward)
            case FilterStage.PAGE_LAYOUT:
                widget = page_layout.OptionsWidget(sequence.page_layout_filter)
                widget.margins_changed.connect(forward)
                widget.alignment_changed.connect(forward)
            case FilterStage.OUTPUT:
                widget = output.OptionsWidget(sequence.output_filter)
                widget.params_changed.connect(forward)
        install_help_icons(widget)
        return widget

    def _connect_view(self, stage: FilterStage) -> None:
        view = self.view
        if isinstance(view, FixOrientationImageView):
            view.rotation_requested.connect(self._on_rotation_requested)
        elif isinstance(view, PageSplitImageView):
            view.page_layout_set_locally.connect(self._on_page_layout_set)
            view.unremove_requested.connect(self._on_unremove_requested)
        elif isinstance(view, DeskewImageView):
            view.manual_deskew_angle_set.connect(self._on_deskew_angle_set)
            view.manual_oblique_angle_set.connect(self._on_oblique_angle_set)
        elif isinstance(view, SelectContentImageView):
            view.manual_content_rect_set.connect(self._on_content_rect_set)
            view.manual_page_rect_set.connect(self._on_page_rect_set)
        elif isinstance(view, PageLayoutImageView):
            view.margins_set_locally.connect(self._on_margins_set)
            if isinstance(self.options, page_layout.OptionsWidget):
                self._keep_page_layout_state_in_settings(self.options.state)
                view.set_state(self.options.state)
        elif isinstance(view, OutputImageView):
            view.zones_changed.connect(self._on_zones_changed)
            view.distortion_model_changed.connect(self._on_distortion_model_edited)
            view.tab_changed.connect(self._on_output_tab_changed)
            if isinstance(self.options, output.OptionsWidget):
                self.options.despeckle_level_changed.connect(view.set_despeckle_level)
            view.set_tabs(list(ImageViewTab))

    def _keep_page_layout_state_in_settings(self, state: PageLayoutState) -> None:
        """Load the guides from the Page Layout settings and write changes back."""
        settings = self.sequence.page_layout_filter.settings
        state.load(
            [(g.position_mm, g.is_horizontal) for g in settings.get_guides()],
            settings.is_middle_rect_shown(),
        )

        def store_guides() -> None:
            settings.set_guides(
                StoredGuide(position_mm=g.position_mm, is_horizontal=g.is_horizontal)
                for g in state.guides
            )

        def store_middle_rect_choice() -> None:
            settings.set_middle_rect_shown(state.middle_rect_shown)

        state.guides_changed.connect(store_guides)
        state.middle_rect_shown_changed.connect(store_middle_rect_choice)

    def _on_output_tab_changed(self, tab: ImageViewTab) -> None:
        if isinstance(self.options, output.OptionsWidget):
            self.options.set_dewarping_tab_active(tab is ImageViewTab.DEWARPING)
            self.options.set_despeckle_tab_active(tab is ImageViewTab.DESPECKLING)
        view, page, xform = self.view, self._page, self._output_transformation
        if isinstance(view, OutputImageView) and page and xform is not None:
            params = self.sequence.output_filter.get_params(page)
            self._feed_distortion_model(view, params, xform, self._output_scale)

    def set_output_extras(self, extras: OutputExtras) -> None:
        """Take the pictures the Output view's tabs show besides the output page.

        They are shown once the render of the page arrives.

        Args:
            extras (OutputExtras): What the render of the current page made.
        """
        self._extras = extras
        metrics = extras.dewarp_metrics
        if isinstance(self.options, output.OptionsWidget):
            self.options.set_dewarp_metrics(metrics)
            self.options.show_black_on_white()
        if isinstance(self.view, OutputImageView):
            self.view.set_dewarp_note(None if metrics is None else metrics.describe())

    def _on_option_edited(self, *_args: object) -> None:
        # The widgets have already stored the new values, so the signal
        # arguments are not needed.
        self._refresh_geometry()
        self.settings_changed.emit()

    def set_page(self, page_id: PageId | None) -> None:
        """Make the options edit the settings of a page and show it.

        Args:
            page_id (PageId | None): The page, or None when there is none.
        """
        self._page = page_id
        self._stage_image = None
        self._clear_overlays()
        self.options.setEnabled(page_id is not None)
        if page_id is None:
            self.page_size_changed.emit(None, 0.0)
            return
        if isinstance(self.options, fix_orientation.OptionsWidget):
            self.options.set_current_image(page_id.image_id)
        else:
            self.options.set_current_page(page_id)  # type: ignore[attr-defined]
        self._request_source(page_id.image_id)

    def show_image(self, image: NDArray[np.uint8]) -> None:
        """Take the page's render, which also means its parameters are settled.

        Args:
            image (NDArray[np.uint8]): ``(H, W)`` gray or ``(H, W, 3)`` RGB pixels.
        """
        self._stage_image = image
        self._refresh_detected_options()
        self._present()

    def _refresh_detected_options(self) -> None:
        """Show what processing detected in the options that display it."""
        if self._page is not None and isinstance(
            self.options, page_split.OptionsWidget | deskew.OptionsWidget
        ):
            self.options.refresh()

    def _request_source(self, image_id: ImageId) -> None:
        if self._source is not None and self._source.image_id == image_id:
            self._present()
            return
        self._source = None
        self._source_generation += 1
        cached = (
            None
            if self._sources is None
            else self._sources.peek(image_id, self._dpi_overrides)
        )
        if cached is not None:
            self._adopt_source(image_id, cached.pixels, cached.resolution())
            return
        self._source_pool.start(
            _SourceTask(
                self._source_generation,
                image_id,
                self._dpi_overrides,
                self._source_signals,
                self._sources,
            )
        )

    @QtCore.Slot(int, object, object, object)
    def _on_source_loaded(
        self,
        generation: int,
        image_id: ImageId,
        pixels: NDArray[np.uint8],
        dpi: Dpi,
    ) -> None:
        if generation != self._source_generation or not self._widgets_alive():
            return
        self._adopt_source(image_id, pixels, dpi)

    def _widgets_alive(self) -> bool:
        """Return whether the options and view still exist.

        The widgets have no Qt parent, so their owner can delete them while a
        decode is still running; a late result must not reach them then.
        """
        return shiboken6.isValid(self.options) and shiboken6.isValid(self.view)

    def _adopt_source(
        self, image_id: ImageId, pixels: NDArray[np.uint8], dpi: Dpi
    ) -> None:
        self._source = _Source(image_id, pixels, dpi)
        height, width = pixels.shape[:2]
        self.source_decoded.emit(image_id, width, height)
        if isinstance(
            self.options, page_layout.OptionsWidget | select_content.OptionsWidget
        ):
            square = view_dpi(dpi)
            self.options.set_page_dpi(square, square)
        if isinstance(self.options, fix_orientation.OptionsWidget):
            height, width = pixels.shape[:2]
            self.options.set_image_size(width, height)
        self._present()

    @QtCore.Slot(int, object, str)
    def _on_source_failed(
        self, generation: int, image_id: ImageId, message: str
    ) -> None:
        if generation != self._source_generation or not self._widgets_alive():
            return
        logger.warning(f"Source image {image_id.file_path} cannot be shown: {message}")
        self.source_failed.emit(image_id)

    def _current(self) -> tuple[PageId, _Source] | None:
        """Return the page and its source once both are available."""
        page, source = self._page, self._source
        if page is None or source is None or source.image_id != page.image_id:
            return None
        return page, source

    def _clear_overlays(self) -> None:
        view = self.view
        if isinstance(view, PageSplitImageView):
            view.set_page_layout(None)
        elif isinstance(view, SelectContentImageView):
            view.set_content_rect(QtCore.QRectF())
            view.set_page_rect(QtCore.QRectF())
        elif isinstance(view, PageLayoutImageView):
            empty = QtCore.QRectF()
            view.set_rectangles(empty, empty, empty)
        elif isinstance(view, OutputImageView):
            view.set_zones([])
            self._output_transformation = None
            self._fill_zone_mapper = None
            self._extras = None
            self._fed_extras = None
            self._fed_model = None
            view.set_tab_image(ImageViewTab.DEWARPING, None)
            view.set_tab_image(ImageViewTab.PICTURE_ZONES, None)
            view.set_tab_image(ImageViewTab.FILL_ZONES, None)
            view.set_picture_mask(None)
            view.set_despeckle_source(None, 0, DespeckleLevel.OFF)
            view.set_tab_message(ImageViewTab.DESPECKLING, None)
            view.set_dewarp_note(None)

    def _present(self) -> None:
        """Show the page's image once the source and the render have arrived."""
        current = self._current()
        if current is None or self._stage_image is None:
            return
        page, source = current
        self._show_image_for(page, source)
        self._refresh_geometry()
        self._announce_page_size(view_dpi(source.dpi))

    def _rotation(self, page: PageId) -> OrthogonalRotation:
        return self.sequence.fix_orientation_filter.get_rotation(page.image_id)

    def _view_state_of_same_page(
        self, view: ImageViewBase, key: tuple[object, ...]
    ) -> _ViewState | None:
        """Capture a zoomed view that is about to show the same page at another size."""
        shown = self._view_key
        if shown is None or shown[:2] != key[:2] or view._fit_mode:
            return None
        return _ViewState.of(view)

    def _already_shown(self, key: tuple[object, ...], *objects: object) -> bool:
        """Record what the view now shows; tell whether it was showing it already."""
        same = (
            key == self._view_key
            and len(objects) == len(self._shown_objects)
            and all(a is b for a, b in zip(objects, self._shown_objects))
        )
        self._view_key = key
        self._shown_objects = objects
        return same

    def _show_image_for(self, page: PageId, source: _Source) -> None:
        """Give the view its image and transformation unless it already has them."""
        view = self.view
        rotation = self._rotation(page)
        height, width = source.pixels.shape[:2]

        if isinstance(view, OutputImageView):
            image = self._stage_image
            assert image is not None
            key = (self.stage, page, image.shape[:2])
            keep = key == self._view_key
            carried = None if keep else self._view_state_of_same_page(view, key)
            if self._already_shown(key, image):
                return
            view.set_image(image, keep_view=keep)
            if carried is not None:
                carried.apply_to(view)
            return

        if isinstance(view, FixOrientationImageView):
            if self._already_shown((self.stage, source.image_id), source):
                return
            view.set_rotation(rotation)
            view.set_trim(self.sequence.fix_orientation_filter.get_trim(page.image_id))
            view.set_image(
                _to_display(source.pixels),
                transformation=ImageTransformation.from_size(
                    width, height, transformation_dpi(source.dpi)
                ),
            )
            return

        if self._already_shown((self.stage, source.image_id, rotation.degrees), source):
            return
        oriented = rotate_orthogonal(source.pixels, rotation.degrees)
        oriented_size = (oriented.shape[1], oriented.shape[0])
        if isinstance(view, DeskewImageView):
            through = FilterStage.PAGE_SPLIT
            params = self.sequence.deskew_filter.get_params(page)
            view.set_deskew_angle(params.deskew_angle_deg)
            view.set_oblique_angle(params.oblique_angle_deg)
        elif isinstance(view, PageSplitImageView):
            through = FilterStage.FIX_ORIENTATION
        else:
            through = FilterStage.DESKEW
        view.set_image(
            _to_display(oriented),
            transformation=build_transformation(
                self.sequence, page, oriented_size, source.dpi, through
            ),
        )

    def _refresh_geometry(self) -> None:
        """Redraw the stage's geometry from the stored parameters."""
        current = self._current()
        if current is None:
            return
        page, source = current
        view = self.view
        sequence = self.sequence

        if isinstance(view, FixOrientationImageView):
            view.set_rotation(self._rotation(page))
            view.set_trim(sequence.fix_orientation_filter.get_trim(page.image_id))
            return
        if self._stage_image is None:
            return

        oriented_size = self._oriented_size(page, source)
        if isinstance(view, PageSplitImageView):
            if self._removed_pages is not None:
                view.set_removed_pages(*self._removed_pages(page.image_id))
            view.set_page_layout(
                sequence.page_split_filter.get_params(page).page_layout
            )
        elif isinstance(view, DeskewImageView):
            view.set_transformation(
                build_transformation(
                    sequence, page, oriented_size, source.dpi, FilterStage.PAGE_SPLIT
                )
            )
            params = sequence.deskew_filter.get_params(page)
            view.set_deskew_angle(params.deskew_angle_deg)
            view.set_oblique_angle(params.oblique_angle_deg)
        elif isinstance(view, SelectContentImageView):
            self._refresh_select_content(view, page, oriented_size, source.dpi)
        elif isinstance(view, PageLayoutImageView):
            self._refresh_page_layout(view, page, oriented_size, source.dpi)
        elif isinstance(view, OutputImageView):
            self._refresh_zones(view, page, oriented_size, source.dpi)

    def _content_mask_provider(
        self, page: PageId, source_dpi: Dpi, xform: ImageTransformation
    ) -> Callable[[], ContentMask | None]:
        """Return a function that finds the content of the page when first asked.

        The mask costs a binarisation of the whole page, so it is only made for
        a double click and then kept while the page and its transformation stay.
        """
        rotation = self._rotation(page).degrees
        key = (
            page.image_id,
            rotation,
            source_dpi,
            xform.transform.tobytes(),
            xform.resulting_pre_crop_area.tobytes(),
        )

        # The view keeps the provider, and these panels keep the view.
        owner = weakref.ref(self)

        def provide() -> ContentMask | None:
            panels = owner()
            if panels is None:
                return None
            source = panels._source
            if source is None or source.image_id != page.image_id:
                return None
            cached = panels._content_mask
            if cached is None or cached[0] != key:
                oriented = rotate_orthogonal(source.pixels, rotation)
                detection = ApplicationSettings.get_instance().polarity_detection
                black_on_white = not detection.enabled or is_black_on_white(
                    oriented, xform
                )
                cached = (key, compute_content_mask(oriented, xform, black_on_white))
                panels._content_mask = cached
            return cached[1]

        return provide

    def _oriented_size(self, page: PageId, source: _Source) -> tuple[int, int]:
        height, width = source.pixels.shape[:2]
        if self._rotation(page).degrees in (90, 270):
            return height, width
        return width, height

    def _refresh_select_content(
        self,
        view: SelectContentImageView,
        page: PageId,
        oriented_size: tuple[int, int],
        dpi: Dpi,
    ) -> None:
        params = self.sequence.select_content_filter.get_params(page)
        xform = build_transformation(
            self.sequence, page, oriented_size, dpi, FilterStage.DESKEW
        )
        view.set_transformation(xform)
        view.set_content_mask_provider(self._content_mask_provider(page, dpi, xform))
        view.set_page_rect_enabled(not params.is_page_disabled())
        view.set_page_rect(
            _qrect(params.page_box) if params.page_box.is_valid() else QtCore.QRectF()
        )
        view.set_content_rect(
            _qrect(params.content_box)
            if params.content_box.is_valid()
            else QtCore.QRectF()
        )

    def _refresh_page_layout(
        self,
        view: PageLayoutImageView,
        page: PageId,
        oriented_size: tuple[int, int],
        dpi: Dpi,
    ) -> None:
        sequence = self.sequence
        filter_ = sequence.page_layout_filter
        xform = build_transformation(
            sequence, page, oriented_size, dpi, FilterStage.DESKEW
        )
        view.set_transformation(xform)
        view.set_content_mask_provider(self._content_mask_provider(page, dpi, xform))

        mm_x, mm_y = rect_size_mm(xform, Rect(x=0, y=0, width=1, height=1))
        view.set_mm_per_pixel(mm_x, mm_y)
        view.set_aligned(filter_.get_params(page).alignment.aligned_with_others())
        content = adapt_content_rect(xform, content_rect_of(sequence, page))
        hard = filter_.get_margins(page)
        soft = filter_.calculate_soft_margins(page)

        inner = QtCore.QRectF(content.x, content.y, content.width, content.height)
        middle = _grown(inner, hard, mm_x, mm_y)
        outer = _grown(middle, soft, mm_x, mm_y)
        view.set_rectangles(inner, middle, outer)

    def _refresh_zones(
        self,
        view: OutputImageView,
        page: PageId,
        oriented_size: tuple[int, int],
        dpi: Dpi,
    ) -> None:
        xform = build_transformation(
            self.sequence, page, oriented_size, dpi, FilterStage.PAGE_LAYOUT
        )
        image = self._stage_image
        assert image is not None
        rect = xform.resulting_rect
        scale = np.array([image.shape[1] / rect.width, image.shape[0] / rect.height])
        self._output_transformation = xform
        self._output_scale = scale
        mm_x, mm_y = rect_size_mm(xform, Rect(x=0, y=0, width=1, height=1))
        view.set_mm_per_pixel(mm_x / scale[0], mm_y / scale[1])
        params = self.sequence.output_filter.get_params(page)
        mapper = _fill_zone_mapper(params, xform, scale, image)
        self._fill_zone_mapper = mapper
        zones = [
            self._view_zone(spec, ZoneKind.PICTURE, xform, scale)
            for spec in params.picture_zones
        ] + [
            self._view_zone(spec, ZoneKind.FILL, xform, scale, mapper)
            for spec in params.fill_zones
        ]
        view.set_zones(zones)
        self._feed_output_tabs(view, params, xform, scale)

    def _feed_output_tabs(
        self,
        view: OutputImageView,
        params: OutputParams,
        xform: ImageTransformation,
        scale: NDArray[np.float64],
    ) -> None:
        """Give the dewarping, despeckling and picture zone tabs their pictures."""
        extras = self._extras
        if extras is None:
            return
        if extras is not self._fed_extras:
            self._fed_extras = extras
            self._fed_model = None
            page = _to_display(extras.page)
            shown = None if view._fit_mode else _ViewState.of(view)
            view.set_tab_image(ImageViewTab.DEWARPING, page)
            view.set_tab_image(ImageViewTab.PICTURE_ZONES, page)
            view.set_tab_image(ImageViewTab.FILL_ZONES, None)
            if shown is not None:
                shown.apply_to(view)
            view.set_picture_mask(extras.picture_mask)
            view.set_despeckle_source(
                extras.despeckle_source,
                extras.despeckle_dpi,
                DespeckleStrength(params.despeckle_strength),
                self._stage_image,
            )
            view.set_tab_message(
                ImageViewTab.DESPECKLING,
                QtCore.QCoreApplication.translate(
                    "output::Task::UiUpdater", _NO_DESPECKLING
                )
                if extras.despeckle_source is None
                and params.color_mode == ColorMode.COLOR_GRAYSCALE
                else None,
            )
        self._feed_distortion_model(view, params, xform, scale)

    def _feed_distortion_model(
        self,
        view: OutputImageView,
        params: OutputParams,
        xform: ImageTransformation,
        scale: NDArray[np.float64],
    ) -> None:
        """Give the dewarping grid its model once the Dewarping tab is shown.

        Laying out the grid's curves takes a good part of a second on a large
        page, so it waits until someone opens the tab.

        Args:
            view (OutputImageView): The Output view.
            params (OutputParams): The page's output parameters.
            xform (ImageTransformation): Source to output transformation.
            scale (NDArray[np.float64]): Output pixels per transformation pixel.
        """
        if view.current_tab() is not ImageViewTab.DEWARPING:
            return
        model = (params.distortion_model, params.depth_perception)
        if model != self._fed_model:
            self._fed_model = model
            view.set_distortion_model(
                _model_in_view_pixels(params.distortion_model, xform, scale),
                params.depth_perception,
            )

    def _on_distortion_model_edited(self, model: DistortionModelSpec) -> None:
        xform = self._output_transformation
        if xform is None or not isinstance(self.options, output.OptionsWidget):
            return
        page = self._page
        if page is None:
            return
        stored = _model_in_page_pixels(model, xform, self._output_scale)
        filter_ = self.sequence.output_filter
        params = filter_.get_params(page)
        if params.dewarping.mode is DewarpingMode.OFF:
            filter_.set_params(
                page,
                params.with_dewarping(params.dewarping.with_mode(DewarpingMode.MANUAL)),
            )
        depth = params.depth_perception
        self._fed_model = (stored, depth)
        self.options.set_distortion_model(stored)

    @staticmethod
    def _view_zone(
        spec: PictureZoneSpec | FillZoneSpec,
        kind: ZoneKind,
        xform: ImageTransformation,
        scale: NDArray[np.float64],
        mapper: DewarpingPointMapper | None = None,
    ) -> ViewZone:
        points = xform.map_polygon(np.asarray(spec.points, dtype=np.float64)) * scale
        if mapper is not None:
            points = mapper.to_dewarped(points)
        polygon = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in points])
        properties = spec.properties
        color = (
            properties.color
            if isinstance(properties, FillZoneProperties)
            else (255, 255, 255)
        )
        return ViewZone(kind, polygon, color, properties)

    def _after_edit(self) -> None:
        self._refresh_geometry()
        self._refresh_options()
        self.settings_changed.emit()

    def _refresh_options(self) -> None:
        page = self._page
        if page is None:
            return
        if isinstance(self.options, fix_orientation.OptionsWidget):
            self.options.set_current_image(page.image_id)
        else:
            self.options.set_current_page(page)  # type: ignore[attr-defined]

    def _on_rotation_requested(self, rotation: OrthogonalRotation) -> None:
        page = self._page
        if page is None:
            return
        self.sequence.fix_orientation_filter.set_rotation(page.image_id, rotation)
        self._after_edit()

    def _on_page_layout_set(self, layout: PageLayout) -> None:
        page = self._page
        if page is None:
            return
        rotation = self.sequence.fix_orientation_filter.get_rotation(page.image_id)
        self.sequence.page_split_filter.set_manual_layout(
            page, layout, rotation_degrees=rotation.degrees
        )
        self._after_edit()

    def _on_unremove_requested(self, sub_page: SubPage) -> None:
        page = self._page
        if page is not None:
            self.unremove_page_requested.emit(
                PageId(image_id=page.image_id, sub_page=sub_page)
            )

    def _on_deskew_angle_set(self, degrees: float) -> None:
        page = self._page
        if page is None:
            return
        self.sequence.deskew_filter.set_manual_angle(page, degrees)
        self._after_edit()

    def _on_oblique_angle_set(self, degrees: float) -> None:
        page = self._page
        if page is None:
            return
        self.sequence.deskew_filter.set_manual_oblique(page, degrees)
        self._after_edit()

    def _on_content_rect_set(self, rect: QtCore.QRectF) -> None:
        page = self._page
        if page is None:
            return
        box = (
            ContentBox()
            if rect.isEmpty()
            else ContentBox(
                x=rect.x(), y=rect.y(), width=rect.width(), height=rect.height()
            )
        )
        self.sequence.select_content_filter.set_manual_content(page, box)
        self._after_edit()

    def _on_page_rect_set(self, rect: QtCore.QRectF) -> None:
        page = self._page
        if page is None:
            return
        box = PageBox(x=rect.x(), y=rect.y(), width=rect.width(), height=rect.height())
        self.sequence.select_content_filter.set_manual_page(page, box)
        self._after_edit()

    def _on_margins_set(self, margins: Margins) -> None:
        page = self._page
        if page is None:
            return
        self.sequence.page_layout_filter.set_margins(page, margins)
        self._after_edit()

    def _on_zones_changed(self) -> None:
        page, xform = self._page, self._output_transformation
        view = self.view
        if page is None or xform is None or not isinstance(view, OutputImageView):
            return
        picture: list[PictureZoneSpec] = []
        fill: list[FillZoneSpec] = []
        for zone in view.zones():
            corners = np.array(
                [[p.x(), p.y()] for p in zone.polygon.toList()], dtype=np.float64
            )
            if zone.kind == ZoneKind.FILL and self._fill_zone_mapper is not None:
                corners = self._fill_zone_mapper.to_warped(corners)
            points = tuple(
                (float(x), float(y))
                for x, y in xform.map_polygon_back(corners / self._output_scale)
            )
            if zone.kind == ZoneKind.PICTURE:
                picture_properties = (
                    zone.properties
                    if isinstance(zone.properties, PictureZoneProperties)
                    else PictureZoneProperties(layer=_NEW_PICTURE_LAYER)
                )
                picture.append(
                    PictureZoneSpec(points=points, properties=picture_properties)
                )
            else:
                fill_properties = (
                    zone.properties
                    if isinstance(zone.properties, FillZoneProperties)
                    else FillZoneProperties(color=zone.fill_color)
                )
                fill.append(FillZoneSpec(points=points, properties=fill_properties))
        filter_ = self.sequence.output_filter
        filter_.set_params(
            page,
            filter_.get_params(page)
            .with_picture_zones(tuple(picture))
            .with_fill_zones(tuple(fill)),
        )
        self._after_edit()


def _grown(
    rect: QtCore.QRectF, margins: Margins, mm_x: float, mm_y: float
) -> QtCore.QRectF:
    """Return a rectangle grown by millimetre margins on each side."""
    left, right = margins.left / mm_x, margins.right / mm_x
    top, bottom = margins.top / mm_y, margins.bottom / mm_y
    return QtCore.QRectF(
        rect.x() - left,
        rect.y() - top,
        rect.width() + left + right,
        rect.height() + top + bottom,
    )
