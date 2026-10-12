"""The options widget and image view that belong to a filter stage.

The view of every stage but Output shows the oriented source image through the
transformation the earlier stages define, with the stage's own geometry drawn
on top. Dragging a handle stores the new geometry as manual parameters of the
current page and asks for a re-render. The source image is decoded here, off
the GUI thread, because the views need unprocessed pixels that the renderer's
stage results no longer hold.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui, QtWidgets

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.app.ui.image_views.deskew import DeskewImageView
from scantailor.app.ui.image_views.fix_orientation import FixOrientationImageView
from scantailor.app.ui.image_views.output import OutputImageView, ViewZone, ZoneKind
from scantailor.app.ui.image_views.page_layout import PageLayoutImageView
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
from scantailor.core.image_io import ImageLoadError
from scantailor.core.models import ImageId, Margins, OrthogonalRotation, PageId
from scantailor.core.page_geometry import (
    build_transformation,
    content_rect_of,
    transformation_dpi,
)
from scantailor.core.page_source import DpiOverrides, load_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation, Rect
from scantailor.core.zones import (
    FillZoneProperties,
    PictureLayer,
    PictureZoneProperties,
)
from scantailor.filters.output.params import FillZoneSpec, PictureZoneSpec
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


@dataclass
class _Source:
    """Decoded source image of a page, BGR or gray as the loader returns it."""

    image_id: ImageId
    pixels: NDArray[np.uint8]
    dpi: float


class _SourceSignals(QtCore.QObject):
    loaded = QtCore.Signal(int, object, object, float)
    failed = QtCore.Signal(int, object, str)


class _SourceTask(QtCore.QRunnable):
    """Decodes one source image; touches no widget."""

    def __init__(
        self,
        generation: int,
        image_id: ImageId,
        dpi_overrides: DpiOverrides,
        signals: _SourceSignals,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._generation = generation
        self._image_id = image_id
        self._dpi_overrides = dpi_overrides
        self._signals = signals

    def run(self) -> None:
        """Decode the image and report it, or the failure."""
        try:
            page = load_page(self._image_id, self._dpi_overrides)
        except (ImageLoadError, OSError) as error:
            self._report(self._signals.failed, str(error))
            return
        self._report(self._signals.loaded, page.pixels, page.horizontal_dpi())

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


def _qrect(box: ContentBox | PageBox) -> QtCore.QRectF:
    return QtCore.QRectF(box.x, box.y, box.width, box.height)


class StagePanels(QtCore.QObject):
    """Options widget and image view of one stage, bound to a stage sequence.

    The options widgets write their edits to the stage's settings themselves;
    edits made in the image view are written here. Either way this class
    reports that an edit happened.

    Signals:
        settings_changed: The user edited the current page.
    """

    settings_changed = QtCore.Signal()

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

        self._source_pool = QtCore.QThreadPool(self)
        self._source_pool.setMaxThreadCount(1)
        self._source_signals = _SourceSignals(self)
        self._source_signals.loaded.connect(self._on_source_loaded)
        self._source_signals.failed.connect(self._on_source_failed)
        self._source_generation = 0
        self._dpi_overrides: DpiOverrides = {}

        self._connect_view(stage)

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
            case FilterStage.PAGE_SPLIT:
                widget = page_split.OptionsWidget(sequence.page_split_filter)
                widget.layout_type_changed.connect(forward)
                widget.split_mode_changed.connect(forward)
            case FilterStage.DESKEW:
                widget = deskew.OptionsWidget(sequence.deskew_filter)
                widget.angle_changed.connect(forward)
            case FilterStage.SELECT_CONTENT:
                widget = select_content.OptionsWidget(sequence.select_content_filter)
                widget.page_mode_changed.connect(forward)
                widget.content_mode_changed.connect(forward)
                widget.fine_tune_changed.connect(forward)
            case FilterStage.PAGE_LAYOUT:
                widget = page_layout.OptionsWidget(sequence.page_layout_filter)
                widget.margins_changed.connect(forward)
                widget.alignment_changed.connect(forward)
            case FilterStage.OUTPUT:
                widget = output.OptionsWidget(sequence.output_filter)
                widget.params_changed.connect(forward)
        return widget

    def _connect_view(self, stage: FilterStage) -> None:
        view = self.view
        if isinstance(view, FixOrientationImageView):
            view.rotation_requested.connect(self._on_rotation_requested)
        elif isinstance(view, PageSplitImageView):
            view.page_layout_set_locally.connect(self._on_page_layout_set)
        elif isinstance(view, DeskewImageView):
            view.manual_deskew_angle_set.connect(self._on_deskew_angle_set)
        elif isinstance(view, SelectContentImageView):
            view.manual_content_rect_set.connect(self._on_content_rect_set)
            view.manual_page_rect_set.connect(self._on_page_rect_set)
        elif isinstance(view, PageLayoutImageView):
            view.margins_set_locally.connect(self._on_margins_set)
        elif isinstance(view, OutputImageView):
            view.zones_changed.connect(self._on_zones_changed)

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
        self._source_pool.start(
            _SourceTask(
                self._source_generation,
                image_id,
                self._dpi_overrides,
                self._source_signals,
            )
        )

    @QtCore.Slot(int, object, object, float)
    def _on_source_loaded(
        self,
        generation: int,
        image_id: ImageId,
        pixels: NDArray[np.uint8],
        dpi: float,
    ) -> None:
        if generation != self._source_generation:
            return
        self._source = _Source(image_id, pixels, dpi)
        if isinstance(self.options, page_layout.OptionsWidget):
            self.options.set_page_dpi(dpi, dpi)
        self._present()

    @QtCore.Slot(int, object, str)
    def _on_source_failed(
        self, generation: int, image_id: ImageId, message: str
    ) -> None:
        if generation == self._source_generation:
            logger.warning(
                f"Source image {image_id.file_path} cannot be shown: {message}"
            )

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

    def _present(self) -> None:
        """Show the page's image once the source and the render have arrived."""
        current = self._current()
        if current is None or self._stage_image is None:
            return
        page, source = current
        self._show_image_for(page, source)
        self._refresh_geometry()

    def _rotation(self, page: PageId) -> OrthogonalRotation:
        return self.sequence.fix_orientation_filter.get_rotation(page.image_id)

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
            key = (self.stage, page, image.shape)
            keep = key == self._view_key
            if self._already_shown(key, image):
                return
            view.set_image(image, keep_view=keep)
            return

        if isinstance(view, FixOrientationImageView):
            if self._already_shown((self.stage, source.image_id), source):
                return
            view.set_rotation(rotation)
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
            view.set_deskew_angle(
                self.sequence.deskew_filter.get_params(page).deskew_angle_deg
            )
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
            return
        if self._stage_image is None:
            return

        oriented_size = self._oriented_size(page, source)
        if isinstance(view, PageSplitImageView):
            view.set_page_layout(
                sequence.page_split_filter.get_params(page).page_layout
            )
        elif isinstance(view, DeskewImageView):
            view.set_transformation(
                build_transformation(
                    sequence, page, oriented_size, source.dpi, FilterStage.PAGE_SPLIT
                )
            )
            view.set_deskew_angle(
                sequence.deskew_filter.get_params(page).deskew_angle_deg
            )
        elif isinstance(view, SelectContentImageView):
            self._refresh_select_content(view, page, oriented_size, source.dpi)
        elif isinstance(view, PageLayoutImageView):
            self._refresh_page_layout(view, page, oriented_size, source.dpi)
        elif isinstance(view, OutputImageView):
            self._refresh_zones(view, page, oriented_size, source.dpi)

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
        dpi: float,
    ) -> None:
        params = self.sequence.select_content_filter.get_params(page)
        view.set_transformation(
            build_transformation(
                self.sequence, page, oriented_size, dpi, FilterStage.DESKEW
            )
        )
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
        dpi: float,
    ) -> None:
        sequence = self.sequence
        filter_ = sequence.page_layout_filter
        xform = build_transformation(
            sequence, page, oriented_size, dpi, FilterStage.DESKEW
        )
        view.set_transformation(xform)

        mm_x, mm_y = rect_size_mm(xform, Rect(x=0, y=0, width=1, height=1))
        view.set_mm_per_pixel(mm_x, mm_y)
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
        dpi: float,
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
        params = self.sequence.output_filter.get_params(page)
        zones = [
            self._view_zone(spec, ZoneKind.PICTURE, xform, scale)
            for spec in params.picture_zones
        ] + [
            self._view_zone(spec, ZoneKind.FILL, xform, scale)
            for spec in params.fill_zones
        ]
        view.set_zones(zones)

    @staticmethod
    def _view_zone(
        spec: PictureZoneSpec | FillZoneSpec,
        kind: ZoneKind,
        xform: ImageTransformation,
        scale: NDArray[np.float64],
    ) -> ViewZone:
        points = xform.map_polygon(np.asarray(spec.points, dtype=np.float64)) * scale
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
        filter_ = self.sequence.page_split_filter
        filter_.set_params(page, filter_.get_params(page).with_manual_layout(layout))
        self._after_edit()

    def _on_deskew_angle_set(self, degrees: float) -> None:
        page = self._page
        if page is None:
            return
        self.sequence.deskew_filter.set_manual_angle(page, degrees)
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
