"""Background rendering of a page up to a filter stage."""

import hashlib
import threading
import weakref
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui

from scantailor.app.ui.source_cache import SourceCache
from scantailor.app.ui.thumbnails.overlays import paint_stage_overlay, view_stage
from scantailor.app.ui.thumbnails.render_memo import (
    ReducedSource,
    ThumbnailMemo,
    reduced_copy,
)
from scantailor.core import pipeline
from scantailor.core.debug_images import DebugImages
from scantailor.core.image_io import ImageLoadError
from scantailor.core.models import Dpi, ImageId, PageId, page_name
from scantailor.core.page_source import (
    FALLBACK_DPI,
    DpiOverrides,
    LoadedPage,
    load_page,
)
from scantailor.core.pipeline import OutputExtras
from scantailor.core.settings import ApplicationSettings
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.output.params import DistortionModelSpec

__all__ = [
    "FALLBACK_DPI",
    "PageRenderer",
    "render_page",
    "render_page_with_model",
    "render_thumbnail",
    "thumbnail_file_digest",
    "warm_up_qt_image_types",
]

# One more than the pool needs for the render the user waits for: a render that
# was superseded cannot be interrupted and keeps its thread until it returns.
_MAX_THREADS = 3

# Longest a thumbnail worker waits for the render the user is waiting for.
THUMBNAIL_YIELD_SECONDS = 30.0

# A shrunk source is only used while it holds this multiple of the detail the
# thumbnail shows; the area averaging down to the box then hides the extra
# resampling step.
_OVERSAMPLE = 2
# A shrunk source keeps this multiple of the box along its longer side, so views
# of two thirds of the image or more can still be oversampled from it.
_REDUCED_SOURCE_FACTOR = 3

# The stages that may detect parameters when a render reaches them.
_DETECTING_STAGES = (
    FilterStage.PAGE_SPLIT,
    FilterStage.DESKEW,
    FilterStage.SELECT_CONTENT,
)

SourceSizeReport = Callable[[ImageId, int, int], None]
"""Receives an image and the width and height its source file decoded to."""

_SourceGeometry = tuple[tuple[int, int], Dpi]
"""Width and height of a decoded source before Fix Orientation, and its resolution."""


# Bump when a change to thumbnail rendering makes the thumbnails kept on disk stale.
THUMBNAIL_FILE_FORMAT = 1

# Keys name their content, so every window and project can share one memo.
_SHARED_MEMO = ThumbnailMemo()


def release_thumbnail_memory() -> None:
    """Drop the kept thumbnails and shrunk sources, such as when a project closes.

    Only frees memory early: kept entries are keyed by the file's content and
    parameters, so none could be handed out wrongly later.
    """
    _SHARED_MEMO.clear()


def warm_up_qt_image_types() -> None:
    """Resolve the PySide types that thumbnail workers touch, on this thread.

    PySide creates enum and value types on first use, and two threads that
    first use one at the same moment can see it half made (``QImage`` has no
    attribute ``Format``). Calling this once on the GUI thread, before any
    worker starts, makes every later use a plain lookup.
    """
    for pixel_format in (
        QtGui.QImage.Format.Format_Grayscale8,
        QtGui.QImage.Format.Format_RGB888,
    ):
        QtGui.QImage(1, 1, pixel_format).copy()
    QtCore.QSize(1, 1).width()


# Importing happens on the GUI thread, before the thumbnail pool exists.
warm_up_qt_image_types()


def render_page(
    page_id: PageId,
    sequence: StageSequence,
    end_stage: FilterStage,
    dpi_overrides: DpiOverrides | None = None,
) -> NDArray[np.uint8]:
    """Load a page and run the pipeline up to ``end_stage``.

    Args:
        page_id (PageId): Page to render.
        sequence (StageSequence): Stages holding the page's parameters; only read.
        end_stage (FilterStage): Last stage to run.
        dpi_overrides (DpiOverrides | None): Project resolutions that win over
            the image file's.

    Returns:
        NDArray[np.uint8]: Display image, ``(H, W)`` gray or ``(H, W, 3)`` RGB.

    Raises:
        ImageLoadError: If the source image cannot be decoded.
    """
    return render_page_with_model(page_id, sequence, end_stage, dpi_overrides)[0]


def render_page_with_model(
    page_id: PageId,
    sequence: StageSequence,
    end_stage: FilterStage,
    dpi_overrides: DpiOverrides | None = None,
    *,
    sources: SourceCache | None = None,
    keep_source: bool = True,
    pixels: bool = True,
    on_extras: Callable[[OutputExtras], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_source: Callable[[LoadedPage], None] | None = None,
    debug_images: DebugImages | None = None,
) -> tuple[NDArray[np.uint8], DistortionModelSpec | None]:
    """Render a page like ``render_page`` and return the dewarping model it built.

    Args:
        page_id (PageId): Page to render.
        sequence (StageSequence): Stages holding the page's parameters; only read.
        end_stage (FilterStage): Last stage to run.
        dpi_overrides (DpiOverrides | None): Project resolutions that win over
            the image file's.
        sources (SourceCache | None): Where to take the decoded source from, so
            that it is decoded once for everyone; decoded here when None.
        keep_source (bool): Whether a source decoded for this render stays in
            ``sources``. False for pages that are not the one on screen.
        pixels (bool): False when the caller does not draw the rendered image,
            which then comes back as an empty array without being resampled
            or converted. Output is still generated in full.
        on_extras (Callable[[OutputExtras], None] | None): Called with the
            pictures the Output view's tabs show, when the render reaches the
            Output stage; they are not produced without it.
        should_cancel (Callable[[], bool] | None): Asked between stages; True
            stops the render with ``pipeline.PipelineCancelled``.
        on_source (Callable[[LoadedPage], None] | None): Called with the
            decoded source before the pipeline runs; it must not modify it.
        debug_images (DebugImages | None): Receives the intermediate images of
            ``end_stage``.

    Returns:
        tuple[NDArray[np.uint8], DistortionModelSpec | None]: The display image
            and the page edges automatic or marginal dewarping found, or None
            when the render built no model. The caller stores the model.

    Raises:
        ImageLoadError: If the source image cannot be decoded.
    """
    if sources is None:
        page = load_page(page_id.image_id, dpi_overrides)
    else:
        page = sources.load(page_id.image_id, dpi_overrides, keep=keep_source)
    if on_source is not None:
        on_source(page)
    options = pipeline.PipelineOptions(
        end_stage=end_stage,
        dpi=page.resolution(),
        keep_output_extras=on_extras is not None,
        should_cancel=should_cancel,
        render_image=pixels,
        black_on_white_detection=ApplicationSettings.get_instance().polarity_detection,
        debug_images=debug_images,
    )
    result = pipeline.process_page(page.pixels, page_id, sequence, options)
    if on_extras is not None and result.output_extras is not None:
        on_extras(result.output_extras)
    if not pixels:
        return np.empty((0, 0), dtype=np.uint8), result.distortion_model
    return _bgr_to_rgb(result.image), result.distortion_model


def _bgr_to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return a colour image in display channel order; gray is returned as is."""
    if image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(image[:, :, ::-1])
    return image


def render_thumbnail(
    page_id: PageId,
    sequence: StageSequence,
    end_stage: FilterStage,
    max_size: QtCore.QSize,
    dpi_overrides: DpiOverrides | None = None,
    sources: SourceCache | None = None,
    memo: ThumbnailMemo | None = None,
    removed_halves: tuple[bool, bool] = (False, False),
    on_source_size: SourceSizeReport | None = None,
) -> QtGui.QImage:
    """Render a page up to ``end_stage`` and shrink it to fit ``max_size``.

    Safe to call from worker threads. Stages may store what they detect in
    ``sequence``, as any render does. Before Output, the stage's result is
    drawn over the picture (see ``thumbnails.overlays``); at Page Split the
    picture is the whole image after Fix Orientation, not the page alone.

    Before Output, a thumbnail whose parameters are unchanged since it was last
    made is handed out again, and one whose detecting stages a full render has
    already settled is rendered from a shrunk copy of the source. Only the rest
    decode the source at full size.

    Args:
        page_id (PageId): Page to render.
        sequence (StageSequence): Stages holding the page's parameters.
        end_stage (FilterStage): Last stage to run.
        max_size (QtCore.QSize): Bounding size of the result.
        dpi_overrides (DpiOverrides | None): Project resolutions that win over
            the image file's.
        sources (SourceCache | None): Decoded sources to reuse when the page is
            in it; a page decoded here is not added to it.
        memo (ThumbnailMemo | None): Where thumbnails, shrunk sources and
            settled parameters are kept; one shared by every caller when None.
        removed_halves (tuple[bool, bool]): Whether the left and the right half
            of the page's image were removed from the project; Page Split's
            overlay tints them as off-cuts.
        on_source_size (SourceSizeReport | None): Called on this thread with
            the width and height of the source the thumbnail was made from,
            before Fix Orientation. Not called when a kept thumbnail is handed
            out again.

    Returns:
        QtGui.QImage: 8-bit gray or RGB888, possibly handed out before; do not
            modify it.

    Raises:
        ImageLoadError: If the source image cannot be decoded.
    """
    report = on_source_size or (lambda _image_id, _width, _height: None)
    if end_stage >= FilterStage.OUTPUT:

        def report_decoded(page: LoadedPage) -> None:
            height, width = page.pixels.shape[:2]
            report(page_id.image_id, width, height)

        image = render_page_with_model(
            page_id,
            sequence,
            end_stage,
            dpi_overrides,
            sources=sources,
            keep_source=False,
            on_source=report_decoded,
        )[0]
        return shrink_to_thumbnail(image, max_size)
    memo = _SHARED_MEMO if memo is None else memo
    source_key = _source_key(page_id.image_id, dpi_overrides)
    before = _ParamDigests.of(page_id, sequence, end_stage, source_key)
    image_key = (before.everything, max_size.width(), max_size.height(), removed_halves)
    thumbnail = memo.image(image_key)
    if thumbnail is not None:
        return thumbnail
    rendered = _render_from_reduced_source(
        page_id, sequence, end_stage, max_size, memo, source_key, before
    )
    from_full_size = rendered is None
    if rendered is None:
        rendered = _render_from_full_source(
            page_id,
            sequence,
            end_stage,
            max_size,
            (dpi_overrides, sources),
            memo,
            source_key,
        )
    picture, (source_size, dpi) = rendered
    report(page_id.image_id, *source_size)
    oriented_size = _oriented(source_size, page_id, sequence)
    thumbnail = paint_stage_overlay(
        picture, end_stage, page_id, sequence, (oriented_size, dpi), removed_halves
    )
    # Unchanged digests mean no stage detected and nobody edited meanwhile.
    if _ParamDigests.of(page_id, sequence, end_stage, source_key) == before:
        memo.keep_image(image_key, thumbnail)
        if from_full_size:
            for digest in before.detecting:
                memo.settle(digest)
    return thumbnail


def _render_from_reduced_source(
    page_id: PageId,
    sequence: StageSequence,
    end_stage: FilterStage,
    max_size: QtCore.QSize,
    memo: ThumbnailMemo,
    source_key: Hashable,
    digests: _ParamDigests,
) -> tuple[QtGui.QImage, _SourceGeometry] | None:
    """Render from the kept shrunk source when no stage would detect.

    Returns:
        tuple[QtGui.QImage, _SourceGeometry] | None: The thumbnail without its
            overlay and the full-size source it stands for, or None when it
            needs the full-size source.
    """
    reduced = memo.reduced_source(source_key)
    if reduced is None:
        return None
    if digests.detecting and not memo.is_settled(digests.detecting[-1]):
        return None
    picture = _picture_from_reduced(
        reduced, page_id, sequence, view_stage(end_stage), max_size
    )
    return None if picture is None else (picture, (reduced.full_size, reduced.dpi))


def _picture_from_reduced(
    reduced: ReducedSource,
    page_id: PageId,
    sequence: StageSequence,
    stage: FilterStage,
    max_size: QtCore.QSize,
) -> QtGui.QImage | None:
    """Render a stage's view from a shrunk source with the stored parameters.

    Returns:
        QtGui.QImage | None: The picture, or None when the shrunk source is too
            coarse for ``max_size``.
    """
    rendered = pipeline.render_from_reduced_source(
        reduced.pixels,
        reduced.full_size,
        page_id,
        sequence,
        pipeline.PipelineOptions(end_stage=stage, dpi=reduced.dpi),
        (max_size.width() * _OVERSAMPLE, max_size.height() * _OVERSAMPLE),
    )
    if rendered is None:
        return None
    view, full_size = rendered
    return shrink_to_thumbnail(_bgr_to_rgb(view), max_size, full_size)


def _render_from_full_source(
    page_id: PageId,
    sequence: StageSequence,
    end_stage: FilterStage,
    max_size: QtCore.QSize,
    origin: tuple[DpiOverrides | None, SourceCache | None],
    memo: ThumbnailMemo,
    source_key: Hashable,
) -> tuple[QtGui.QImage, _SourceGeometry]:
    """Render from the full-size source, keeping a shrunk copy of it.

    Returns:
        tuple[QtGui.QImage, _SourceGeometry]: The thumbnail without its overlay
            and the source it was rendered from.
    """
    dpi_overrides, sources = origin
    longest_side = _REDUCED_SOURCE_FACTOR * max(max_size.width(), max_size.height())
    decoded: list[LoadedPage] = []

    def keep_reduced(page: LoadedPage) -> None:
        decoded.append(page)
        kept = memo.reduced_source(source_key)
        height, width = page.pixels.shape[:2]
        # A copy made for this size falls short of it by the rounding of its sides.
        if kept is not None and kept.scale * max(width, height) >= longest_side - 1:
            return
        reduced = reduced_copy(page.pixels, page.resolution(), longest_side)
        if reduced is not None:
            memo.keep_reduced_source(source_key, reduced)

    shown_stage = view_stage(end_stage)
    image = render_page_with_model(
        page_id,
        sequence,
        end_stage,
        dpi_overrides,
        sources=sources,
        keep_source=False,
        pixels=shown_stage == end_stage,
        on_source=keep_reduced,
    )[0]
    page = decoded[-1]
    height, width = page.pixels.shape[:2]
    geometry = ((width, height), page.resolution())
    if shown_stage == end_stage:
        return shrink_to_thumbnail(image, max_size), geometry
    # The run only detected. The view shown stops before every detecting stage,
    # so the shrunk copy just made can render it without waiting to be settled.
    reduced = memo.reduced_source(source_key)
    picture = (
        None
        if reduced is None
        else _picture_from_reduced(reduced, page_id, sequence, shown_stage, max_size)
    )
    if picture is None:
        options = pipeline.PipelineOptions(end_stage=shown_stage, dpi=page.resolution())
        view = pipeline.process_page(page.pixels, page_id, sequence, options).image
        picture = shrink_to_thumbnail(_bgr_to_rgb(view), max_size)
    return picture, geometry


def _oriented(
    size: tuple[int, int], page_id: PageId, sequence: StageSequence
) -> tuple[int, int]:
    """Return a source size as Fix Orientation turns it."""
    rotation = sequence.fix_orientation_filter.get_rotation(page_id.image_id)
    width, height = size
    return (height, width) if rotation.degrees % 180 else (width, height)


def _source_key(
    image_id: ImageId, dpi_overrides: DpiOverrides | None
) -> tuple[ImageId, Dpi | None, tuple[int, int] | None]:
    """Name a source by the image, its project resolution and the file's state."""
    override = dpi_overrides.get(image_id) if dpi_overrides else None
    try:
        stat = image_id.file_path.stat()
        version: tuple[int, int] | None = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        version = None
    return (image_id, override, version)


@dataclass(frozen=True)
class _ParamDigests:
    """Digests of everything a render up to a stage before Output reads.

    Attributes:
        detecting (tuple[str, ...]): One digest per stage that can detect, up to
            the end stage, each covering the source and every stage up to it.
        everything (str): Covers the source, every stage up to the end stage and
            the size the pages are matched to.
    """

    detecting: tuple[str, ...]
    everything: str

    @classmethod
    def of(
        cls,
        page_id: PageId,
        sequence: StageSequence,
        end_stage: FilterStage,
        source_key: Hashable,
    ) -> _ParamDigests:
        """Read the parameters a render of a page up to ``end_stage`` uses."""
        digest = hashlib.blake2b(repr((source_key, page_id)).encode(), digest_size=16)
        orientation = sequence.fix_orientation_filter
        for model in (
            orientation.get_rotation(page_id.image_id),
            orientation.get_trim(page_id.image_id),
        ):
            digest.update(model.model_dump_json().encode())
        params_of = {
            FilterStage.PAGE_SPLIT: sequence.page_split_filter.get_params,
            FilterStage.DESKEW: sequence.deskew_filter.get_params,
            FilterStage.SELECT_CONTENT: sequence.select_content_filter.get_params,
            FilterStage.PAGE_LAYOUT: sequence.page_layout_filter.get_params,
            FilterStage.OUTPUT: sequence.output_filter.get_params,
        }
        detecting: list[str] = []
        for stage, params in params_of.items():
            if stage > end_stage:
                break
            digest.update(params(page_id).model_dump_json().encode())
            if stage in _DETECTING_STAGES:
                detecting.append(digest.hexdigest())
        if end_stage >= FilterStage.PAGE_LAYOUT:
            aggregate = sequence.page_layout_filter.get_aggregate_hard_size()
            digest.update(aggregate.model_dump_json().encode())
        digest.update(str(int(end_stage)).encode())
        return cls(tuple(detecting), digest.hexdigest())


def thumbnail_file_digest(
    pages: Sequence[PageId],
    sequence: StageSequence,
    end_stage: FilterStage,
    dpi_overrides: DpiOverrides | None = None,
    removed_halves: tuple[bool, bool] = (False, False),
) -> str | None:
    """Digest what a thumbnail is rendered from, for keeping it on disk.

    Covers each page's source file (its path, size and modification time) and
    project resolution, every stage's parameters up to ``end_stage``, the size
    the pages are matched to, the removed halves, the polarity detection
    setting and ``THUMBNAIL_FILE_FORMAT``. Nothing in it lives only as long as
    the session, so a thumbnail stored with it can be shown in a later session.

    Safe to call from worker threads; settings are only read.

    Args:
        pages (Sequence[PageId]): Every page the thumbnail stands for: one page,
            or all pages of an image where the list shows images.
        sequence (StageSequence): Stages holding the pages' parameters.
        end_stage (FilterStage): Stage the thumbnail shows.
        dpi_overrides (DpiOverrides | None): Project resolutions that win over
            the image file's.
        removed_halves (tuple[bool, bool]): Whether the left and the right half
            of the image were removed from the project.

    Returns:
        str | None: A hex digest, or None when a source file cannot be read,
            so that there is nothing to tell a replaced file by.
    """
    polarity = ApplicationSettings.get_instance().polarity_detection
    digest = hashlib.blake2b(
        repr((THUMBNAIL_FILE_FORMAT, removed_halves, polarity)).encode(),
        digest_size=16,
    )
    for page_id in pages:
        source_key = _source_key(page_id.image_id, dpi_overrides)
        if source_key[-1] is None:
            return None
        params = _ParamDigests.of(page_id, sequence, end_stage, source_key)
        digest.update(params.everything.encode())
    return digest.hexdigest()


def shrink_to_thumbnail(
    image: NDArray[np.uint8],
    max_size: QtCore.QSize,
    full_size: tuple[int, int] | None = None,
) -> QtGui.QImage:
    """Shrink a rendered page to fit ``max_size``; smaller pages keep their size.

    Args:
        image (NDArray[np.uint8]): Display image, ``(H, W)`` gray or ``(H, W, 3)``
            RGB, as ``render_page_with_model`` returns it; only read.
        max_size (QtCore.QSize): Bounding size of the result.
        full_size (tuple[int, int] | None): Width and height of the page at
            full size when ``image`` is a reduced copy of it, so that the result
            is as large as one shrunk from the full-size page.

    Returns:
        QtGui.QImage: Detached image, 8-bit gray or RGB888.
    """
    height, width = image.shape[:2]
    if full_size is not None:
        width, height = full_size
    scale = min(max_size.width() / width, max_size.height() / height, 1.0)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size != (image.shape[1], image.shape[0]):
        image = np.asarray(
            cv2.resize(image, size, interpolation=cv2.INTER_AREA), dtype=np.uint8
        )
    image = np.ascontiguousarray(image)
    pixel_format = (
        QtGui.QImage.Format.Format_Grayscale8
        if image.ndim == 2
        else QtGui.QImage.Format.Format_RGB888
    )
    height, width = image.shape[:2]
    return QtGui.QImage(
        image.data, width, height, image.strides[0], pixel_format
    ).copy()


class _RenderSignals(QtCore.QObject):
    done = QtCore.Signal(int, object, object)
    model_found = QtCore.Signal(int, object, object)
    extras_found = QtCore.Signal(int, object, object)
    debug_found = QtCore.Signal(int, object, object)
    failed = QtCore.Signal(int, object, str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        # Collectors sent through ``debug_found`` that the GUI thread has not
        # received yet, so that they can be closed if it never does.
        self.debug_in_flight: set[DebugImages] = set()
        self.debug_lock = threading.Lock()


class _RenderTask(QtCore.QRunnable):
    """One render; a cancelled task never reports."""

    def __init__(
        self,
        generation: int,
        page_id: PageId,
        sequence: StageSequence,
        stage: FilterStage,
        dpi_overrides: DpiOverrides | None,
        signals: _RenderSignals,
        sources: SourceCache,
        pixels: bool,
        finished: Callable[[_RenderTask], None],
        extras: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.cancelled = threading.Event()
        self._generation = generation
        self._page_id = page_id
        self._sequence = sequence
        self._stage = stage
        self._dpi_overrides = dpi_overrides
        self._signals = signals
        self._sources = sources
        self._pixels = pixels
        # Weak, so that a task finishing on a pool thread never holds the last
        # reference to the renderer and destroys it there.
        self._finished = weakref.WeakMethod(finished)
        self._extras = extras
        self._debug = debug
        self.image: NDArray[np.uint8] | None = None

    def run(self) -> None:
        """Render on a pool thread and report unless cancelled."""
        try:
            if not self.cancelled.is_set():
                with logger.contextualize(page=page_name(self._page_id)):
                    self._render_and_report()
        finally:
            finished = self._finished()
            if finished is not None:
                finished(self)

    def _report_extras(self, extras: OutputExtras) -> None:
        if not self.cancelled.is_set():
            self._signals.extras_found.emit(self._generation, self._page_id, extras)

    def _render_and_report(self) -> None:
        debug_images = DebugImages() if self._debug else None
        try:
            self._render_and_report_into(debug_images)
        finally:
            if debug_images is not None and not self._handed_over(debug_images):
                debug_images.close()

    def _handed_over(self, debug_images: DebugImages) -> bool:
        """Send the images of a finished render to the GUI thread.

        Returns:
            bool: True when they were sent and the GUI thread owns their files;
                False for a cancelled or failed render, whose caller closes them.
        """
        if self.cancelled.is_set() or self.image is None:
            return False
        with self._signals.debug_lock:
            self._signals.debug_in_flight.add(debug_images)
        self._signals.debug_found.emit(self._generation, self._page_id, debug_images)
        return True

    def _render_and_report_into(self, debug_images: DebugImages | None) -> None:
        try:
            image, model = render_page_with_model(
                self._page_id,
                self._sequence,
                self._stage,
                self._dpi_overrides,
                sources=self._sources,
                pixels=self._pixels,
                on_extras=self._report_extras if self._extras else None,
                should_cancel=self.cancelled.is_set,
                debug_images=debug_images,
            )
        except pipeline.PipelineCancelled:
            return
        except Exception as error:
            logger.opt(exception=not isinstance(error, ImageLoadError)).warning(
                "Cannot render: {}", error
            )
            if not self.cancelled.is_set():
                self._signals.failed.emit(self._generation, self._page_id, str(error))
            return
        if not self.cancelled.is_set():
            self.image = image
            if model is not None:
                self._signals.model_found.emit(self._generation, self._page_id, model)
            self._signals.done.emit(self._generation, self._page_id, image)

    @property
    def page_id(self) -> PageId:
        """The page this task renders."""
        return self._page_id

    @property
    def stage(self) -> FilterStage:
        """The last stage this task runs."""
        return self._stage


class PageRenderer(QtCore.QObject):
    """Renders pages off the GUI thread and delivers results as signals.

    Only the latest request is reported: requesting again cancels the previous
    one, whose result is dropped even if it is already running.

    Pages are decoded through ``sources``, which the stage panels share, so one
    decode serves every stage and render of a page.

    Signals:
        render_done: A render finished. Args: page_id (PageId), image (ndarray).
        render_failed: A render failed. Args: page_id (PageId), message (str).
        distortion_model_found: A render built a dewarping model, emitted just
            before ``render_done``. Args: page_id (PageId), model
            (DistortionModelSpec).
        output_extras_found: A render asked for the pictures of the Output
            view's tabs and made them, emitted just before ``render_done``.
            Args: page_id (PageId), extras (OutputExtras).
        debug_images_found: A render asked for debug images, emitted after
            ``render_done``. The receiver owns the files and must close the
            collector. Args: page_id (PageId), images (DebugImages).
    """

    render_done = QtCore.Signal(object, object)
    output_extras_found = QtCore.Signal(object, object)
    debug_images_found = QtCore.Signal(object, object)
    distortion_model_found = QtCore.Signal(object, object)
    render_failed = QtCore.Signal(object, str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        """Create the renderer.

        Args:
            parent (QtCore.QObject | None): Owner of the renderer.
        """
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(_MAX_THREADS)
        self._signals = _RenderSignals(self)
        self._signals.done.connect(self._on_done)
        self._signals.model_found.connect(self._on_model_found)
        self._signals.extras_found.connect(self._on_extras_found)
        self._signals.debug_found.connect(self._on_debug_found)
        self._signals.failed.connect(self._on_failed)
        self._generation = 0
        self._sources = SourceCache()
        self._state = threading.Lock()
        self._task: _RenderTask | None = None
        self._task_tag: Hashable | None = None
        self._kept: tuple[PageId, FilterStage, Hashable, NDArray[np.uint8]] | None = (
            None
        )
        self._idle = threading.Event()
        self._idle.set()

    @property
    def sources(self) -> SourceCache:
        """The decoded sources this renderer works from."""
        return self._sources

    def request(
        self,
        page_id: PageId,
        sequence: StageSequence,
        stage: FilterStage,
        dpi_overrides: DpiOverrides | None = None,
        pixels: bool = True,
        extras: bool = False,
        tag: Hashable | None = None,
        debug: bool = False,
    ) -> None:
        """Start rendering a page, cancelling any render still pending.

        Args:
            page_id (PageId): Page to render.
            sequence (StageSequence): Stages holding the parameters to use.
            stage (FilterStage): Last stage to run.
            dpi_overrides (DpiOverrides | None): Snapshot of the project's
                resolutions, read on the worker thread; do not mutate it.
            pixels (bool): False when the image of the render is not drawn, as
                for the stages whose views show the source; ``render_done`` then
                carries an empty array.
            extras (bool): True to also report the pictures of the Output
                view's tabs through ``output_extras_found``.
            tag (Hashable | None): Names the parameters the render is made
                with. A tagged render's image is kept, and ``rendered`` hands it
                out for the same page, stage and tag until the next request
                or cancel.
            debug (bool): True to record the intermediate images of ``stage``
                and report them through ``debug_images_found``.
        """
        self.cancel()
        task = _RenderTask(
            self._generation,
            page_id,
            sequence,
            stage,
            dpi_overrides,
            self._signals,
            self._sources,
            pixels,
            self._on_task_finished,
            extras,
            debug,
        )
        with self._state:
            self._task = task
            self._task_tag = tag
            self._idle.clear()
        self._pool.start(task)

    def rendered(
        self, page_id: PageId, stage: FilterStage, tag: Hashable
    ) -> NDArray[np.uint8] | None:
        """Return the image of the last finished render, if it is the one asked for.

        Safe to call from worker threads. Lets a thumbnail of the page on screen
        reuse the render instead of running the pipeline again.

        Args:
            page_id (PageId): The page wanted.
            stage (FilterStage): The last stage wanted.
            tag (Hashable): The tag the render must have been requested with.

        Returns:
            NDArray[np.uint8] | None: The display image, not to be modified, or
                None when the last render was of something else.
        """
        with self._state:
            kept = self._kept
        if kept is None or kept[:3] != (page_id, stage, tag):
            return None
        return kept[3]

    def cancel(self) -> None:
        """Drop the pending render, if any, and the image a tagged render kept."""
        with self._state:
            task, self._task = self._task, None
            self._kept = None
            self._idle.set()
        if task is not None:
            task.cancelled.set()
        self._generation += 1

    def release_debug_images(self) -> None:
        """Cancel the pending render and close debug images not yet delivered.

        For a window or project that closes. Collectors already delivered
        through ``debug_images_found`` belong to their receiver.
        """
        self.cancel()
        with self._signals.debug_lock:
            in_flight, self._signals.debug_in_flight = (
                self._signals.debug_in_flight,
                set(),
            )
        for images in in_flight:
            images.close()

    def wait_until_idle(self, timeout: float = THUMBNAIL_YIELD_SECONDS) -> bool:
        """Block until no render that the user waits for is running.

        For worker threads whose own work can wait, such as thumbnails, so that
        the page on screen gets the processors first.

        Args:
            timeout (float): Longest wait in seconds.

        Returns:
            bool: False if the timeout expired first.
        """
        return self._idle.wait(timeout)

    def _on_task_finished(self, task: _RenderTask) -> None:
        with self._state:
            if task is self._task:
                if self._task_tag is not None and task.image is not None:
                    self._kept = (task.page_id, task.stage, self._task_tag, task.image)
                self._idle.set()

    def wait_for_done(self, timeout_ms: int = -1) -> bool:
        """Block until every started render has returned.

        Args:
            timeout_ms (int): Longest wait in milliseconds; negative waits forever.

        Returns:
            bool: False if the timeout expired first.
        """
        return self._pool.waitForDone(timeout_ms)

    @QtCore.Slot(int, object, object)
    def _on_done(self, generation: int, page_id: PageId, image: object) -> None:
        if generation == self._generation:
            self.render_done.emit(page_id, image)

    @QtCore.Slot(int, object, object)
    def _on_model_found(
        self, generation: int, page_id: PageId, model: DistortionModelSpec
    ) -> None:
        if generation == self._generation:
            self.distortion_model_found.emit(page_id, model)

    @QtCore.Slot(int, object, object)
    def _on_extras_found(
        self, generation: int, page_id: PageId, extras: OutputExtras
    ) -> None:
        if generation == self._generation:
            self.output_extras_found.emit(page_id, extras)

    @QtCore.Slot(int, object, object)
    def _on_debug_found(
        self, generation: int, page_id: PageId, images: DebugImages
    ) -> None:
        with self._signals.debug_lock:
            if images not in self._signals.debug_in_flight:
                return
            self._signals.debug_in_flight.discard(images)
        if generation == self._generation:
            self.debug_images_found.emit(page_id, images)
        else:
            images.close()

    @QtCore.Slot(int, object, str)
    def _on_failed(self, generation: int, page_id: PageId, message: str) -> None:
        if generation == self._generation:
            self.render_failed.emit(page_id, message)
