"""Output filter image views.

``OutputImageView`` shows the output page and carries the tools of the Output
stage: editing picture and fill zones, the manual dewarping grid and the
despeckling preview. Without tabs it edits both kinds of zone at once; with
tabs, as in upstream, each tab has one job.

``DewarpingImageView`` is the dewarping grid on a view of its own.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from itertools import pairwise
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import (
    QT_TRANSLATE_NOOP,
    QCoreApplication,
    QLineF,
    QPointF,
    QRectF,
    QSettings,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QShortcut,
)

from scantailor.app.ui.icons import icon
from scantailor.app.ui.image_views import zone_geometry as geometry
from scantailor.app.ui.image_views.base import (
    DEFAULT_STATUS_TIP,
    ImageViewBase,
    numpy_to_qimage,
)
from scantailor.app.ui.image_views.despeckle_preview import DespecklePreviewer
from scantailor.app.ui.image_views.dewarp_editor import DewarpEditor
from scantailor.app.ui.image_views.picture_mask_overlay import (
    MaskZone,
    PictureMaskOverlay,
)
from scantailor.app.ui.theme.tokens import view_color
from scantailor.app.ui.widgets.color_pickup import ColorPickupInteraction
from scantailor.app.ui.widgets.picture_zone_dialog import (
    PictureZoneDialog,
    layer_label,
)
from scantailor.core.settings import ApplicationSettings, ZoneCreationMode
from scantailor.core.transformation import ImageTransformation
from scantailor.core.zones import (
    FillZoneProperties,
    PictureLayer,
    PictureZoneProperties,
)
from scantailor.filters.output.despeckle import DespeckleLevel, DespeckleStrength
from scantailor.filters.output.params import DistortionModelSpec


class ZoneKind(Enum):
    """The two kinds of zone the Output view edits."""

    PICTURE = auto()
    FILL = auto()


class ImageViewTab(Enum):
    """Tabs of the Output view, in upstream's order."""

    OUTPUT = auto()
    PICTURE_ZONES = auto()
    FILL_ZONES = auto()
    DEWARPING = auto()
    DESPECKLING = auto()


TAB_LABELS: dict[ImageViewTab, str] = {
    ImageViewTab.OUTPUT: cast(
        str, QT_TRANSLATE_NOOP("output::Task::UiUpdater", "Output")
    ),
    ImageViewTab.PICTURE_ZONES: cast(
        str, QT_TRANSLATE_NOOP("output::Task::UiUpdater", "Picture Zones")
    ),
    ImageViewTab.FILL_ZONES: cast(
        str, QT_TRANSLATE_NOOP("output::Task::UiUpdater", "Fill Zones")
    ),
    ImageViewTab.DEWARPING: cast(
        str, QT_TRANSLATE_NOOP("output::Task::UiUpdater", "Dewarping")
    ),
    ImageViewTab.DESPECKLING: cast(
        str, QT_TRANSLATE_NOOP("output::Task::UiUpdater", "Despeckling")
    ),
}

_TAB_ICONS: dict[ImageViewTab, str] = {
    ImageViewTab.OUTPUT: "eye",
    ImageViewTab.PICTURE_ZONES: "image",
    ImageViewTab.FILL_ZONES: "paint-bucket",
    ImageViewTab.DEWARPING: "dewarp-grid",
    ImageViewTab.DESPECKLING: "wand-sparkles",
}

_ZONE_TAB_KINDS: dict[ImageViewTab, ZoneKind] = {
    ImageViewTab.PICTURE_ZONES: ZoneKind.PICTURE,
    ImageViewTab.FILL_ZONES: ZoneKind.FILL,
}

_ZONE_DEFAULT_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "ZoneDefaultInteraction",
        "Click to start creating a new zone. Ctrl+Alt+Click to copy the latest "
        "created zone. Use Z, X and C keys to switch zone creation mode.",
    ),
)
_VERTEX_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "ZoneDefaultInteraction",
        "Drag the vertex. Hold Ctrl to make the vertex angle right. Press D to "
        "delete the vertex.",
    ),
)
_SEGMENT_TIP = cast(
    str,
    QT_TRANSLATE_NOOP("ZoneDefaultInteraction", "Click to create a new vertex here."),
)
_ZONE_AREA_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "ZoneDefaultInteraction",
        "Right click to edit zone properties. Hold Shift to drag the zone or "
        "Shift+Ctrl to copy. Press Del to delete this zone.",
    ),
)
_ZONE_DRAG_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "ZoneDefaultInteraction", "Hold left mouse button to drag the zone."
    ),
)
_ZONE_COPY_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "ZoneDefaultInteraction", "Hold left mouse button to copy and drag the zone."
    ),
)
_FINISH_DRAG_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "ZoneDragInteraction", "Release left mouse button to finish dragging."
    ),
)
_MERGE_TIP = cast(
    str, QT_TRANSLATE_NOOP("ZoneVertexDragInteraction", "Merge these two vertices.")
)
_MERGE_HINT_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "ZoneVertexDragInteraction",
        "Move the vertex to one of its neighbors to merge them.",
    ),
)
_PICKUP_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "ColorPickupInteraction",
        "Click on an area to pick up its color, or ESC to cancel.",
    ),
)
_FINISH_RECTANGLE_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "ZoneCreationInteraction",
        "Click to finish this rectangular zone.  ESC to cancel.",
    ),
)
_FINISH_ZONE_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "ZoneCreationInteraction", "Click to finish this zone.  ESC to cancel."
    ),
)
_CONNECT_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "ZoneCreationInteraction",
        "Connect first and last points to finish this zone.  ESC to cancel.",
    ),
)
_CREATION_MODE_TIP = cast(
    str,
    QT_TRANSLATE_NOOP(
        "ZoneCreationInteraction",
        "Use Z and X keys to switch zone creation mode.  ESC to cancel.",
    ),
)

_TAB_SHORTCUT_COUNT = 5
# Colours that tell zones apart in the menu when several lie under the cursor.
_MENU_FIRST_HUE = 20
_MENU_HUE_STEP = 65
_MENU_SATURATION = 255 * 64 // 100
_MENU_VALUE = 255 * 96 // 100
_MENU_ALPHA = 150
_NEW_PICTURE_LAYER = PictureLayer.PAINTER2
_SUBTRACTING_LAYERS = (PictureLayer.ERASER1, PictureLayer.ERASER3)


def _swatch(color: QColor, size: int) -> QtGui.QIcon:
    opaque = color.toRgb()
    opaque.setAlpha(255)
    pixmap = QPixmap(size, size)
    pixmap.fill(opaque)
    return QtGui.QIcon(pixmap)


@dataclass
class ViewZone:
    """A zone as the view shows it.

    Attributes:
        kind: Picture or fill zone.
        polygon: Vertices in virtual coordinates.
        fill_color: RGB colour a fill zone paints; unused for picture zones.
        properties: Zone properties the view edits (the layer of a picture zone,
            the colour of a fill zone) and carries along; None for a zone that
            has none yet.
    """

    kind: ZoneKind
    polygon: QPolygonF
    fill_color: tuple[int, int, int] = (255, 255, 255)
    properties: object | None = None

    def copy(self) -> "ViewZone":
        """Return a copy with its own polygon."""
        return ViewZone(
            self.kind, QPolygonF(self.polygon), self.fill_color, self.properties
        )


@dataclass(frozen=True)
class _Pick:
    """What lies under the cursor among the editable zones.

    Attributes:
        zone: Index of the zone.
        vertex: Index of the vertex, or of the first vertex of the edge.
        on_edge: True when the cursor is on an edge, not on a vertex.
        point: Position of the vertex or of the nearest point of the edge, in
            widget coordinates.
    """

    zone: int
    vertex: int
    on_edge: bool
    point: QPointF


@dataclass
class _ZoneDrag:
    """A whole zone being moved."""

    zone: int
    last: QPointF
    moved: bool = False


class OutputImageView(ImageViewBase):
    """Image view of the Output stage.

    Zones are polygons in virtual coordinates. A zone is created in one of
    three modes, chosen with Z (polygonal), X (lasso) and C (rectangular): by
    clicking its corners and finishing with a double click, Enter or a click on
    the first corner, by dragging a freehand outline, or by dragging a
    rectangle. The mode a new view starts in is the application's default zone
    creation mode.

    A zone is reshaped by dragging a corner (Ctrl keeps the corner square,
    dropping it on a neighbour merges the two), by pressing on an edge to add a
    corner there, and by D over a corner to remove it. Shift drags the whole
    zone, Shift+Ctrl drags a copy. The right button opens a menu of the zones
    under the cursor, from which a picture zone's layer is set and a fill zone's
    colour is picked from the page. Delete removes a zone.

    ``set_tabs`` splits the view into upstream's tabs. A zone tab shows and
    edits one kind of zone and starts a zone with a click anywhere that is not a
    corner or an edge; Ctrl+Alt+click copies the latest zone of that kind there.
    Without tabs both kinds are edited together, a zone must be selected before
    its corners react, and zones are started from the menu.

    Optional guides, a dashed cross through the centre of the view, help to
    judge the alignment of the output. The choice is kept in the application
    settings under ``SHOW_GUIDES_KEY``.

    Signals:
        zones_changed: The user created, reshaped, removed or recoloured a zone.
        tab_changed: Another tab was chosen (ImageViewTab).
        distortion_model_changed: The user reshaped a page edge in the dewarping
            tab (DistortionModelSpec).
    """

    zones_changed = Signal()
    tab_changed = Signal(ImageViewTab)
    distortion_model_changed = Signal(object)

    SHOW_GUIDES_KEY = "output/showGuides"

    VERTEX_HIT_RADIUS = 8
    MIN_ZONE_VERTICES = geometry.MIN_VERTICES
    LASSO_STEP = 4
    SELECTED_FILL_ALPHA = 80
    FILL_PREVIEW_ALPHA = 128
    MENU_SWATCH_SIZE = 16

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the view.

        Args:
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent=parent)
        self._zones: list[ViewZone] = []
        self._selected = -1
        self._creating: ZoneKind | None = None
        self._new_points: list[QPointF] = []
        self._creation_mode = (
            ApplicationSettings.get_instance().default_zone_creation_mode
        )
        self._outlining = False
        self._outline_anchor = QPointF()
        self._dragged_vertex = -1
        self._drag_offset = QPointF()
        self._drag_moved = False
        self._zone_drag: _ZoneDrag | None = None
        self._cursor = QPointF(-1, -1)
        self._hover: _Pick | None = None
        self._menu_highlight: tuple[int, QColor] | None = None
        self._default_layer = _NEW_PICTURE_LAYER
        self._default_fill_color = (255, 255, 255)
        self._show_guides = bool(
            QSettings().value(self.SHOW_GUIDES_KEY, False, type=bool)
        )

        self._tab: ImageViewTab | None = None
        self._tabs: list[ImageViewTab] = []
        self._main_image: QImage | None = None
        self._tab_images: dict[ImageViewTab, QImage] = {}
        self._tab_bar = QtWidgets.QTabBar(self)
        self._tab_bar.setObjectName("outputTabBar")
        self._tab_bar.setExpanding(False)
        self._tab_bar.setUsesScrollButtons(False)
        self._tab_bar.setElideMode(Qt.TextElideMode.ElideRight)
        self._tab_bar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._tab_bar.hide()
        self._tab_bar.currentChanged.connect(self._on_tab_bar_changed)
        self._tab_shortcuts = [
            QShortcut(QKeySequence(f"Ctrl+{number}"), self)
            for number in range(1, _TAB_SHORTCUT_COUNT + 1)
        ]
        for index, shortcut in enumerate(self._tab_shortcuts):
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(lambda index=index: self._choose_tab_at(index))

        self._tab_messages: dict[ImageViewTab, str] = {}
        self._message = QtWidgets.QLabel(self)
        self._message.setObjectName("outputTabMessage")
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setWordWrap(True)
        self._message.setAutoFillBackground(True)
        self._message.hide()
        self._dewarp_note = QtWidgets.QLabel(self)
        self._dewarp_note.setObjectName("dewarpMetricsNote")
        self._dewarp_note.setWordWrap(True)
        self._dewarp_note.setAutoFillBackground(True)
        self._dewarp_note.setMargin(4)
        self._dewarp_note.hide()
        self._dewarp = DewarpEditor(self, self.distortion_model_changed.emit)
        self._pickup = ColorPickupInteraction(self)
        self._pickup.color_picked.connect(self._apply_picked_color)
        self._pickup.cancelled.connect(self._show_status_tip_here)
        self._pickup_zone = -1
        self._despeckle = DespecklePreviewer(self)
        self._despeckle.preview_ready.connect(self._show_despeckle_preview)
        self._despeckle_source: (
            tuple[NDArray[np.uint8], int, NDArray[np.uint8] | None] | None
        ) = None
        self._picture_mask = PictureMaskOverlay(self, self.update)
        self._mask_zones_key: tuple[object, ...] = ()
        self.setMouseTracking(True)

    # Guides

    def guides_shown(self) -> bool:
        """Return whether the centre guides are drawn.

        Returns:
            bool: True while the guides show.
        """
        return self._show_guides

    def set_guides_shown(self, shown: bool) -> None:
        """Show or hide the centre guides and remember the choice.

        Args:
            shown (bool): True to draw the guides.
        """
        self._show_guides = shown
        QSettings().setValue(self.SHOW_GUIDES_KEY, shown)
        self.update()

    # Images and tabs

    def set_image(
        self,
        image: QImage | NDArray[np.uint8],
        downscaled_image: QImage | None = None,
        transformation: ImageTransformation | None = None,
        keep_view: bool = False,
    ) -> None:
        """Set the output image.

        A tab that has an image of its own keeps showing that one.

        Args:
            image (QImage | NDArray[np.uint8]): The output page.
            downscaled_image (QImage | None): Optional pre-downscaled version.
            transformation (ImageTransformation | None): Maps image to virtual
                coordinates; identity when None.
            keep_view (bool): Keep the zoom and the part of the image in view.
        """
        if isinstance(image, np.ndarray):
            image = numpy_to_qimage(image)
        self._main_image = image
        if self._tab not in self._tab_images:
            super().set_image(image, downscaled_image, transformation, keep_view)

    def set_tab_image(
        self, tab: ImageViewTab, image: QImage | NDArray[np.uint8] | None
    ) -> None:
        """Give a tab an image other than the output page, or take it away.

        Upstream's dewarping and despeckling tabs show other images than the
        output.

        Args:
            tab (ImageViewTab): The tab.
            image (QImage | NDArray[np.uint8] | None): The image; None goes back
                to the output page.
        """
        if image is None:
            self._tab_images.pop(tab, None)
        else:
            if isinstance(image, np.ndarray):
                image = numpy_to_qimage(image)
            self._tab_images[tab] = image
        if tab == self._tab:
            self._show_tab_image(tab)

    def set_tab_message(self, tab: ImageViewTab, text: str | None) -> None:
        """Show a message in place of a tab's image, or take it away.

        Args:
            tab (ImageViewTab): The tab.
            text (str | None): The message; None shows the image again.
        """
        if text:
            self._tab_messages[tab] = text
        else:
            self._tab_messages.pop(tab, None)
        self._update_message()

    def _update_message(self) -> None:
        text = self._tab_messages.get(self._tab) if self._tab is not None else None
        self._message.setText(text or "")
        self._message.setVisible(bool(text))
        top = self._tab_bar.height() if self._tabs else 0
        self._message.setGeometry(0, top, self.width(), max(0, self.height() - top))

    def _show_tab_image(self, tab: ImageViewTab | None) -> None:
        image = self._main_image
        if tab is not None:
            image = self._tab_images.get(tab, image)
        if image is None or image is self._image:
            return
        same_size = self._image is not None and image.size() == self._image.size()
        ImageViewBase.set_image(self, image, keep_view=same_size)

    def set_tabs(
        self, tabs: Sequence[ImageViewTab], current: ImageViewTab | None = None
    ) -> None:
        """Show a tab bar with the given tabs, or hide it with none.

        Args:
            tabs (Sequence[ImageViewTab]): The tabs, left to right. Ctrl+1 to
                Ctrl+5 choose the first five.
            current (ImageViewTab | None): The tab to start on; the first one when
                None or not among ``tabs``.
        """
        self._leave_interactions()
        self._tabs = list(tabs)
        self._tab = None
        bar = self._tab_bar
        bar.blockSignals(True)
        while bar.count():
            bar.removeTab(0)
        for tab in self._tabs:
            bar.addTab(
                icon(_TAB_ICONS[tab]),
                QCoreApplication.translate("output::Task::UiUpdater", TAB_LABELS[tab]),
            )
        bar.blockSignals(False)
        bar.setVisible(bool(self._tabs))
        self.setStatusTip(
            QCoreApplication.translate(
                "output::TabbedImageView", "Use Ctrl+1..5 to switch the tabs."
            )
            if self._tabs
            else ""
        )
        self._fit_tab_bar()
        if self._tabs:
            start = current if current in self._tabs else self._tabs[0]
            assert start is not None
            self.set_current_tab(start)
        else:
            self.default_status_tip = DEFAULT_STATUS_TIP
            self.default_status_tip_context = "ImageViewBase"
            self.set_status_tip("")
            self._show_tab_image(None)
            self._update_mask_pulse()
            self.update()

    def tabs(self) -> list[ImageViewTab]:
        """Return the tabs shown, left to right; empty without a tab bar.

        Returns:
            list[ImageViewTab]: The tabs.
        """
        return list(self._tabs)

    def current_tab(self) -> ImageViewTab | None:
        """Return the active tab.

        Returns:
            ImageViewTab | None: The tab, None without a tab bar.
        """
        return self._tab

    def set_current_tab(self, tab: ImageViewTab) -> None:
        """Switch to a tab; ignored when the view has no such tab.

        Args:
            tab (ImageViewTab): The tab to show.
        """
        if tab not in self._tabs:
            return
        self._tab_bar.blockSignals(True)
        self._tab_bar.setCurrentIndex(self._tabs.index(tab))
        self._tab_bar.blockSignals(False)
        self._enter_tab(tab)

    def _choose_tab_at(self, index: int) -> None:
        if 0 <= index < len(self._tabs):
            self.set_current_tab(self._tabs[index])

    def _on_tab_bar_changed(self, index: int) -> None:
        if 0 <= index < len(self._tabs):
            self._enter_tab(self._tabs[index])

    def _enter_tab(self, tab: ImageViewTab) -> None:
        if tab == self._tab:
            return
        self._leave_interactions()
        self._tab = tab
        if tab in _ZONE_TAB_KINDS:
            self.default_status_tip = _ZONE_DEFAULT_TIP
            self.default_status_tip_context = "ZoneDefaultInteraction"
        else:
            self.default_status_tip = DEFAULT_STATUS_TIP
            self.default_status_tip_context = "ImageViewBase"
        self.set_status_tip("")
        self._show_tab_image(tab)
        self._update_message()
        self._update_dewarp_note()
        self._update_mask_pulse()
        self.update()
        self.tab_changed.emit(tab)

    def _leave_interactions(self) -> None:
        self.cancel_zone_creation()
        self._pickup.cancel()
        self._dragged_vertex = -1
        self._zone_drag = None
        self._hover = None

    def _fit_tab_bar(self) -> None:
        height = self._tab_bar.sizeHint().height() if self._tabs else 0
        self.setViewportMargins(0, height, 0, 0)
        self._tab_bar.setGeometry(0, 0, self.width(), height)
        if self._fit_mode:
            self._fit_to_widget()
        else:
            self._update_widget_transform()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Keep the tab bar across the top.

        Args:
            event (QtGui.QResizeEvent): The resize event.
        """
        super().resizeEvent(event)
        self._tab_bar.setGeometry(0, 0, self.width(), self._tab_bar.height())
        self._update_message()
        self._update_dewarp_note()

    # Dewarping tab

    def set_dewarp_note(self, text: str | None) -> None:
        """Show a line about the automatic model along the bottom of the dewarping tab.

        Args:
            text (str | None): What automatic dewarping found on the page; None
                or empty shows nothing.
        """
        self._dewarp_note.setText(text or "")
        self._update_dewarp_note()

    def dewarp_note(self) -> str:
        """Return the line shown about the automatic model, empty when none.

        Returns:
            str: The text of the note.
        """
        return self._dewarp_note.text()

    def _update_dewarp_note(self) -> None:
        note = self._dewarp_note
        shown = self._tab is ImageViewTab.DEWARPING and bool(note.text())
        note.setVisible(shown)
        if shown:
            height = note.heightForWidth(self.width())
            note.setGeometry(0, self.height() - height, self.width(), height)
            note.raise_()

    def set_distortion_model(
        self,
        model: DistortionModelSpec,
        depth_perception: float,
        default_area: QRectF | None = None,
    ) -> None:
        """Show a distortion model in the dewarping tab without reporting a change.

        Args:
            model (DistortionModelSpec): Edges in the pixels of the dewarping
                tab's image. When it is not defined the grid starts from straight
                edges along ``default_area``.
            depth_perception (float): Distance to the page plane, 1 to 3.
            default_area (QRectF | None): Where straight edges start; the whole
                image when None.
        """
        self._dewarp.set_distortion_model(model, depth_perception, default_area)

    def set_depth_perception(self, depth_perception: float) -> None:
        """Redraw the dewarping grid for another depth perception.

        Args:
            depth_perception (float): Distance to the page plane, 1 to 3.
        """
        self._dewarp.set_depth_perception(depth_perception)

    def distortion_model(self) -> DistortionModelSpec:
        """Return the model as the edges are now.

        Returns:
            DistortionModelSpec: Edges sampled from the curves.
        """
        return self._dewarp.distortion_model()

    def control_points(self) -> tuple[list[QPointF], list[QPointF]]:
        """Return the control points of the top and bottom edge.

        Returns:
            tuple[list[QPointF], list[QPointF]]: Copies, in image pixels.
        """
        return self._dewarp.control_points()

    # Despeckling tab

    def set_despeckle_source(
        self,
        binary: NDArray[np.uint8] | None,
        dpi: int,
        level: DespeckleLevel | DespeckleStrength,
        page: NDArray[np.uint8] | None = None,
    ) -> None:
        """Give the despeckling tab the page as it is before despeckling.

        Args:
            binary (NDArray[np.uint8] | None): Bilevel page, 0 is ink, not changed
                afterwards by the caller; None removes the preview.
            dpi (int): Resolution of the page.
            level (DespeckleLevel | DespeckleStrength):
                The level or in-between strength to preview.
            page (NDArray[np.uint8] | None): The output page whose pictures the
                preview shows, gray or RGB; ignored when its size differs from
                ``binary``. A white page when None.
        """
        if binary is None:
            self._despeckle_source = None
            self.set_tab_image(ImageViewTab.DESPECKLING, None)
            return
        if page is not None and page.shape[:2] != binary.shape:
            page = None
        self._despeckle_source = (binary, dpi, page)
        self._despeckle.request(binary, dpi, level, page)

    def set_despeckle_level(self, level: DespeckleLevel | DespeckleStrength) -> None:
        """Preview another level on the page given to ``set_despeckle_source``.

        Args:
            level (DespeckleLevel | DespeckleStrength):
                The level or in-between strength to preview.
        """
        if self._despeckle_source is not None:
            binary, dpi, page = self._despeckle_source
            self._despeckle.request(binary, dpi, level, page)

    def _show_despeckle_preview(self, preview: NDArray[np.uint8]) -> None:
        self.set_tab_image(ImageViewTab.DESPECKLING, preview)

    # Picture mask

    def set_picture_mask(self, mask: NDArray[np.bool_] | None) -> None:
        """Show what the output keeps as pictures in the Picture Zones tab.

        Args:
            mask (NDArray[np.bool_] | None): True where a picture is kept, in the
                pixels of the Picture Zones tab's image; None removes the mask.
        """
        self._mask_zones_key = ()
        self._picture_mask.set_mask(mask)
        self._update_mask_pulse()

    def picture_mask_phase(self) -> int:
        """Return where the picture mask's pulse is, in degrees."""
        return self._picture_mask.phase

    def _update_mask_pulse(self) -> None:
        self._picture_mask.set_running(
            self._tab is ImageViewTab.PICTURE_ZONES and self.isVisible()
        )

    def _sync_picture_mask_zones(self) -> None:
        """Hand the picture zones to the mask overlay when they changed."""
        zones = [
            (self._layer_of(zone), tuple(zone.polygon.toList()))
            for zone in self._zones
            if zone.kind is ZoneKind.PICTURE
        ]
        key = tuple((layer, tuple((p.x(), p.y()) for p in pts)) for layer, pts in zones)
        if key != self._mask_zones_key:
            self._mask_zones_key = key
            self._picture_mask.set_zones(
                [MaskZone(QPolygonF(list(pts)), layer) for layer, pts in zones],
                repaint=False,
            )

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Let the picture mask pulse again.

        Args:
            event (QtGui.QShowEvent): The show event.
        """
        super().showEvent(event)
        self._update_mask_pulse()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        """Stop the picture mask pulse while nothing is shown.

        Args:
            event (QtGui.QHideEvent): The hide event.
        """
        super().hideEvent(event)
        self._picture_mask.set_running(False)

    # Zones

    def set_zones(self, zones: list[ViewZone]) -> None:
        """Show zones without reporting a change.

        Args:
            zones (list[ViewZone]): Zones in virtual coordinates.
        """
        self._zones = [zone.copy() for zone in zones]
        self._selected = -1
        self._dragged_vertex = -1
        self._zone_drag = None
        self._hover = None
        self._pickup.cancel()
        self.cancel_zone_creation()

    def zones(self) -> list[ViewZone]:
        """Return the zones as shown.

        Returns:
            list[ViewZone]: Copies of the zones in virtual coordinates.
        """
        return [zone.copy() for zone in self._zones]

    def default_picture_layer(self) -> PictureLayer:
        """Return the layer a new picture zone gets.

        Returns:
            PictureLayer: The layer last chosen in the properties dialog.
        """
        return self._default_layer

    def default_fill_color(self) -> tuple[int, int, int]:
        """Return the colour a new fill zone gets.

        Returns:
            tuple[int, int, int]: The colour last picked.
        """
        return self._default_fill_color

    def creation_mode(self) -> ZoneCreationMode:
        """Return how new zones are drawn.

        Returns:
            ZoneCreationMode: The mode the next zone is created in.
        """
        return self._creation_mode

    def set_creation_mode(self, mode: ZoneCreationMode) -> None:
        """Choose how new zones are drawn and make it the application default.

        A zone being drawn in another mode is dropped.

        Args:
            mode (ZoneCreationMode): The mode for the zones created from now on.
        """
        if mode is self._creation_mode:
            return
        self._creation_mode = mode
        if self._creating is not None:
            self._new_points = []
            self._outlining = False
            self.update()
        settings = ApplicationSettings.get_instance()
        settings.default_zone_creation_mode = mode
        settings.save()

    def start_zone_creation(self, kind: ZoneKind) -> None:
        """Start collecting the corners of a new zone.

        Args:
            kind (ZoneKind): Kind of zone to create.
        """
        self._creating = kind
        self._new_points = []
        self.setCursor(Qt.CursorShape.CrossCursor)

    def cancel_zone_creation(self) -> None:
        """Drop the zone being created."""
        self._creating = None
        self._new_points = []
        self._outlining = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def _new_zone(self, kind: ZoneKind, polygon: QPolygonF) -> ViewZone:
        if kind is ZoneKind.PICTURE:
            return ViewZone(
                kind,
                polygon,
                properties=PictureZoneProperties(layer=self._default_layer),
            )
        color = self._default_fill_color
        return ViewZone(kind, polygon, color, FillZoneProperties(color=color))

    def _finish_zone_creation(self) -> None:
        kind = self._creating
        points = self._new_points
        self.cancel_zone_creation()
        if kind is not None and len(points) >= self.MIN_ZONE_VERTICES:
            self._zones.append(self._new_zone(kind, QPolygonF(points)))
            self._selected = len(self._zones) - 1
            self.zones_changed.emit()

    def remove_zone(self, index: int) -> None:
        """Delete a zone.

        Args:
            index (int): Index of the zone in ``zones()``.
        """
        if not 0 <= index < len(self._zones):
            return
        del self._zones[index]
        if self._selected == index:
            self._selected = -1
        elif self._selected > index:
            self._selected -= 1
        self._hover = None
        self.update()
        self.zones_changed.emit()

    def remove_selected_zone(self) -> None:
        """Delete the selected zone."""
        self.remove_zone(self._selected)

    def clear_zones(self) -> None:
        """Delete every zone."""
        if self._zones:
            self._zones.clear()
            self._selected = -1
            self._hover = None
            self.update()
            self.zones_changed.emit()

    def start_color_pickup(self, index: int) -> None:
        """Let the user click a colour in the view for a fill zone.

        Args:
            index (int): Index of the fill zone in ``zones()``.
        """
        if not 0 <= index < len(self._zones):
            return
        if self._zones[index].kind is not ZoneKind.FILL:
            return
        self._pickup_zone = index
        self._pickup.start(self._cursor.toPoint() if self._cursor.x() >= 0 else None)
        self.set_status_tip(
            QCoreApplication.translate("ColorPickupInteraction", _PICKUP_TIP)
        )

    def color_pickup_active(self) -> bool:
        """Return whether the view waits for a colour to be clicked.

        Returns:
            bool: True between "Pick color" and the click or Escape.
        """
        return self._pickup.is_active()

    def pick_color_at(self, widget_pos: QPointF) -> None:
        """Pick the colour under a position for the zone being recoloured.

        Args:
            widget_pos (QPointF): Position in viewport coordinates.
        """
        if self._pickup.is_active():
            self._pickup.pick_at(widget_pos.toPoint())

    def _apply_picked_color(self, color: QColor) -> None:
        index, self._pickup_zone = self._pickup_zone, -1
        if not 0 <= index < len(self._zones):
            return
        rgb = (color.red(), color.green(), color.blue())
        zone = self._zones[index]
        zone.fill_color = rgb
        zone.properties = (
            zone.properties.model_copy(update={"color": rgb})
            if isinstance(zone.properties, FillZoneProperties)
            else FillZoneProperties(color=rgb)
        )
        self._default_fill_color = rgb
        self._show_status_tip_here()
        self.update()
        self.zones_changed.emit()

    def _layer_of(self, zone: ViewZone) -> PictureLayer:
        if isinstance(zone.properties, PictureZoneProperties):
            return zone.properties.layer
        return self._default_layer

    def edit_zone_properties(self, index: int) -> bool:
        """Ask for the layer of a picture zone and apply it.

        The zone shows the choice while the dialog is open and goes back to its
        old layer when the dialog is cancelled.

        Args:
            index (int): Index of the picture zone in ``zones()``.

        Returns:
            bool: True when the layer was changed.
        """
        if not 0 <= index < len(self._zones):
            return False
        zone = self._zones[index]
        if zone.kind is not ZoneKind.PICTURE:
            return False
        before = zone.properties
        base = (
            before
            if isinstance(before, PictureZoneProperties)
            else PictureZoneProperties(layer=self._default_layer)
        )
        original_layer = base.layer

        def show(layer: PictureLayer) -> None:
            zone.properties = base.model_copy(update={"layer": layer})
            self.update()

        dialog = PictureZoneDialog(base.layer, self)
        dialog.layer_changed.connect(show)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            zone.properties = before
            self.update()
            return False
        show(dialog.layer())
        self._default_layer = dialog.layer()
        changed = dialog.layer() != original_layer
        if changed:
            self.zones_changed.emit()
        return changed

    # Status tips

    def _show_status_tip_here(self) -> None:
        """Show the hint for the pointer's last position."""
        self.set_status_tip(
            self._status_tip_at(self._cursor, Qt.KeyboardModifier.NoModifier)
        )

    def _status_tip_at(
        self, widget_pos: QPointF, modifiers: Qt.KeyboardModifier
    ) -> str:
        if self._pickup.is_active():
            return QCoreApplication.translate("ColorPickupInteraction", _PICKUP_TIP)
        if self._tab is ImageViewTab.DEWARPING:
            return self._dewarp.status_tip_at(widget_pos)
        if not self._active_kinds():
            return ""
        if self._creating is not None:
            return self._creation_tip(widget_pos)
        pick = self._pick(widget_pos)
        if pick is not None:
            tip = _SEGMENT_TIP if pick.on_edge else _VERTEX_TIP
            return QCoreApplication.translate("ZoneDefaultInteraction", tip)
        if self._zone_at(widget_pos) < 0:
            return ""
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                return QCoreApplication.translate(
                    "ZoneDefaultInteraction", _ZONE_COPY_TIP
                )
            return QCoreApplication.translate("ZoneDefaultInteraction", _ZONE_DRAG_TIP)
        return QCoreApplication.translate("ZoneDefaultInteraction", _ZONE_AREA_TIP)

    def _creation_tip(self, widget_pos: QPointF) -> str:
        if self._creation_mode is ZoneCreationMode.RECTANGULAR:
            if not self._outlining:
                return ""
            return QCoreApplication.translate(
                "ZoneCreationInteraction", _FINISH_RECTANGLE_TIP
            )
        if len(self._new_points) >= 2:
            if self._closes_polygon(widget_pos):
                return QCoreApplication.translate(
                    "ZoneCreationInteraction", _FINISH_ZONE_TIP
                )
            return QCoreApplication.translate("ZoneCreationInteraction", _CONNECT_TIP)
        return QCoreApplication.translate("ZoneCreationInteraction", _CREATION_MODE_TIP)

    # Hit testing

    def _active_kinds(self) -> frozenset[ZoneKind]:
        if self._tab is None:
            return frozenset(ZoneKind)
        kind = _ZONE_TAB_KINDS.get(self._tab)
        return frozenset() if kind is None else frozenset({kind})

    def _tab_kind(self) -> ZoneKind | None:
        """The kind a click starts in the active zone tab, None without one."""
        return None if self._tab is None else _ZONE_TAB_KINDS.get(self._tab)

    def _widget_corners(self, index: int) -> list[QPointF]:
        return [
            self.virtual_to_widget_point(point)
            for point in self._zones[index].polygon.toList()
        ]

    def _zone_at(self, widget_pos: QPointF) -> int:
        virtual = self.widget_to_virtual_point(widget_pos)
        kinds = self._active_kinds()
        for index in reversed(range(len(self._zones))):
            zone = self._zones[index]
            if zone.kind in kinds and zone.polygon.containsPoint(
                virtual, Qt.FillRule.OddEvenFill
            ):
                return index
        return -1

    def _zones_under(self, widget_pos: QPointF) -> list[int]:
        """Return every editable zone under a position, the smallest first."""
        virtual = self.widget_to_virtual_point(widget_pos)
        kinds = self._active_kinds()
        found = [
            index
            for index, zone in enumerate(self._zones)
            if zone.kind in kinds
            and zone.polygon.containsPoint(virtual, Qt.FillRule.OddEvenFill)
        ]

        def area(index: int) -> float:
            rect = self._zones[index].polygon.boundingRect()
            return rect.width() * rect.height()

        return sorted(found, key=area)

    def _pickable_zones(self) -> list[int]:
        """The zones whose corners and edges react to the mouse."""
        if self._tab is None:
            return [self._selected] if 0 <= self._selected < len(self._zones) else []
        kinds = self._active_kinds()
        return [i for i, zone in enumerate(self._zones) if zone.kind in kinds]

    def _pick(self, widget_pos: QPointF) -> _Pick | None:
        """Find the corner, else the edge, closest to a position."""
        corners_hit: list[_Pick] = []
        edges_hit: list[_Pick] = []
        for index in self._pickable_zones():
            corners = self._widget_corners(index)
            vertex = geometry.vertex_near(corners, widget_pos, self.VERTEX_HIT_RADIUS)
            if vertex >= 0:
                corners_hit.append(_Pick(index, vertex, False, corners[vertex]))
            edge = geometry.segment_near(corners, widget_pos, self.VERTEX_HIT_RADIUS)
            if edge is not None:
                edges_hit.append(_Pick(index, edge[0], True, edge[1]))
        candidates = corners_hit or edges_hit
        if not candidates:
            return None
        return min(
            candidates, key=lambda pick: geometry.distance(pick.point, widget_pos)
        )

    # Painting

    def _paint_overlay(self, painter: QPainter) -> None:
        if self._tab is ImageViewTab.DEWARPING:
            self._dewarp.paint(painter)
        elif not self._pickup.is_active():
            if self._tab is ImageViewTab.PICTURE_ZONES:
                self._sync_picture_mask_zones()
                self._picture_mask.paint(painter, self.image_to_widget())
            kinds = self._active_kinds()
            for index, zone in enumerate(self._zones):
                if zone.kind in kinds:
                    self._paint_zone(painter, zone, index == self._selected)
            self._paint_menu_highlight(painter)
            if self._creating is not None and self._new_points:
                self._paint_new_zone(painter)
            self._paint_hover(painter)
        if self._show_guides:
            self._paint_guides(painter)
        self._pickup.paint_overlay(painter)

    def _paint_guides(self, painter: QPainter) -> None:
        viewport = self.viewport()
        assert viewport is not None
        centre = QPointF(viewport.width() / 2, viewport.height() / 2)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        pen = QPen(view_color("guide"))
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QLineF(centre.x(), 0, centre.x(), viewport.height()))
        painter.drawLine(QLineF(0, centre.y(), viewport.width(), centre.y()))
        painter.restore()

    @staticmethod
    def _zone_colors(kind: ZoneKind) -> tuple[QColor, QColor]:
        role = "zone_picture" if kind == ZoneKind.PICTURE else "zone_fill"
        return view_color(f"{role}_border"), view_color(f"{role}_fill")

    def _paint_zone(self, painter: QPainter, zone: ViewZone, selected: bool) -> None:
        border, fill = self._zone_colors(zone.kind)
        layer = self._layer_of(zone)
        if zone.kind == ZoneKind.FILL:
            fill = QColor(*zone.fill_color, self.FILL_PREVIEW_ALPHA)
        if selected:
            border = view_color("zone_selected")
            fill = QColor(fill)
            fill.setAlpha(self.SELECTED_FILL_ALPHA)
        polygon = self._virtual_to_widget.map(zone.polygon)
        subtracts = zone.kind is ZoneKind.PICTURE and layer in _SUBTRACTING_LAYERS

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(
            QBrush(border, Qt.BrushStyle.BDiagPattern) if subtracts else fill
        )
        painter.setPen(QPen(border, 2 if selected else 1))
        painter.drawPolygon(polygon)
        if selected:
            painter.setBrush(view_color("zone_vertex"))
            painter.setPen(Qt.PenStyle.NoPen)
            for point in polygon.toList():
                painter.drawEllipse(point, 5, 5)
        if zone.kind is ZoneKind.PICTURE:
            self._paint_layer_label(painter, polygon, layer, border)
        painter.restore()

    @staticmethod
    def _paint_layer_label(
        painter: QPainter, polygon: QPolygonF, layer: PictureLayer, color: QColor
    ) -> None:
        text = layer_label(layer)
        if not text:
            return
        font = painter.font()
        font.setPixelSize(11)
        painter.setFont(font)
        bounds = polygon.boundingRect()
        if painter.fontMetrics().horizontalAdvance(text) + 8 > bounds.width():
            return
        painter.setPen(color)
        painter.drawText(bounds, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_menu_highlight(self, painter: QPainter) -> None:
        if self._menu_highlight is None:
            return
        index, color = self._menu_highlight
        if not 0 <= index < len(self._zones):
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(color)
        painter.setPen(QPen(view_color("zone_selected"), 2))
        painter.drawPolygon(self._virtual_to_widget.map(self._zones[index].polygon))
        painter.restore()

    def _paint_hover(self, painter: QPainter) -> None:
        pick = self._hover
        if pick is None or self._dragging() or self._creating is not None:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(view_color("zone_vertex"))
        painter.setPen(QPen(view_color("zone_selected"), 2))
        radius = 4 if pick.on_edge else 7
        painter.drawEllipse(pick.point, radius, radius)
        painter.restore()

    def _paint_new_zone(self, painter: QPainter) -> None:
        assert self._creating is not None
        border = self._zone_colors(self._creating)[0]
        points = [self.virtual_to_widget_point(p) for p in self._new_points]
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(border, 2, Qt.PenStyle.DashLine))
        for start, end in pairwise(points):
            painter.drawLine(start, end)
        if len(points) >= 2:
            painter.setPen(QPen(border.lighter(), 1, Qt.PenStyle.DotLine))
            painter.drawLine(points[-1], points[0])
        if self._creation_mode is not ZoneCreationMode.LASSO:
            painter.setBrush(view_color("zone_vertex"))
            painter.setPen(Qt.PenStyle.NoPen)
            for point in points:
                painter.drawEllipse(point, 5, 5)
        painter.restore()

    # Mouse

    def _dragging(self) -> bool:
        return self._dragged_vertex >= 0 or self._zone_drag is not None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Pick a colour, edit a zone or dewarping edge, or start a zone.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if self._pickup.handle_mouse_press(event):
            return
        if self._tab is ImageViewTab.DEWARPING:
            if not self._dewarp.mouse_press(event):
                super().mousePressEvent(event)
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._active_kinds()
            and self._press_zone_tools(event)
        ):
            return
        super().mousePressEvent(event)

    def _press_zone_tools(self, event: QtGui.QMouseEvent) -> bool:
        """Handle a left press on the zone tools; False leaves it to panning."""
        pos = event.position()
        if self._creating is not None:
            return self._press_while_creating(pos, event)

        pick = self._pick(pos)
        if pick is not None:
            self._start_vertex_drag(pick, pos)
            event.accept()
            return True

        modifiers = event.modifiers()
        under = self._zone_at(pos)
        if under >= 0 and modifiers & Qt.KeyboardModifier.ShiftModifier:
            self._start_zone_drag(
                under, pos, copy=bool(modifiers & Qt.KeyboardModifier.ControlModifier)
            )
            event.accept()
            return True

        kind = self._tab_kind()
        if kind is not None:
            both = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
            if modifiers & both == both and self._copy_latest_zone(kind, pos):
                event.accept()
                return True
            self.start_zone_creation(kind)
            return self._press_while_creating(pos, event)

        if under >= 0:
            self._selected = under
            self.update()
            event.accept()
            return True
        if self._selected >= 0:
            self._selected = -1
            self.update()
        return False

    def _press_while_creating(self, pos: QPointF, event: QtGui.QMouseEvent) -> bool:
        point = self.widget_to_virtual_point(pos)
        if self._creation_mode is ZoneCreationMode.POLYGONAL:
            if self._closes_polygon(pos):
                self._finish_zone_creation()
            else:
                self._new_points.append(point)
        else:
            self._outlining = True
            self._outline_anchor = point
            self._new_points = [point]
        self.set_status_tip(self._creation_tip(pos))
        self.update()
        event.accept()
        return True

    def _closes_polygon(self, widget_pos: QPointF) -> bool:
        if len(self._new_points) < self.MIN_ZONE_VERTICES:
            return False
        first = self.virtual_to_widget_point(self._new_points[0])
        return geometry.distance(first, widget_pos) <= self.VERTEX_HIT_RADIUS

    def _start_vertex_drag(self, pick: _Pick, pos: QPointF) -> None:
        self._selected = pick.zone
        self._drag_moved = False
        if pick.on_edge:
            corners = self._zones[pick.zone].polygon.toList()
            vertex = geometry.insert_vertex(
                corners, pick.vertex, self.widget_to_virtual_point(pick.point)
            )
            self._zones[pick.zone].polygon = QPolygonF(corners)
            self._drag_moved = True
        else:
            vertex = pick.vertex
        self._dragged_vertex = vertex
        corner = self._zones[pick.zone].polygon.at(vertex)
        self._drag_offset = self.widget_to_virtual_point(pos) - corner
        self._hover = None
        self.set_status_tip(_MERGE_HINT_TIP)
        self.update()

    def _start_zone_drag(self, index: int, pos: QPointF, copy: bool) -> None:
        if copy:
            self._zones.append(self._zones[index].copy())
            index = len(self._zones) - 1
        self._selected = index
        self._zone_drag = _ZoneDrag(
            index, self.widget_to_virtual_point(pos), moved=copy
        )
        self._hover = None
        self.set_status_tip(_FINISH_DRAG_TIP)
        self.update()

    def _copy_latest_zone(self, kind: ZoneKind, widget_pos: QPointF) -> bool:
        latest = next(
            (z for z in reversed(self._zones) if z.kind is kind),
            None,
        )
        if latest is None:
            return False
        copy = latest.copy()
        shift = (
            self.widget_to_virtual_point(widget_pos)
            - copy.polygon.boundingRect().center()
        )
        copy.polygon = copy.polygon.translated(shift)
        self._zones.append(copy)
        self._selected = len(self._zones) - 1
        self.update()
        self.zones_changed.emit()
        return True

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Extend the outline, move the picked corner or zone, or show what is under it.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        self._cursor = event.position()
        if self._pickup.handle_mouse_move(event):
            return
        if self._tab is ImageViewTab.DEWARPING:
            if not self._dewarp.mouse_move(event):
                super().mouseMoveEvent(event)
            return
        if self._outlining:
            self._extend_outline(self.widget_to_virtual_point(event.position()))
            event.accept()
        elif self._dragged_vertex >= 0:
            self._move_vertex(event)
            event.accept()
        elif self._zone_drag is not None:
            self._move_zone(event.position())
            event.accept()
        else:
            self._track_hover(event.position())
            super().mouseMoveEvent(event)

    def _track_hover(self, widget_pos: QPointF) -> None:
        if not self._active_kinds() or self._creating is not None:
            return
        hover = self._pick(widget_pos)
        if hover != self._hover:
            self._hover = hover
            self.update()

    def _move_vertex(self, event: QtGui.QMouseEvent) -> None:
        zone = self._zones[self._selected]
        corners = zone.polygon.toList()
        index = self._dragged_vertex
        point = self.widget_to_virtual_point(event.position()) - self._drag_offset
        if corners[index] == point and not event.modifiers():
            return
        corners[index] = point

        in_widget = [self.virtual_to_widget_point(c) for c in corners]
        neighbour = geometry.merge_target(in_widget, index, self.VERTEX_HIT_RADIUS)
        self.set_status_tip(_MERGE_HINT_TIP if neighbour is None else _MERGE_TIP)
        if neighbour is not None:
            corners[index] = corners[neighbour]
            in_widget[index] = in_widget[neighbour]
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            previous, following = index - 1, (index + 1) % len(corners)
            before, after = geometry.right_angle_neighbours(
                in_widget[index], in_widget[previous], in_widget[following]
            )
            corners[previous] = self.widget_to_virtual_point(before)
            corners[following] = self.widget_to_virtual_point(after)

        zone.polygon = QPolygonF(corners)
        self._drag_moved = True
        self.update()

    def _move_zone(self, widget_pos: QPointF) -> None:
        drag = self._zone_drag
        assert drag is not None
        here = self.widget_to_virtual_point(widget_pos)
        shift = here - drag.last
        if shift.isNull():
            return
        zone = self._zones[drag.zone]
        zone.polygon = zone.polygon.translated(shift)
        drag.last = here
        drag.moved = True
        self.update()

    def _extend_outline(self, point: QPointF) -> None:
        if self._creation_mode is ZoneCreationMode.RECTANGULAR:
            anchor = self._outline_anchor
            self._new_points = [
                anchor,
                QPointF(point.x(), anchor.y()),
                point,
                QPointF(anchor.x(), point.y()),
            ]
        else:
            last = self.virtual_to_widget_point(self._new_points[-1])
            here = self.virtual_to_widget_point(point)
            if math.hypot(here.x() - last.x(), here.y() - last.y()) >= self.LASSO_STEP:
                self._new_points.append(point)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Finish a dragged outline, corner or zone and report the shape.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if self._tab is ImageViewTab.DEWARPING:
            if not self._dewarp.mouse_release(event):
                super().mouseReleaseEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._outlining:
            self._outlining = False
            if self._creation_mode is ZoneCreationMode.RECTANGULAR:
                corners = self._new_points
                if len(corners) == 4 and (
                    corners[0].x() == corners[2].x() or corners[0].y() == corners[2].y()
                ):
                    self._new_points = []
            self._finish_zone_creation()
            self.set_status_tip(
                self._status_tip_at(event.position(), event.modifiers())
            )
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._dragged_vertex >= 0:
            self._finish_vertex_drag()
            self.set_status_tip(
                self._status_tip_at(event.position(), event.modifiers())
            )
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._zone_drag is not None:
            drag, self._zone_drag = self._zone_drag, None
            self.set_status_tip(
                self._status_tip_at(event.position(), event.modifiers())
            )
            event.accept()
            if drag.moved:
                self.zones_changed.emit()
            return
        super().mouseReleaseEvent(event)

    def _finish_vertex_drag(self) -> None:
        index, self._dragged_vertex = self._dragged_vertex, -1
        zone = self._zones[self._selected]
        corners = zone.polygon.toList()
        neighbours = (corners[index - 1], corners[(index + 1) % len(corners)])
        if corners[index] in neighbours and geometry.remove_vertex(corners, index):
            zone.polygon = QPolygonF(corners)
            self._drag_moved = True
        self.update()
        if self._drag_moved:
            self.zones_changed.emit()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Finish the zone being created.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self._creating is not None:
            self._finish_zone_creation()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # Keys

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Finish, cancel, deselect or delete zones and corners.

        Enter finishes the zone being drawn, Escape drops it or the selection,
        Delete removes a zone, D removes the corner under the cursor, and Z, X
        and C choose the polygonal, lasso and rectangular creation modes.

        Args:
            event (QtGui.QKeyEvent): The key event.
        """
        if self._pickup.handle_key_press(event):
            return
        if self._tab is ImageViewTab.DEWARPING:
            if not self._dewarp.key_press(event):
                super().keyPressEvent(event)
            return
        key = event.key()
        if not self._active_kinds():
            super().keyPressEvent(event)
            return
        if key in _MODE_KEYS and not event.modifiers():
            self.set_creation_mode(_MODE_KEYS[key])
        elif key == Qt.Key.Key_Escape and self._creating is not None:
            self.cancel_zone_creation()
        elif key == Qt.Key.Key_Escape and self._selected >= 0:
            self._selected = -1
            self.update()
        elif key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self._delete_zone():
            pass
        elif key == Qt.Key.Key_D and self._remove_vertex_under_cursor():
            pass
        elif (
            key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._creating is not None
        ):
            self._finish_zone_creation()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def _delete_zone(self) -> bool:
        index = self._selected
        if self._tab is not None:
            index = self._zone_at(self._cursor)
            if index < 0:
                index = self._selected
        if index < 0:
            return False
        self.remove_zone(index)
        return True

    def _remove_vertex_under_cursor(self) -> bool:
        pick = self._pick(self._cursor)
        if pick is None or pick.on_edge:
            return False
        corners = self._zones[pick.zone].polygon.toList()
        if not geometry.remove_vertex(corners, pick.vertex):
            return False
        self._zones[pick.zone].polygon = QPolygonF(corners)
        self._hover = None
        self.update()
        self.zones_changed.emit()
        return True

    # Menu

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        """Show the zone menu.

        Args:
            event (QtGui.QContextMenuEvent): The context menu event.
        """
        if self._tab in (ImageViewTab.DEWARPING, ImageViewTab.DESPECKLING):
            return
        if self._pickup.is_active():
            self._pickup.cancel()
            return
        menu = self.build_context_menu(QPointF(event.pos()))
        menu.exec(event.globalPos())
        self._menu_highlight = None
        self.update()

    def build_context_menu(self, widget_pos: QPointF) -> QtWidgets.QMenu:
        """Build the menu for a position.

        It lists, for every zone under the position, what can be done with that
        zone. Without tabs it also offers to start zones and to clear them all.

        Args:
            widget_pos (QPointF): Position in viewport coordinates.

        Returns:
            QtWidgets.QMenu: The menu, not shown.
        """
        menu = QtWidgets.QMenu(self)
        if self._creating is not None:
            menu.addAction(self.tr("Finish Zone"), self._finish_zone_creation)
            menu.addAction(self.tr("Cancel"), self.cancel_zone_creation)
        else:
            self._add_zone_items(menu, widget_pos)
            if self._tab is None:
                self._add_creation_items(menu)
        if menu.actions():
            menu.addSeparator()
        guides = menu.addAction(icon("crosshair"), self.tr("Show guides"))
        guides.setCheckable(True)
        guides.setChecked(self._show_guides)
        guides.toggled.connect(self.set_guides_shown)
        return menu

    def _add_creation_items(self, menu: QtWidgets.QMenu) -> None:
        menu.addAction(
            self.tr("Create Picture Zone"),
            lambda: self.start_zone_creation(ZoneKind.PICTURE),
        )
        menu.addAction(
            self.tr("Create Fill Zone"), lambda: self.start_zone_creation(ZoneKind.FILL)
        )
        if self._selected >= 0:
            menu.addSeparator()
            menu.addAction(self.tr("Delete Zone"), self.remove_selected_zone)
        if self._zones:
            menu.addSeparator()
            menu.addAction(self.tr("Clear All Zones"), self.clear_zones)

    def _add_zone_items(self, menu: QtWidgets.QMenu, widget_pos: QPointF) -> None:
        under = self._zones_under(widget_pos)
        several = len(under) > 1
        for position, index in enumerate(under):
            color = QColor.fromHsv(
                (_MENU_FIRST_HUE + position * _MENU_HUE_STEP) % 360,
                _MENU_SATURATION,
                _MENU_VALUE,
                _MENU_ALPHA,
            )
            if self._zones[index].kind is ZoneKind.PICTURE:
                first = (
                    QCoreApplication.translate(
                        "ZoneContextMenuInteraction", "Properties"
                    ),
                    lambda i=index: self.edit_zone_properties(i),
                )
            else:
                first = (
                    QCoreApplication.translate("output::FillZoneEditor", "Pick color"),
                    lambda i=index: self.start_color_pickup(i),
                )
            for text, callback in (
                first,
                (
                    QCoreApplication.translate("ZoneContextMenuInteraction", "Delete"),
                    lambda i=index: self.remove_zone(i),
                ),
            ):
                action = menu.addAction(text)
                if several:
                    action.setIcon(_swatch(color, self.MENU_SWATCH_SIZE))
                    action.hovered.connect(
                        lambda i=index, c=color: self._highlight_zone(i, c)
                    )
                action.triggered.connect(lambda _checked=False, cb=callback: cb())
            if position < len(under) - 1:
                menu.addSeparator()
        if several:
            self._highlight_zone(
                under[0],
                QColor.fromHsv(
                    _MENU_FIRST_HUE, _MENU_SATURATION, _MENU_VALUE, _MENU_ALPHA
                ),
            )

    def _highlight_zone(self, index: int, color: QColor) -> None:
        self._menu_highlight = (index, color)
        self.update()


_MODE_KEYS = {
    Qt.Key.Key_Z: ZoneCreationMode.POLYGONAL,
    Qt.Key.Key_X: ZoneCreationMode.LASSO,
    Qt.Key.Key_C: ZoneCreationMode.RECTANGULAR,
}


class DewarpingImageView(ImageViewBase):
    """Shows the distortion model of a page as a grid and edits its edges.

    The model is two edges, top and bottom, each a curve through control points
    that run left to right. Dragging a control point reshapes the curve, pressing
    on a curve between points adds a point, and D removes the point under the
    cursor. Everything is in the pixel coordinates of the view's image.

    Signals:
        distortion_model_changed: The user reshaped an edge; carries the new
            DistortionModelSpec, whose edges are sampled from the curves.
    """

    distortion_model_changed = Signal(object)

    HIT_RADIUS = DewarpEditor.HIT_RADIUS
    MAX_CONTROL_POINTS = DewarpEditor.MAX_CONTROL_POINTS
    GRID_LINES = DewarpEditor.GRID_LINES

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the view.

        Args:
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent=parent)
        self._editor = DewarpEditor(self, self.distortion_model_changed.emit)
        self.setMouseTracking(True)

    def set_distortion_model(
        self,
        model: DistortionModelSpec,
        depth_perception: float,
        default_area: QRectF | None = None,
    ) -> None:
        """Show a model without reporting a change.

        Args:
            model (DistortionModelSpec): Edges in image pixels. When it is not
                defined the view starts from straight edges along the top and
                bottom of ``default_area``.
            depth_perception (float): Distance to the page plane, 1 to 3.
            default_area (QRectF | None): Where straight edges start; the whole
                image when None.
        """
        self._editor.set_distortion_model(model, depth_perception, default_area)

    def set_depth_perception(self, depth_perception: float) -> None:
        """Redraw the grid for another depth perception.

        Args:
            depth_perception (float): Distance to the page plane, 1 to 3.
        """
        self._editor.set_depth_perception(depth_perception)

    def control_points(self) -> tuple[list[QPointF], list[QPointF]]:
        """Return the control points of the top and bottom edge.

        Returns:
            tuple[list[QPointF], list[QPointF]]: Copies, in image pixels.
        """
        return self._editor.control_points()

    def distortion_model(self) -> DistortionModelSpec:
        """Return the model as the edges are now.

        Returns:
            DistortionModelSpec: Edges sampled from the curves, in image pixels.
        """
        return self._editor.distortion_model()

    def grid(self) -> NDArray[np.float64] | None:
        """Return the grid drawn over the page.

        Returns:
            NDArray[np.float64] | None: ``(columns, rows, 2)`` image points, None
                while the model gives no valid grid.
        """
        return self._editor.grid

    def _paint_overlay(self, painter: QPainter) -> None:
        self._editor.paint(painter)

    def _status_tip_at(
        self, widget_pos: QPointF, modifiers: Qt.KeyboardModifier
    ) -> str:
        return self._editor.status_tip_at(widget_pos)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Pick a control point, or add one when pressing on an edge.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if not self._editor.mouse_press(event):
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Move the picked control point.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if not self._editor.mouse_move(event):
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Drop the control point and report the new model.

        Args:
            event (QtGui.QMouseEvent): The mouse event.
        """
        if not self._editor.mouse_release(event):
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Remove the control point under the cursor with D.

        Args:
            event (QtGui.QKeyEvent): The key event.
        """
        if not self._editor.key_press(event):
            super().keyPressEvent(event)
