"""Tabs that put a stage view beside the debug images of its last render."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from PySide6 import QtGui, QtWidgets

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.core.debug_images import DebugImages

# Upstream leaves the tab title untranslated.
MAIN_TAB_LABEL = "Main"


class DebugImageTab(QtWidgets.QWidget):
    """One debug image, decoded from its file only while the tab is live."""

    def __init__(self, path: Path, parent: QtWidgets.QWidget | None = None) -> None:
        """Create the tab without decoding the image.

        Args:
            path (Path): PNG file of the image; it must outlive the tab.
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self.path = path
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.view: ImageViewBase | None = None

    def set_live(self, live: bool) -> None:
        """Decode the image into a view, or drop the view to free its pixels.

        Args:
            live (bool): Whether the image should be shown.
        """
        if live and self.view is None:
            self.view = ImageViewBase(QtGui.QImage(str(self.path)), parent=self)
            self._layout.addWidget(self.view)
        elif not live and self.view is not None:
            self._layout.removeWidget(self.view)
            self.view.deleteLater()
            self.view = None


class DebugImageTabs(QtWidgets.QTabWidget):
    """A stage view as the ``Main`` tab, followed by one tab per debug image.

    The widget owns the collector it shows and closes it, deleting the files,
    when the images are replaced or taken down. At most ``MAX_LIVE_VIEWS``
    debug images are decoded at a time; the least recently shown one is
    dropped first, so a render with dozens of full-page images stays within a
    few pages of memory.
    """

    MAX_LIVE_VIEWS = 3

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Create the empty tab widget.

        Args:
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self.setDocumentMode(True)
        self._images: DebugImages | None = None
        self._main: QtWidgets.QWidget | None = None
        self._live: deque[DebugImageTab] = deque()
        self.currentChanged.connect(self._on_current_changed)

    @property
    def main(self) -> QtWidgets.QWidget | None:
        """The stage view in the first tab, or None while none is shown."""
        return self._main

    def show_images(self, main: QtWidgets.QWidget, images: DebugImages) -> None:
        """Show ``main`` and the images, taking ownership of ``images``.

        Images already shown are closed first. The current tab stays on
        ``main`` if it was there, and on the tab of the same index otherwise,
        so that a re-render after an edit keeps the step the user looks at.

        Args:
            main (QtWidgets.QWidget): The stage view; reparented into the tabs.
            images (DebugImages): The images of the view's last render.
        """
        index = self.currentIndex()
        self._remove_debug_tabs()
        if self._main is not main:
            self.take_main()
            self.insertTab(0, main, MAIN_TAB_LABEL)
            self._main = main
        self._images = images
        for item in images.images:
            self.addTab(DebugImageTab(item.path), item.label)
        self.setCurrentIndex(index if 0 < index < self.count() else 0)

    def take_main(self) -> QtWidgets.QWidget | None:
        """Remove every tab and close the images; give the stage view back.

        Returns:
            QtWidgets.QWidget | None: The stage view, now without a parent
                tab, or None when none was shown.
        """
        self._remove_debug_tabs()
        main, self._main = self._main, None
        if main is not None:
            self.removeTab(self.indexOf(main))
        return main

    def _remove_debug_tabs(self) -> None:
        self._live.clear()
        # Removing the current tab makes another one current, which would
        # decode an image that is about to go as well.
        blocked = self.blockSignals(True)
        try:
            for index in reversed(range(self.count())):
                tab = self.widget(index)
                if isinstance(tab, DebugImageTab):
                    self.removeTab(index)
                    tab.deleteLater()
        finally:
            self.blockSignals(blocked)
        if self._images is not None:
            self._images.close()
            self._images = None

    def _on_current_changed(self, index: int) -> None:
        tab = self.widget(index)
        if not isinstance(tab, DebugImageTab):
            return
        if tab in self._live:
            self._live.remove(tab)
        self._live.append(tab)
        tab.set_live(True)
        while len(self._live) > self.MAX_LIVE_VIEWS:
            self._live.popleft().set_live(False)
