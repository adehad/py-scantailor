"""Run Python's cyclic garbage collector on the GUI thread only, between events.

Left automatic, the collector runs on whichever thread happens to allocate when a
threshold is crossed: a render pool thread, or the GUI thread in the middle of a
Qt call that walks every widget, such as ``QApplication.setStyle``. A widget whose
last owner is an unreachable wrapper cycle is then deleted on the wrong thread, or
under Qt's feet, and the application crashes.
"""

import gc

from PySide6 import QtCore, QtWidgets

__all__ = ["GuiThreadCollector", "collect_garbage_on_the_gui_thread"]

_COLLECTOR_NAME = "scantailorGuiThreadCollector"
INTERVAL_MS = 1000


class GuiThreadCollector(QtCore.QObject):
    """Owns the collection timer; automatic collection is off while it runs."""

    def __init__(self, app: QtWidgets.QApplication) -> None:
        """Create the collector, stopped.

        Args:
            app (QtWidgets.QApplication): The running application; it owns the
                collector, so the timer fires on the GUI thread.
        """
        super().__init__(app)
        self.setObjectName(_COLLECTOR_NAME)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(INTERVAL_MS)
        self._timer.timeout.connect(self.collect_if_due)

    def start(self) -> None:
        """Turn automatic collection off and collect from the timer instead."""
        gc.disable()
        self._timer.start()

    @staticmethod
    def collect_if_due() -> int:
        """Collect when as many objects are pending as would start an automatic run.

        Returns:
            int: Unreachable objects found, 0 when no collection was due.
        """
        young, young_collections, _ = gc.get_count()
        young_threshold, young_collections_threshold, _ = gc.get_threshold()
        if young <= young_threshold:
            return 0
        # A young collection misses cycles that already survived one, so every
        # so often the whole heap is collected instead.
        if young_collections >= young_collections_threshold:
            return gc.collect()
        return gc.collect(0)

    def stop(self) -> None:
        """Stop the timer and hand collection back to Python."""
        self._timer.stop()
        gc.enable()


def collect_garbage_on_the_gui_thread(
    app: QtWidgets.QApplication,
) -> GuiThreadCollector:
    """Make the GUI thread the only one that collects cyclic garbage.

    Safe to call again; the collector already installed is started and returned.

    Args:
        app (QtWidgets.QApplication): The running application.

    Returns:
        GuiThreadCollector: The collector, owned by ``app``.
    """
    collector = app.findChild(GuiThreadCollector, _COLLECTOR_NAME)
    if collector is None:
        collector = GuiThreadCollector(app)
    collector.start()
    return collector
