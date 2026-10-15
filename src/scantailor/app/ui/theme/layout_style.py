"""Layout margins and gaps from the spacing scale, wherever Qt defaults apply."""

from PySide6 import QtWidgets

from .stylesheet import SPACE_L, SPACE_S

_MARGINS = {
    QtWidgets.QStyle.PixelMetric.PM_LayoutLeftMargin,
    QtWidgets.QStyle.PixelMetric.PM_LayoutTopMargin,
    QtWidgets.QStyle.PixelMetric.PM_LayoutRightMargin,
    QtWidgets.QStyle.PixelMetric.PM_LayoutBottomMargin,
}
_GAPS = {
    QtWidgets.QStyle.PixelMetric.PM_LayoutHorizontalSpacing,
    QtWidgets.QStyle.PixelMetric.PM_LayoutVerticalSpacing,
}


class LayoutStyle(QtWidgets.QProxyStyle):
    """Fusion with layout margins and gaps taken from the spacing scale.

    A group box insets its contents with stylesheet padding, so its layout adds
    no margin; a window gets the dialog margin; any other widget the small step.
    Layouts that set their own values in code or in a .ui file keep them.
    """

    def pixelMetric(
        self,
        metric: QtWidgets.QStyle.PixelMetric,
        option: QtWidgets.QStyleOption | None = None,
        widget: QtWidgets.QWidget | None = None,
    ) -> int:
        """Return the spacing-scale value for layout metrics, Fusion's otherwise.

        Args:
            metric (QtWidgets.QStyle.PixelMetric): Metric asked for.
            option (QtWidgets.QStyleOption | None): Style option, if any.
            widget (QtWidgets.QWidget | None): Widget the metric is for.

        Returns:
            int: Size in logical pixels.
        """
        if metric in _MARGINS:
            if isinstance(widget, QtWidgets.QGroupBox):
                return 0
            if widget is not None and widget.isWindow():
                return SPACE_L
            return SPACE_S
        if metric in _GAPS:
            return SPACE_S
        return super().pixelMetric(metric, option, widget)
