"""UI Helpers & Utilities."""

import typing

from PySide6 import QtCore, QtWidgets

QWidgetT = typing.TypeVar("QWidgetT", bound=QtCore.QObject)


# -------------------- Overloads for get_cwidget()
@typing.overload
def get_cwidget[QWidgetT: QtCore.QObject](
    ui: QtWidgets.QWidget,
    qtype: type[QWidgetT],
    qname: str,
    *,
    required: typing.Literal[False],
) -> QWidgetT | None: ...


@typing.overload
def get_cwidget[QWidgetT: QtCore.QObject](
    ui: QtWidgets.QWidget,
    qtype: type[QWidgetT],
    qname: str,
    *,
    required: typing.Literal[True],
) -> QWidgetT: ...


@typing.overload
def get_cwidget[QWidgetT: QtCore.QObject](
    ui: QtWidgets.QWidget,
    qtype: type[QWidgetT],
    qname: str,
) -> QWidgetT: ...


# --------------------


def get_cwidget[QWidgetT: QtCore.QObject](
    ui: QtWidgets.QWidget,
    qtype: type[QWidgetT],
    qname: str,
    *,
    required: bool = True,
) -> QWidgetT | None:
    """Get the (child) widget from the loaded ``.ui`` file.

    Args:
        ui (QtWidgets.QWidget): The loaded UI file.
        qtype (QWidgetT): The Qt widget type to find in the ``.ui`` file.
        qname (str): Object name of the widget in the ``.ui``.
        required (bool): True will raise exception if not Found. Defaults to True.

    Returns:
        QWidgetT: The Widget found.
    """
    widget = typing.cast(QWidgetT, ui.findChild(qtype, qname))
    if widget is None:
        if required:
            raise RuntimeError(
                f"Required UI widget '{qname}' of type {qtype.__name__} not found."
            )
    return widget


def format_number(value: float, decimals: int | None = None) -> str:
    """Format a number for a label the way the spin boxes beside it show numbers.

    QAbstractSpinBox writes numbers in the default QLocale without group
    separators, so a label must do the same or a decimal-comma user sees two
    notations on one screen.

    Args:
        value (float): The number to show; an int works too.
        decimals (int | None): Fixed number of decimals. None gives the shortest
            text up to ten significant digits, with no trailing zeros.

    Returns:
        str: The number in the default locale's notation, such as "2,7" in
        Swedish and "2.7" in English.
    """
    locale = QtCore.QLocale()
    locale.setNumberOptions(
        locale.numberOptions() | QtCore.QLocale.NumberOption.OmitGroupSeparator
    )
    if decimals is None:
        return locale.toString(float(value), "g", 10)
    return locale.toString(float(value), "f", decimals)


def coordinate_separator() -> str:
    """Return the text that separates the two numbers of a coordinate pair.

    Returns:
        str: ", " normally; "; " when the default locale's decimal point is a
        comma, so "12,3; 45,6" cannot be read as four numbers.
    """
    return "; " if QtCore.QLocale().decimalPoint() == "," else ", "
