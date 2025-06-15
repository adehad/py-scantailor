"""UI Helpers & Utilities."""

from __future__ import annotations

import typing

from PySide6 import QtCore, QtWidgets

QWidgetT = typing.TypeVar("QWidgetT", bound=QtCore.QObject)


# -------------------- Overloads for get_cwidget()
@typing.overload
def get_cwidget(
    ui: QtWidgets.QWidget,
    qtype: type[QWidgetT],
    qname: str,
    *,
    required: typing.Literal[False],
) -> None | QWidgetT: ...


@typing.overload
def get_cwidget(
    ui: QtWidgets.QWidget,
    qtype: type[QWidgetT],
    qname: str,
    *,
    required: typing.Literal[True],
) -> QWidgetT: ...


@typing.overload
def get_cwidget(
    ui: QtWidgets.QWidget,
    qtype: type[QWidgetT],
    qname: str,
) -> QWidgetT: ...


# --------------------


def get_cwidget(
    ui: QtWidgets.QWidget,
    qtype: type[QWidgetT],
    qname: str,
    *,
    required: bool = True,
) -> None | QWidgetT:
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
