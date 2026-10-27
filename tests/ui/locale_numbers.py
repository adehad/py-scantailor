"""Read numbers back out of UI text in whatever locale the run uses.

Tests compare these numbers rather than the formatted text, so a run under
--qt-locale with a decimal comma checks the same values as one in English.
"""

import re

from PySide6 import QtCore


def _number_pattern(locale: QtCore.QLocale) -> re.Pattern[str]:
    minus = re.escape(locale.negativeSign())
    point = re.escape(locale.decimalPoint())
    return re.compile(rf"(?:{minus})?\d+(?:{point}\d+)?")


def shown_numbers(text: str) -> list[float]:
    """Return every number in a label's text, parsed with the default QLocale.

    Args:
        text (str): Text as a label or spin box shows it, such as "5,1 x 6,8 cm".

    Returns:
        list[float]: The numbers in the order they appear, such as [5.1, 6.8].
    """
    locale = QtCore.QLocale()
    numbers = []
    for match in _number_pattern(locale).finditer(text):
        value, ok = locale.toDouble(match.group())
        assert ok, f"{match.group()!r} in {text!r} is not a {locale.name()} number"
        numbers.append(value)
    return numbers


def without_numbers(text: str) -> str:
    """Return a label's text with its numbers taken out, for checking the words.

    Args:
        text (str): Text as a label shows it, such as "Manual (deskew: 1,5°)".

    Returns:
        str: The same text without the numbers, such as "Manual (deskew: °)".
    """
    return _number_pattern(QtCore.QLocale()).sub("", text)
