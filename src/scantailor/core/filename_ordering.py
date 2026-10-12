"""Natural ordering of image files, after ScanTailor's SmartFilenameOrdering."""

import os
import unicodedata
from collections.abc import Iterable
from pathlib import Path

# Digits sort before any other character at the same position.
_DIGIT_RUN = 0
_OTHER_CHAR = 1

# C++ SmartFilenameOrdering never compares two non-digit characters: its check for
# them sits inside a loop that has already ruled them out, so it steps over them in
# lockstep. Matching that keeps page order identical for projects made in C++, so
# every non-digit character is the same token.
_ANY_OTHER_CHAR = (_OTHER_CHAR, 0)

type _Token = tuple[int, int]
type SmartFilenameKey = tuple[str, tuple[_Token, ...], str]


def smart_filename_key(path: Path | str) -> SmartFilenameKey:
    """Build the sort key that orders files the way people expect.

    Files are grouped by folder first. Within a folder, runs of digits compare by
    value (``p2`` before ``p10``) and a digit sorts before any other character.
    Non-digit characters only count as "some character", so ``b1`` sorts before
    ``a2``, as in C++. Names that are equal under that comparison fall back to a
    plain comparison of the whole name, so the order is always total.

    Args:
        path (Path | str): Image file path; relative paths are made absolute.

    Returns:
        SmartFilenameKey: Key for ``sorted(..., key=...)``.
    """
    absolute = os.path.abspath(path)
    folder, name = os.path.split(absolute)
    return folder, tuple(_tokens(name)), name


def sort_smart(paths: Iterable[Path]) -> list[Path]:
    """Return ``paths`` in smart filename order.

    Args:
        paths (Iterable[Path]): Image file paths, possibly from several folders.

    Returns:
        list[Path]: A new list, ordered by folder and then natural file name.
    """
    return sorted(paths, key=smart_filename_key)


def _tokens(name: str) -> list[_Token]:
    """Split a file name into digit runs (as numbers) and one token per other char."""
    tokens: list[_Token] = []
    number: int | None = None
    for char in name:
        if char.isdecimal():
            digit = unicodedata.decimal(char)
            number = digit if number is None else number * 10 + digit
            continue
        if number is not None:
            tokens.append((_DIGIT_RUN, number))
            number = None
        tokens.append(_ANY_OTHER_CHAR)
    if number is not None:
        tokens.append((_DIGIT_RUN, number))
    return tokens
