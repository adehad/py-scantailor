"""Carry a source PDF's bookmarks and page labels onto an exported PDF.

Both refer to pages by position, so they are only meaningful when the output has
the same pages as the source. Bookmarks always open their page at the full-page
view, because the coordinates of an original view no longer match the processed
page.
"""

from pathlib import Path
from typing import Any

import pikepdf
from loguru import logger
from pikepdf import Name, OutlineItem, PageLocation

_FIT: PageLocation = PageLocation.Fit


def carry_outline_and_labels(source_path: Path, target: pikepdf.Pdf) -> bool:
    """Copy the bookmarks and page labels of a PDF into ``target``.

    Args:
        source_path (Path): The source PDF.
        target (pikepdf.Pdf): The exported document, all pages already added.
            Nothing is copied unless it has as many pages as the source.

    Returns:
        bool: True when the page counts matched and the copy was attempted; an
        unreadable source or a source without either structure copies nothing.
    """
    try:
        source = pikepdf.open(source_path)
    except (pikepdf.PdfError, OSError) as error:
        logger.warning("Cannot read bookmarks of {}: {}", source_path, error)
        return False
    with source:
        if len(source.pages) != len(target.pages):
            logger.info("Bookmarks and page labels not carried: page count differs")
            return False
        _carry_page_labels(source, target)
        _carry_outline(source, target)
    return True


def _carry_page_labels(source: pikepdf.Pdf, target: pikepdf.Pdf) -> None:
    labels = source.Root.get("/PageLabels")
    if labels is not None:
        if not labels.is_indirect:
            labels = source.make_indirect(labels)
        target.Root.PageLabels = target.copy_foreign(labels)


def _carry_outline(source: pikepdf.Pdf, target: pikepdf.Pdf) -> None:
    if "/Outlines" not in source.Root:
        return
    page_of = {page.objgen: index for index, page in enumerate(source.pages)}
    with source.open_outline() as outline:
        roots = [_translate(item, source, page_of) for item in outline.root]
    kept = [item for item in roots if item is not None]
    if not kept:
        return
    with target.open_outline() as new_outline:
        new_outline.root.extend(kept)


def _translate(
    item: OutlineItem, source: pikepdf.Pdf, page_of: dict[Any, int]
) -> OutlineItem | None:
    """Rebuild a bookmark for the output; None when its page cannot be found.

    A bookmark without a page of its own still stays when it has children that
    do, so its title keeps them grouped.
    """
    children = [
        child
        for child in (_translate(c, source, page_of) for c in item.children)
        if child is not None
    ]
    index = _page_index(item, source, page_of)
    if index is None and not children:
        return None
    translated = OutlineItem(
        str(item.title), index, _FIT if index is not None else None
    )
    translated.is_closed = item.is_closed
    translated.children = children
    return translated


def _page_index(
    item: OutlineItem, source: pikepdf.Pdf, page_of: dict[Any, int]
) -> int | None:
    destination = item.destination
    if destination is None and item.action is not None:
        action = item.action
        if str(action.get("/S")) == "/GoTo":
            destination = action.get("/D")
    return _resolve(destination, source, page_of)


def _resolve(
    destination: object, source: pikepdf.Pdf, page_of: dict[Any, int]
) -> int | None:
    if isinstance(destination, pikepdf.Array):
        if not len(destination) or not isinstance(destination[0], pikepdf.Dictionary):
            return None
        return page_of.get(destination[0].objgen)
    if isinstance(destination, Name):
        dests = source.Root.get("/Dests")
        named = None if dests is None else dests.get(str(destination))
    elif isinstance(destination, pikepdf.String):
        names = source.Root.get("/Names")
        tree = None if names is None else names.get("/Dests")
        named = None
        if tree is not None:
            lookup = pikepdf.NameTree(tree)
            named = lookup.get(str(destination))
    else:
        return None
    if isinstance(named, pikepdf.Dictionary):
        named = named.get("/D")
    return (
        _resolve(named, source, page_of) if isinstance(named, pikepdf.Array) else None
    )
