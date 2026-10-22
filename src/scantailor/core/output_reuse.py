"""Decide whether a page's output files can be kept instead of rendered again.

A batch records, per written page, a digest of everything that decides its
output and the size and modification time of its source and output files. A
later batch keeps a page whose digest and files still match, as upstream's
output stage does with its stored output parameters.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from scantailor.core.models import Dpi, PageId
from scantailor.core.project import FileStamp, PageOutputRecord
from scantailor.core.settings import TiffCompressions
from scantailor.core.stage_params import (
    collect_effective_stage_params,
    collect_stage_defaults,
)
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.page_layout.params import ContentSize

# Bump when a change to rendering makes earlier output files stale.
RECORD_FORMAT = 1

# Stage-wide values that only steer the interactive view, never a page's pixels.
_VIEW_ONLY_DEFAULTS = frozenset({"page_layout_middle_rect_shown", "page_layout_guides"})


def output_params_hash(
    stages: StageSequence,
    page: PageId,
    *,
    dpi_override: Dpi | None,
    default_dpi: float,
    tiff_compressions: TiffCompressions,
    aggregate_mm: ContentSize,
) -> str:
    """Digest everything that decides how a page's output files come out.

    That is the page's effective parameters at every stage, the stage-wide
    defaults, the aggregate page size when the page is aligned with the others,
    the resolution it is read at and the TIFF compression it is written with.

    Args:
        stages (StageSequence): Settings the page would be rendered with.
        page (PageId): The page.
        dpi_override (Dpi | None): Project resolution of the page's image.
        default_dpi (float): Resolution for sources that do not define one.
        tiff_compressions (TiffCompressions): Methods the files are written with.
        aggregate_mm (ContentSize): The shared size of aligned pages. The caller
            takes it once per batch, since computing it reads every page's
            layout.

    Returns:
        str: A hex digest; equal digests mean equal output.
    """
    aligned = stages.page_layout_filter.get_params(page).alignment.aligned_with_others()
    defaults = {
        key: value
        for key, value in collect_stage_defaults(stages).items()
        if key not in _VIEW_ONLY_DEFAULTS
    }
    data: dict[str, Any] = {
        "format": RECORD_FORMAT,
        "params": collect_effective_stage_params(stages, [page]),
        "defaults": defaults,
        "aggregate": aggregate_mm.model_dump(mode="json") if aligned else None,
        "dpi_override": None
        if dpi_override is None
        else dpi_override.model_dump(mode="json"),
        "default_dpi": default_dpi,
        "tiff_compressions": [tiff_compressions.bw.name, tiff_compressions.color.name],
    }
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def build_record(
    page: PageId,
    params_hash: str,
    output_dir: Path,
    written: Iterable[Path],
) -> PageOutputRecord | None:
    """Record the files just written for a page.

    Args:
        page (PageId): The page.
        params_hash (str): ``output_params_hash`` of the settings it was
            rendered with.
        output_dir (pathlib.Path): The output folder the files are in.
        written (Iterable[pathlib.Path]): Every file written for the page.

    Returns:
        PageOutputRecord | None: The record, or None when the source or a
            written file cannot be stamped, so the page is rendered next time.
    """
    source = FileStamp.of(page.image_id.file_path)
    if source is None:
        return None
    files: dict[str, FileStamp] = {}
    for path in written:
        stamp = FileStamp.of(path)
        if stamp is None:
            return None
        files[_relative_key(path, output_dir)] = stamp
    return PageOutputRecord(
        page=page, params_hash=params_hash, source=source, files=files
    )


def _relative_key(path: Path, output_dir: Path) -> str:
    return PurePosixPath(Path(path).relative_to(output_dir)).as_posix()


def is_reusable(
    record: PageOutputRecord,
    params_hash: str,
    output_dir: Path,
    output_name: str,
) -> bool:
    """Tell whether a page's existing output files can stand for a new render.

    Args:
        record (PageOutputRecord): What the files were rendered from.
        params_hash (str): ``output_params_hash`` of the current settings.
        output_dir (pathlib.Path): The project's output folder.
        output_name (str): The page's current output file name.

    Returns:
        bool: True when the settings, the source file and every recorded output
            file are as they were, and the main output file is among them.
    """
    if record.params_hash != params_hash:
        return False
    if output_name not in record.files:
        return False
    if FileStamp.of(record.page.image_id.file_path) != record.source:
        return False
    return _files_unchanged(record.files, output_dir)


def _files_unchanged(files: Mapping[str, FileStamp], output_dir: Path) -> bool:
    return all(
        FileStamp.of(output_dir / PurePosixPath(key)) == stamp
        for key, stamp in files.items()
    )
