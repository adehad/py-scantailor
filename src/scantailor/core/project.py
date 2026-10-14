"""Project management for ScanTailor.

Handles loading, saving, and managing ScanTailor projects. Supports:
- JSON format (primary, for new projects)
- XML format (read-only, for C++ ScanTailor compatibility)
"""

import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pdf_source import PdfSource

LayoutDirection = Literal["LTR", "RTL"]

_MM_PER_INCH = 25.4
_MIN_RELIABLE_DPI = 150
_MAX_DPI = 9999
_MAX_PHYSICAL_SIZE_MM = 500


class DpiStatus(IntEnum):
    """Status of DPI validation for an image."""

    OK = 0
    UNDEFINED = 1
    TOO_LARGE = 2
    TOO_SMALL = 3
    TOO_SMALL_FOR_PIXEL_SIZE = 4


class ImageMetadata(BaseModel):
    """Metadata for an image file.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        dpi (Dpi): Resolution of the image.
    """

    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    dpi: Dpi = Field(default_factory=Dpi)

    def is_dpi_ok(self) -> bool:
        """Check that the DPI is defined on both axes.

        Implausible but defined values still count as OK; only the finer
        ``*_dpi_status`` methods report them.
        """
        return (
            self.horizontal_dpi_status() != DpiStatus.UNDEFINED
            and self.vertical_dpi_status() != DpiStatus.UNDEFINED
        )

    def horizontal_dpi_status(self) -> DpiStatus:
        """Check the status of horizontal DPI."""
        return self._dpi_status(self.width, self.dpi.horizontal)

    def vertical_dpi_status(self) -> DpiStatus:
        """Check the status of vertical DPI."""
        return self._dpi_status(self.height, self.dpi.vertical)

    @staticmethod
    def _dpi_status(pixel_size: int, dpi: int) -> DpiStatus:
        """Determine DPI status based on pixel size and DPI value.

        Args:
            pixel_size (int): Size in pixels along the axis the DPI applies to.
            dpi (int): DPI value.

        Returns:
            DpiStatus: The status of the DPI value.
        """
        if dpi <= 1:
            return DpiStatus.UNDEFINED
        if dpi < _MIN_RELIABLE_DPI:
            return DpiStatus.TOO_SMALL
        if dpi > _MAX_DPI:
            return DpiStatus.TOO_LARGE
        # A physical size this large usually means the DPI is wrong, and a DPI
        # that is too low inflates memory use quadratically.
        if _MM_PER_INCH * pixel_size / dpi > _MAX_PHYSICAL_SIZE_MM:
            return DpiStatus.TOO_SMALL_FOR_PIXEL_SIZE
        return DpiStatus.OK


class ImageInfo(BaseModel):
    """Information about an image in the project.

    Args:
        id (ImageId): Identifier for the image.
        metadata (ImageMetadata): Image dimensions and DPI.
        num_sub_pages (int): Number of logical pages (1 or 2).
        left_half_removed (bool): Whether left half was removed.
        right_half_removed (bool): Whether right half was removed.
    """

    id: ImageId
    metadata: ImageMetadata = Field(default_factory=ImageMetadata)
    num_sub_pages: int = Field(default=1, ge=1, le=2)
    left_half_removed: bool = False
    right_half_removed: bool = False


class PageInfo(BaseModel):
    """Information about a logical page.

    Args:
        page_id (PageId): Identifier for the page.
        image_info (ImageInfo): Information about the source image.
    """

    page_id: PageId
    image_info: ImageInfo


class StageParamsEntry(BaseModel):
    """Serialised parameters of one filter stage for one page.

    Args:
        page (PageId): The page the parameters belong to. Stages that store
            parameters per image use the image's single-page id.
        params (dict[str, Any]): The stage's Params model dumped as JSON data.
    """

    page: PageId
    params: dict[str, Any]


class PageTextGeometry(BaseModel):
    """How one output page relates to its source page, for carrying text over.

    Args:
        page (PageId): The page.
        geometry (dict[str, Any]): A ``text_layer.TextGeometry`` dumped as JSON
            data; kept as data so that loading a project needs no PDF library.
    """

    page: PageId
    geometry: dict[str, Any]


class OutputFileLabel(BaseModel):
    """The number that tells apart source files sharing a file name.

    It appears in the output file name as ``name(label).tif``. The labels
    outlive the images: removing a page keeps its file's label, so the other
    outputs keep their names.

    Args:
        file_path (pathlib.Path): The source file.
        label (int): Zero for the first file of a name, then counting up.
    """

    file_path: Path
    label: int = Field(ge=0)


def assign_output_file_labels(
    files: Iterable[Path], known: Mapping[Path, int] | None = None
) -> dict[Path, int]:
    """Give every source file an output label, keeping the known ones.

    A file without a label gets one more than the highest label of any file
    with the same name, or zero when it is the first of its name.

    Args:
        files (Iterable[pathlib.Path]): Source files in project order; the order decides
            which colliding file gets which new label.
        known (Mapping[pathlib.Path, int] | None): Labels to keep, for instance those
            written by an earlier session.

    Returns:
        dict[pathlib.Path, int]:
            The known labels plus a label for each file in ``files``.
    """
    labels = dict(known or {})
    highest: dict[str, int] = {}
    for file_path, label in labels.items():
        highest[file_path.name] = max(highest.get(file_path.name, -1), label)
    for file_path in files:
        if file_path not in labels:
            labels[file_path] = highest.get(file_path.name, -1) + 1
            highest[file_path.name] = labels[file_path]
    return labels


class Project(BaseModel):
    """A ScanTailor project.

    Args:
        version (int): Project file format version.
        output_directory (pathlib.Path): Directory for output files.
        layout_direction (str): Text layout direction (LTR or RTL).
        images (list[ImageInfo]): List of images in the project.
        selected_page (PageId | None): Currently selected page.
        stage_params (dict[str, list[StageParamsEntry]]): Explicitly set per-page
            parameters, keyed by stage key (see ``core.stage_params``).
        stage_defaults (dict[str, Any]): Stage-wide values that unset pages fall
            back to (see ``core.stage_params.collect_stage_defaults``).
        output_file_labels (list[OutputFileLabel]): Labels that keep output file
            names stable; see ``batch.output_file_names``.
        pdf_sources (list[PdfSource]): The PDFs the pages were rendered from.
        text_geometry (list[PageTextGeometry]): Geometry of pages written by a
            batch, so that their source text can be carried over in a later
            session.
    """

    version: int = 1
    output_directory: Path = Field(default_factory=lambda: Path("out"))
    layout_direction: LayoutDirection = "LTR"
    images: list[ImageInfo] = Field(default_factory=list)
    selected_page: PageId | None = None
    stage_params: dict[str, list[StageParamsEntry]] = Field(default_factory=dict)
    stage_defaults: dict[str, Any] = Field(default_factory=dict)
    output_file_labels: list[OutputFileLabel] = Field(default_factory=list)
    pdf_sources: list[PdfSource] = Field(default_factory=list)
    text_geometry: list[PageTextGeometry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _lift_single_pdf_source(cls, data: Any) -> Any:
        """Read the single ``pdf_source`` that earlier versions wrote as a list.

        Args:
            data (Any): Raw project data before validation.

        Returns:
            Any: The data with ``pdf_sources`` set from the old field.
        """
        if not isinstance(data, dict) or "pdf_source" not in data:
            return data
        data = dict(data)
        legacy = data.pop("pdf_source")
        if legacy is not None and not data.get("pdf_sources"):
            data["pdf_sources"] = [legacy]
        return data

    def source_of(self, image_file: Path) -> PdfSource | None:
        """Return the source PDF that rendered an image file, if any.

        Args:
            image_file (pathlib.Path): Image file of the project, with its folder.

        Returns:
            PdfSource | None: The first source that rendered the file.
        """
        for source in self.pdf_sources:
            if source.page_index(image_file) is not None:
                return source
        return None

    def get_pages(self) -> list[PageId]:
        """Get all page IDs in the project, respecting layout direction.

        Returns:
            list[PageId]: All pages in display order.
        """
        return [page for info in self.images for page in self._pages_of_image(info)]

    def _pages_of_image(self, image_info: ImageInfo) -> list[PageId]:
        """List the pages one image contributes, in display order."""
        if image_info.num_sub_pages == 1:
            if image_info.left_half_removed:
                sub_page = SubPage.RIGHT_PAGE
            elif image_info.right_half_removed:
                sub_page = SubPage.LEFT_PAGE
            else:
                sub_page = SubPage.SINGLE_PAGE
            return [PageId(image_id=image_info.id, sub_page=sub_page)]
        halves = [SubPage.LEFT_PAGE, SubPage.RIGHT_PAGE]
        if self.layout_direction == "RTL":
            halves.reverse()
        return [PageId(image_id=image_info.id, sub_page=half) for half in halves]

    def add_image(
        self, image_id: ImageId, metadata: ImageMetadata | None = None
    ) -> ImageInfo:
        """Add an image to the project.

        Args:
            image_id (ImageId): The image identifier.
            metadata (ImageMetadata | None): Optional metadata for the image.

        Returns:
            ImageInfo: The created image info.
        """
        info = ImageInfo(
            id=image_id,
            metadata=metadata or ImageMetadata(),
        )
        self.images.append(info)
        return info

    def insert_image(
        self,
        image_info: ImageInfo,
        existing: ImageId | None,
        *,
        before: bool,
    ) -> list[PageId]:
        """Insert an image next to an image already in the project.

        Args:
            image_info (ImageInfo): The image to insert. Both halves of a
                two-page image are kept; for a one-page image, removing both
                halves is read as removing none.
            existing (ImageId | None): The image to insert beside. None with
                ``before=True`` appends at the end.
            before (bool): Insert before ``existing`` rather than after it.

        Returns:
            list[PageId]: The pages the image adds, in display order; empty if
                ``existing`` is not in the project (and is not the append case).
        """
        if existing is None:
            if not before:
                return []
            position = len(self.images)
        else:
            position = next(
                (n for n, info in enumerate(self.images) if info.id == existing),
                None,
            )
            if position is None:
                return []
            if not before:
                position += 1

        both_removed = image_info.left_half_removed and image_info.right_half_removed
        if image_info.num_sub_pages == 2 or both_removed:
            image_info = image_info.model_copy(
                update={"left_half_removed": False, "right_half_removed": False}
            )
        self.images.insert(position, image_info)
        return self._pages_of_image(image_info)

    def insert_images(
        self,
        image_infos: Sequence[ImageInfo],
        existing: ImageId | None,
        *,
        before: bool,
    ) -> list[PageId]:
        """Insert several images as one block, keeping their given order.

        Args:
            image_infos (Sequence[ImageInfo]): Images in the order they should
                appear, such as the pages of a multi-page TIFF.
            existing (ImageId | None): The image to insert beside; see
                ``insert_image``.
            before (bool): Insert the block before ``existing`` rather than after.

        Returns:
            list[PageId]: The pages added, in display order.
        """
        added: list[PageId] = []
        anchor = existing
        for image_info in image_infos:
            pages = self.insert_image(image_info, anchor, before=before)
            if not pages:
                break
            added.extend(pages)
            if not before:
                anchor = image_info.id
        return added

    def remove_pages(self, pages: Iterable[PageId]) -> list[PageId]:
        """Remove pages, whole images or single halves of split images.

        Removing a single-page entry removes its image. Removing one half of a
        two-page image keeps the other half as a one-page image; removing both
        halves removes the image. Parameters of pages that remain are kept and
        those of removed pages are dropped.

        Args:
            pages (Iterable[PageId]): Pages to remove; pages not in the project
                are ignored.

        Returns:
            list[PageId]: The pages that were removed, in display order.
        """
        wanted = set(pages)
        present = self.get_pages()
        doomed = [page for page in present if page in wanted]
        if not doomed:
            return []
        doomed_set = set(doomed)

        remaining: list[ImageInfo] = []
        for info in self.images:
            image_pages = self._pages_of_image(info)
            kept = [page for page in image_pages if page not in doomed_set]
            if len(kept) == len(image_pages):
                remaining.append(info)
            elif kept:
                removed_half = next(p for p in image_pages if p in doomed_set)
                remaining.append(
                    info.model_copy(
                        update={
                            "num_sub_pages": 1,
                            "left_half_removed": removed_half.sub_page
                            == SubPage.LEFT_PAGE,
                            "right_half_removed": removed_half.sub_page
                            == SubPage.RIGHT_PAGE,
                        }
                    )
                )
        self.images = remaining

        surviving_images = {info.id for info in remaining}
        survivors = set(self.get_pages())
        if self.selected_page is not None and self.selected_page not in survivors:
            self.selected_page = None
        self.stage_params = {
            stage: kept_entries
            for stage, entries in self.stage_params.items()
            if (
                kept_entries := [
                    entry
                    for entry in entries
                    if entry.page in survivors
                    or (
                        entry.page.sub_page == SubPage.SINGLE_PAGE
                        and entry.page.image_id in surviving_images
                    )
                ]
            )
        }
        self.text_geometry = [
            entry for entry in self.text_geometry if entry.page in survivors
        ]
        return doomed

    def restore_removed_half(self, image_id: ImageId) -> list[PageId]:
        """Bring back the half of a split image that was removed.

        Args:
            image_id (ImageId): The image whose removed half returns.

        Returns:
            list[PageId]: The pages of the image afterwards, in display order;
                empty if the image is unknown or has no removed half.
        """
        for position, info in enumerate(self.images):
            if info.id != image_id:
                continue
            if not (info.left_half_removed or info.right_half_removed):
                return []
            restored = info.model_copy(
                update={
                    "num_sub_pages": 2,
                    "left_half_removed": False,
                    "right_half_removed": False,
                }
            )
            self.images[position] = restored
            return self._pages_of_image(restored)
        return []

    def set_text_geometry(self, page: PageId, geometry: dict[str, Any]) -> None:
        """Remember how a page's output relates to its source page.

        Args:
            page (PageId): The page.
            geometry (dict[str, Any]): A ``TextGeometry`` dumped as JSON data.
        """
        kept = [entry for entry in self.text_geometry if entry.page != page]
        self.text_geometry = [*kept, PageTextGeometry(page=page, geometry=geometry)]

    def text_geometry_of(self, page: PageId) -> dict[str, Any] | None:
        """Return the geometry stored for a page, or None."""
        return next(
            (entry.geometry for entry in self.text_geometry if entry.page == page),
            None,
        )

    def image_ids(self) -> list[ImageId]:
        """Return the id of every image in the project."""
        return [info.id for info in self.images]

    def output_file_label_map(self) -> dict[Path, int]:
        """Return the stored output labels keyed by source file."""
        return {entry.file_path: entry.label for entry in self.output_file_labels}

    def register_output_file_labels(self) -> None:
        """Store a label for every source file of the project that lacks one."""
        labels = assign_output_file_labels(
            (info.id.file_path for info in self.images), self.output_file_label_map()
        )
        self.output_file_labels = [
            OutputFileLabel(file_path=file_path, label=label)
            for file_path, label in labels.items()
        ]

    def remap_image_paths(self, remap: Callable[[Path], Path]) -> None:
        """Replace the file path of every image, including in stored parameters.

        Args:
            remap (Callable[[pathlib.Path], pathlib.Path]):
                Maps an image file path to its new path.
        """

        def remap_id(image_id: ImageId) -> ImageId:
            return ImageId(file_path=remap(image_id.file_path), page=image_id.page)

        def remap_page(page: PageId) -> PageId:
            return PageId(image_id=remap_id(page.image_id), sub_page=page.sub_page)

        self.images = [
            info.model_copy(update={"id": remap_id(info.id)}) for info in self.images
        ]
        if self.selected_page is not None:
            self.selected_page = remap_page(self.selected_page)
        self.output_file_labels = [
            OutputFileLabel(file_path=remap(entry.file_path), label=entry.label)
            for entry in self.output_file_labels
        ]
        self.text_geometry = [
            PageTextGeometry(page=remap_page(entry.page), geometry=entry.geometry)
            for entry in self.text_geometry
        ]
        self.stage_params = {
            stage: [
                StageParamsEntry(page=remap_page(entry.page), params=entry.params)
                for entry in entries
            ]
            for stage, entries in self.stage_params.items()
        }

    def save(self, path: Path) -> None:
        """Save the project to a JSON file.

        Image and output paths are written relative to the project file when
        they can be, so a project moved together with its images still opens.

        Args:
            path (pathlib.Path): Path to save the project to.
        """
        self.register_output_file_labels()
        data = self.model_dump(mode="json")
        base = path.resolve().parent
        _map_paths(data, lambda stored: _to_stored_path(Path(stored), base))
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Project:
        """Load a project from a file.

        Automatically detects JSON or XML format.

        Args:
            path (pathlib.Path): Path to the project file.

        Returns:
            Project: The loaded project.

        Raises:
            ValueError: If the file format is not recognized.
        """
        from scantailor.core.xml_reader import decode_project_bytes

        content = decode_project_bytes(path.read_bytes())

        # Detect format by content
        stripped = content.lstrip()
        if stripped.startswith("{"):
            data = json.loads(content)
            base = path.resolve().parent
            _map_paths(data, lambda stored: _from_stored_path(stored, base))
            return cls.model_validate(data)
        if stripped.startswith("<?xml") or stripped.startswith("<project"):
            return cls._load_xml(path, content)

        msg = f"Unknown project file format: {path}"
        raise ValueError(msg)

    @classmethod
    def _load_xml(cls, path: Path, content: str) -> Project:
        """Load a project from C++ ScanTailor XML format.

        Args:
            path (pathlib.Path):
                Path to the project file (for resolving relative paths).
            content (str): XML content.

        Returns:
            Project: The loaded project.
        """
        # Import here to avoid hard dependency on lxml at module load
        from scantailor.core.xml_compat import parse_xml_project

        return parse_xml_project(path, content)


def _is_drive_less_rooted(path: Path) -> bool:
    """Whether ``path`` starts at a root but names no drive (``/x`` on Windows).

    Such a path is neither absolute nor relative on Windows, so it is kept as is.
    """
    return bool(path.root) and not path.drive and not path.is_absolute()


def _to_stored_path(path: Path, base: Path) -> str:
    """Express ``path`` relative to ``base`` with forward slashes when possible.

    Args:
        path (pathlib.Path): Path to store; relative paths are taken relative to the
            working directory.
        base (pathlib.Path): Absolute directory of the project file.

    Returns:
        str: The relative path, or the absolute path when none exists (e.g. on
            another drive).
    """
    if _is_drive_less_rooted(path):
        return path.as_posix()
    absolute = Path(os.path.abspath(path))
    try:
        return Path(os.path.relpath(absolute, base)).as_posix()
    except ValueError:
        return absolute.as_posix()


def _from_stored_path(stored: str, base: Path) -> str:
    """Resolve a stored path against the project file's directory.

    Args:
        stored (str): Absolute path, or one relative to ``base``.
        base (pathlib.Path): Absolute directory of the project file.

    Returns:
        str: The absolute, normalised path.
    """
    path = Path(stored)
    if path.is_absolute() or _is_drive_less_rooted(path):
        return str(path)
    return str(Path(os.path.normpath(base / path)))


def _dicts(value: object) -> list[dict[str, Any]]:
    """Return the mappings in a JSON list, or nothing for any other value."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _path_holders(data: dict[str, Any]) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield every ``(mapping, key)`` of a dumped project that holds a path.

    Malformed parts are skipped so validation reports them instead.
    """
    yield data, "output_directory"
    image_ids = [info.get("id") for info in _dicts(data.get("images"))]
    pages = [data.get("selected_page")]
    pages.extend(entry.get("page") for entry in _dicts(data.get("text_geometry")))
    for source in [*_dicts(data.get("pdf_sources")), *_dicts([data.get("pdf_source")])]:
        yield source, "pdf_path"
        yield source, "pages_dir"
    stage_params = data.get("stage_params")
    if isinstance(stage_params, dict):
        for entries in stage_params.values():
            pages.extend(entry.get("page") for entry in _dicts(entries))
    image_ids.extend(page.get("image_id") for page in pages if isinstance(page, dict))
    for image_id in image_ids:
        if isinstance(image_id, dict):
            yield image_id, "file_path"
    yield from (
        (entry, "file_path") for entry in _dicts(data.get("output_file_labels"))
    )


def _map_paths(data: dict[str, Any], convert: Callable[[str], str]) -> None:
    """Convert, in place, every path string of a dumped project.

    Args:
        data (dict[str, Any]): Project data as produced by ``model_dump(mode="json")``.
        convert (Callable[[str], str]): Maps one stored path string to another.
    """
    if not isinstance(data, dict):
        return
    for holder, key in _path_holders(data):
        value = holder.get(key)
        if isinstance(value, str):
            holder[key] = convert(value)
