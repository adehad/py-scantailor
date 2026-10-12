"""Image loading, page enumeration and writing.

Ports ImageLoader, ImageMetadataLoader and TiffWriter.

Pixel conventions shared with the rest of the port:

* Grayscale and bilevel images are 2-D ``uint8`` arrays. Bilevel files load as
  0 (black) / 255 (white).
* Colour images are ``(H, W, 3)`` ``uint8`` arrays in BGR order, as cv2 expects.
* Deeper samples (16-bit) are scaled to 8-bit. Alpha channels are discarded.

A missing or unusable DPI is reported as the null ``Dpi`` (0, 0), exactly as the
C++ metadata loaders do; callers decide how to prompt for a fallback using
``ImageMetadata.is_dpi_ok()``.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Final

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from scantailor.core.models import Dpi, ImageId
from scantailor.core.project import ImageMetadata
from scantailor.core.settings import TiffCompression, TiffCompressions

# Scans routinely exceed Pillow's default decompression-bomb threshold.
Image.MAX_IMAGE_PIXELS = None

_TIFF_SUFFIXES: Final = frozenset({".tif", ".tiff"})
_TAG_X_RESOLUTION: Final = 282
_TAG_Y_RESOLUTION: Final = 283
_TAG_RESOLUTION_UNIT: Final = 296
_RESUNIT_INCH: Final = 2
_RESUNIT_CENTIMETER: Final = 3
_CM_PER_INCH: Final = 2.54
_JFIF_UNIT_INCH: Final = 1
_JFIF_UNIT_CENTIMETER: Final = 2
# Pillow's names for the libtiff methods; Adobe deflate is libtiff's
# COMPRESSION_DEFLATE (8), the older zip variant has another tag value.
_PILLOW_COMPRESSION: Final = {
    TiffCompression.NONE: "raw",
    TiffCompression.LZW: "tiff_lzw",
    TiffCompression.DEFLATE: "tiff_adobe_deflate",
    TiffCompression.JPEG: "jpeg",
    TiffCompression.CCITT_FAX4: "group4",
}
_default_compressions: ContextVar[TiffCompressions] = ContextVar(
    "tiff_compressions", default=TiffCompressions()
)


class ImageLoadError(Exception):
    """The file exists but cannot be decoded (corrupt, truncated or bad page)."""


class UnsupportedImageFormatError(ImageLoadError):
    """The file is not in a format the loader recognises."""


def _round_half_up(value: float) -> int:
    return int(value + 0.5)


def _dpi_from_tiff_tags(image: Image.Image) -> Dpi:
    tags = getattr(image, "tag_v2", None)
    if tags is None:
        return Dpi()
    x_res = tags.get(_TAG_X_RESOLUTION)
    y_res = tags.get(_TAG_Y_RESOLUTION)
    if x_res is None or y_res is None:
        return Dpi()
    unit = tags.get(_TAG_RESOLUTION_UNIT, _RESUNIT_INCH)
    if unit == _RESUNIT_INCH:
        scale = 1.0
    elif unit == _RESUNIT_CENTIMETER:
        scale = _CM_PER_INCH
    else:
        return Dpi()
    return Dpi(
        horizontal=_round_half_up(float(x_res) * scale),
        vertical=_round_half_up(float(y_res) * scale),
    )


def _dpi_from_jfif(image: Image.Image) -> Dpi:
    # Pillow also reports EXIF resolution as "dpi"; only the JFIF density counts.
    density = image.info.get("jfif_density")
    unit = image.info.get("jfif_unit", 0)
    if not density or unit not in (_JFIF_UNIT_INCH, _JFIF_UNIT_CENTIMETER):
        return Dpi()
    scale = _CM_PER_INCH if unit == _JFIF_UNIT_CENTIMETER else 1.0
    return Dpi(
        horizontal=_round_half_up(density[0] * scale),
        vertical=_round_half_up(density[1] * scale),
    )


def _dpi_of(image: Image.Image) -> Dpi:
    if image.format == "TIFF":
        return _dpi_from_tiff_tags(image)
    if image.format == "JPEG":
        return _dpi_from_jfif(image)
    dpi = image.info.get("dpi")
    if not dpi:
        return Dpi()
    horizontal, vertical = (_round_half_up(float(v)) for v in dpi[:2])
    return Dpi(horizontal=max(horizontal, 0), vertical=max(vertical, 0))


def _metadata_of(image: Image.Image) -> ImageMetadata:
    width, height = image.size
    return ImageMetadata(width=width, height=height, dpi=_dpi_of(image))


def _open(path: Path) -> Image.Image:
    try:
        return Image.open(path)
    except UnidentifiedImageError as exc:
        raise UnsupportedImageFormatError(f"Unrecognised image format: {path}") from exc
    except FileNotFoundError:
        raise
    except (OSError, SyntaxError, ValueError) as exc:
        raise ImageLoadError(f"Cannot read image {path}: {exc}") from exc


def _select_page(image: Image.Image, path: Path, zero_based_page: int) -> None:
    frames = getattr(image, "n_frames", 1)
    if image.format != "TIFF" and zero_based_page != 0:
        raise ImageLoadError(f"{path}: only TIFF files support multiple pages")
    if not 0 <= zero_based_page < frames:
        raise ImageLoadError(
            f"{path}: page {zero_based_page + 1} out of range (1..{frames})"
        )
    if zero_based_page:
        image.seek(zero_based_page)


def list_image_pages(path: Path) -> list[tuple[ImageId, ImageMetadata]]:
    """List the pages of an image file with metadata, without decoding pixels.

    Multi-page TIFFs yield 1-based page numbers; every other file yields a
    single entry with page 0 (the ``ImageId`` convention for single-page files).

    Args:
        path (Path): Image file to inspect.

    Returns:
        list[tuple[ImageId, ImageMetadata]]: One entry per page, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        UnsupportedImageFormatError: If the format is not recognised.
        ImageLoadError: If the headers are damaged.
    """
    path = Path(path)
    with _open(path) as image:
        frames = getattr(image, "n_frames", 1)
        if frames <= 1:
            return [(ImageId(file_path=path, page=0), _metadata_of(image))]
        pages: list[tuple[ImageId, ImageMetadata]] = []
        for index in range(frames):
            try:
                image.seek(index)
                metadata = _metadata_of(image)
            except (OSError, SyntaxError, EOFError, ValueError) as exc:
                raise ImageLoadError(
                    f"Cannot read page {index + 1} of {path}: {exc}"
                ) from exc
            pages.append((ImageId(file_path=path, page=index + 1), metadata))
        return pages


def _has_gray_palette(image: Image.Image) -> bool:
    palette = image.getpalette()
    if palette is None:
        return False
    entries = np.asarray(palette, dtype=np.uint8)
    entries = entries[: len(entries) // 3 * 3].reshape(-1, 3)
    return bool(
        np.all(entries[:, 0] == entries[:, 1])
        and np.all(entries[:, 1] == entries[:, 2])
    )


def _to_array(image: Image.Image) -> NDArray[np.uint8]:
    mode = image.mode
    if mode in ("I;16", "I;16L", "I;16B", "I"):
        samples = np.asarray(image, dtype=np.uint32)
        return ((samples * 255 + 32767) // 65535).astype(np.uint8)
    if mode in ("1", "L", "LA"):
        return np.asarray(image.convert("L"), dtype=np.uint8)
    if mode in ("P", "PA") and _has_gray_palette(image):
        return np.asarray(image.convert("L"), dtype=np.uint8)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def load_image(
    image_id: ImageId, dpi: Dpi | None = None
) -> tuple[NDArray[np.uint8], ImageMetadata]:
    """Decode one page of an image file.

    Args:
        image_id (ImageId): File and page; ``page`` is 1-based for multi-page
            TIFFs and 0 (or 1) for single-page files.
        dpi (Dpi | None): Resolution that replaces the one in the file; ignored
            when None or null.

    Returns:
        tuple[NDArray[np.uint8], ImageMetadata]: Pixels as ``(H, W)`` gray
            (including bilevel as 0/255) or ``(H, W, 3)`` BGR ``uint8``, and the
            size and DPI. DPI is null when the file carries none.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnsupportedImageFormatError: If the format is not recognised.
        ImageLoadError: If the file is corrupt or the page does not exist.
    """
    path = Path(image_id.file_path)
    with _open(path) as image:
        _select_page(image, path, image_id.zero_based_page())
        try:
            image.load()
            pixels = _to_array(image)
        except (OSError, SyntaxError, EOFError, ValueError) as exc:
            raise ImageLoadError(f"Cannot decode {path}: {exc}") from exc
        metadata = _metadata_of(image)
    if dpi is not None and not dpi.is_null():
        metadata = metadata.model_copy(update={"dpi": dpi})
    logger.debug(
        "Loaded {} page {}: {}x{}", path, image_id.page, metadata.width, metadata.height
    )
    return pixels, metadata


@contextmanager
def tiff_compressions_in_use(compressions: TiffCompressions) -> Iterator[None]:
    """Make ``write_image`` calls inside the block default to these methods.

    For callers that write through helpers which do not take a compression
    argument. The choice is local to the current thread or task.

    Args:
        compressions (TiffCompressions): The methods to use.

    Yields:
        None: Control, with the choice in effect.
    """
    token = _default_compressions.set(compressions)
    try:
        yield
    finally:
        _default_compressions.reset(token)


def write_image(
    path: Path,
    pixels: NDArray,
    dpi: Dpi | None = None,
    compressions: TiffCompressions | None = None,
) -> None:
    """Write an image as TIFF or PNG, chosen by file suffix.

    ``bool`` arrays are bilevel with ``True`` meaning black (the C++ BinaryImage
    convention); TIFF output uses the black-and-white method, CCITT G4 unless
    chosen otherwise. ``uint8`` arrays are gray ``(H, W)`` or BGR ``(H, W, 3)``
    and use the colour method, LZW unless chosen otherwise. Like C++, grey images
    follow the colour method.

    Args:
        path (Path): Destination; suffix must be .tif, .tiff or .png.
        pixels (NDArray): ``bool`` or ``uint8`` array as described above. Not modified.
        dpi (Dpi | None): Resolution tags to store; omitted when None or null.
        compressions (TiffCompressions | None): TIFF methods; the ones set by
            ``tiff_compressions_in_use``, else the defaults, when None. PNG ignores
            them.

    Raises:
        ValueError: On unsupported suffix, dtype or shape.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    is_tiff = suffix in _TIFF_SUFFIXES
    if not is_tiff and suffix != ".png":
        raise ValueError(f"Unsupported output format: {suffix!r}")

    if pixels.size == 0:
        raise ValueError("Cannot write an empty image")

    chosen = compressions or _default_compressions.get()
    if pixels.dtype == np.bool_ and pixels.ndim == 2:
        image = Image.fromarray(~pixels)
        compression = _PILLOW_COMPRESSION[chosen.bw]
    elif pixels.dtype == np.uint8 and pixels.ndim == 2:
        image = Image.fromarray(pixels, mode="L")
        compression = _PILLOW_COMPRESSION[chosen.color]
    elif pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3:
        image = Image.fromarray(np.ascontiguousarray(pixels[..., ::-1]), mode="RGB")
        compression = _PILLOW_COMPRESSION[chosen.color]
    else:
        raise ValueError(f"Unsupported array dtype={pixels.dtype} shape={pixels.shape}")

    options: dict = {}
    if dpi is not None and not dpi.is_null():
        options["dpi"] = (dpi.horizontal, dpi.vertical)
    if is_tiff:
        options["compression"] = compression
        image.save(path, format="TIFF", **options)
    else:
        image.save(path, format="PNG", **options)
