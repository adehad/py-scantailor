"""Image loading, page enumeration and writing.

Ports ImageLoader, ImageMetadataLoader and TiffWriter.

Pixel conventions shared with the rest of the port:

* Grayscale and bilevel images are 2-D ``uint8`` arrays. Bilevel files load as
  0 (black) / 255 (white).
* Colour images are ``(H, W, 3)`` ``uint8`` arrays in BGR order, as cv2 expects.
* Deeper samples (16-bit) are scaled to 8-bit. Alpha channels are discarded
  on load; ``write_image`` stores a ``(H, W, 4)`` BGRA array with its alpha.

A missing or unusable DPI is reported as the null ``Dpi`` (0, 0), exactly as the
C++ metadata loaders do; callers decide how to prompt for a fallback using
``ImageMetadata.is_dpi_ok()``.
"""

import io
import math
import os
import struct
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, BinaryIO, Final

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from scantailor.core.models import Dpi, ImageId
from scantailor.core.project import ImageMetadata
from scantailor.core.settings import TiffCompression, TiffCompressions
from scantailor.imageproc.posterizer import to_indexed

# Scans routinely exceed Pillow's default decompression-bomb threshold of about
# 89 megapixels. Pillow warns above this limit and refuses above twice it.
MAX_IMAGE_PIXELS: Final = 500_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_MAX_SAMPLE_16: Final = 65535
_TIFF_SUFFIXES: Final = frozenset({".tif", ".tiff"})
_MAX_PALETTE_COLOURS: Final = 256
_TAG_X_RESOLUTION: Final = 282
_TAG_Y_RESOLUTION: Final = 283
_TAG_RESOLUTION_UNIT: Final = 296
_TAG_ORIENTATION: Final = 274
_TIFF_MAGIC: Final = 42
_BIGTIFF_MAGIC: Final = 43
_TIFF_TYPE_SHORT: Final = 3
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


def _dpi_value(raw: Any, scale: float = 1.0) -> int:
    """Round a resolution to whole dots per inch; 0 when it is not a usable number.

    TIFF stores resolutions as rationals, and a zero denominator converts to
    nan or infinity.
    """
    try:
        value = float(raw) * scale
    except TypeError, ValueError, ZeroDivisionError:
        return 0
    if not math.isfinite(value) or value <= 0:
        return 0
    return _round_half_up(value)


def _dpi_from_pair(horizontal: int, vertical: int) -> Dpi:
    """Return the resolution, or the null one if either direction is unknown."""
    if horizontal <= 0 or vertical <= 0:
        return Dpi()
    return Dpi(horizontal=horizontal, vertical=vertical)


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
    return _dpi_from_pair(_dpi_value(x_res, scale), _dpi_value(y_res, scale))


def _dpi_from_jfif(image: Image.Image) -> Dpi:
    # Pillow also reports EXIF resolution as "dpi"; only the JFIF density counts.
    density = image.info.get("jfif_density")
    unit = image.info.get("jfif_unit", 0)
    if not density or unit not in (_JFIF_UNIT_INCH, _JFIF_UNIT_CENTIMETER):
        return Dpi()
    scale = _CM_PER_INCH if unit == _JFIF_UNIT_CENTIMETER else 1.0
    return _dpi_from_pair(_dpi_value(density[0], scale), _dpi_value(density[1], scale))


def _dpi_of(image: Image.Image) -> Dpi:
    if image.format == "TIFF":
        return _dpi_from_tiff_tags(image)
    if image.format == "JPEG":
        return _dpi_from_jfif(image)
    dpi = image.info.get("dpi")
    if not dpi:
        return Dpi()
    return _dpi_from_pair(*(_dpi_value(v) for v in dpi[:2]))


# EXIF orientation value to the transpose that makes the stored pixels upright.
_ORIENTATION_TRANSPOSE: Final = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}
_AXIS_SWAPPING_ORIENTATIONS: Final = frozenset({5, 6, 7, 8})


_TIFF_ORIENTATIONS_ATTRIBUTE: Final = "_scantailor_tiff_orientations"
_TIFF_MAX_IFDS: Final = 1_000_000


class _OrientationMaskedFile(io.RawIOBase):
    """Read-only view of a TIFF that shows every orientation tag as 1.

    Pillow applies TIFF orientation itself, but inconsistently: compressed
    files come out upright, uncompressed ones with a 90 degree orientation come
    out scrambled, and the reported size depends on the code path. Masking the
    tag makes every path return the stored pixels so that one transpose here
    covers every format. Wrap it in ``io.BufferedReader`` to hand to Pillow.
    """

    def __init__(self, file: BinaryIO, patches: dict[int, bytes]) -> None:
        super().__init__()
        self._file = file
        self._patches = patches

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        start = self._file.tell()
        count = self._file.readinto(buffer)  # pyrefly: ignore[missing-attribute]
        for offset, replacement in self._patches.items():
            for index, value in enumerate(replacement):
                position = offset + index - start
                if 0 <= position < count:
                    buffer[position] = value
        return count

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._file.seek(offset, whence)

    def tell(self) -> int:
        return self._file.tell()


def _scan_tiff_orientations(file: BinaryIO) -> tuple[list[int], dict[int, bytes]]:
    """Find the orientation of every IFD of a TIFF without decoding it.

    Returns:
        tuple[list[int], dict[int, bytes]]: The orientation of each IFD in file
            order (1 where the tag is absent or unusable), and the byte
            replacements that turn every orientation other than 1 into 1.
            Both are empty for a file that is not a well-formed TIFF.
    """
    header = file.read(16)
    endian = {b"II": "<", b"MM": ">"}.get(header[:2])
    if endian is None:
        return [], {}
    magic = struct.unpack(endian + "H", header[2:4])[0]
    if magic == _TIFF_MAGIC:
        first_ifd = struct.unpack(endian + "I", header[4:8])[0]
        count_format, entry_size, value_offset, link_format = "H", 12, 8, "I"
    elif magic == _BIGTIFF_MAGIC and len(header) == 16:
        first_ifd = struct.unpack(endian + "Q", header[8:16])[0]
        count_format, entry_size, value_offset, link_format = "Q", 20, 12, "Q"
    else:
        return [], {}
    count_size = struct.calcsize(count_format)
    link_size = struct.calcsize(link_format)
    orientations: list[int] = []
    patches: dict[int, bytes] = {}
    visited: set[int] = set()
    ifd = first_ifd
    try:
        while ifd and ifd not in visited and len(visited) < _TIFF_MAX_IFDS:
            visited.add(ifd)
            file.seek(ifd)
            (count,) = struct.unpack(endian + count_format, file.read(count_size))
            table = file.read(entry_size * count)
            (ifd,) = struct.unpack(endian + link_format, file.read(link_size))
            orientation = 1
            for index in range(count):
                entry = table[index * entry_size : (index + 1) * entry_size]
                tag, kind = struct.unpack(endian + "HH", entry[:4])
                if tag != _TAG_ORIENTATION or kind != _TIFF_TYPE_SHORT:
                    continue
                (value,) = struct.unpack(
                    endian + "H", entry[value_offset : value_offset + 2]
                )
                if 1 < value <= 8:
                    orientation = value
                    entry_start = (
                        file.tell() - link_size - len(table) + index * entry_size
                    )
                    patches[entry_start + value_offset] = struct.pack(endian + "H", 1)
            orientations.append(orientation)
    except struct.error, OSError, ValueError:
        return [], {}
    return orientations, patches


def _exif_orientation(image: Image.Image) -> int:
    """Return the EXIF orientation (1 to 8) of the current frame; 1 if unusable."""
    tiff_orientations = getattr(image, _TIFF_ORIENTATIONS_ATTRIBUTE, None)
    if tiff_orientations is not None:
        frame = getattr(image, "tell", lambda: 0)()
        return tiff_orientations[frame] if frame < len(tiff_orientations) else 1
    try:
        value = image.getexif().get(_TAG_ORIENTATION, 1)
    except OSError, SyntaxError, ValueError, TypeError:
        return 1
    return value if isinstance(value, int) and 1 <= value <= 8 else 1


def _upright(image: Image.Image) -> Image.Image:
    """Return the image with its EXIF orientation applied to the pixels."""
    transpose = _ORIENTATION_TRANSPOSE.get(_exif_orientation(image))
    return image if transpose is None else image.transpose(transpose)


def _metadata_of(image: Image.Image) -> ImageMetadata:
    width, height = image.size
    dpi = _dpi_of(image)
    if _exif_orientation(image) in _AXIS_SWAPPING_ORIENTATIONS:
        width, height = height, width
        dpi = Dpi(horizontal=dpi.vertical, vertical=dpi.horizontal)
    return ImageMetadata(width=width, height=height, dpi=dpi)


@contextmanager
def _open(path: Path) -> Iterator[Image.Image]:
    """Open an image with its pixels as stored, leaving EXIF orientation alone."""
    try:
        file = path.open("rb")
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ImageLoadError(f"Cannot read image {path}: {exc}") from exc
    with file:
        orientations, patches = _scan_tiff_orientations(file)
        file.seek(0)
        source: BinaryIO = file
        if patches:
            source = io.BufferedReader(_OrientationMaskedFile(file, patches))
        try:
            image = Image.open(source)
        except UnidentifiedImageError as exc:
            raise UnsupportedImageFormatError(
                f"Unrecognised image format: {path}"
            ) from exc
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
            raise ImageLoadError(f"Cannot read image {path}: {exc}") from exc
        if image.format == "TIFF" and orientations:
            setattr(image, _TIFF_ORIENTATIONS_ATTRIBUTE, orientations)
        with image:
            yield image


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


def _float_to_gray(samples: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Scale floating-point grey samples to 8 bits.

    Samples within 0 to 1 map 0 to black and 1 to white. Any other range is
    stretched from its lowest to its highest finite sample, since such a file
    carries no scale of its own. NaN counts as the lowest sample, infinities as
    the ends of the range.

    The scaling works in place, so the peak memory is ``samples`` plus a byte
    per pixel for the finite mask and another for the result.

    Args:
        samples (NDArray[np.float32]): 2-D grey samples, overwritten.

    Returns:
        NDArray[np.uint8]: The page, 2-D.
    """
    finite = np.isfinite(samples)
    low = float(samples.min(where=finite, initial=np.inf))
    high = float(samples.max(where=finite, initial=-np.inf))
    del finite
    if not low <= high:
        low = high = 0.0
    if 0.0 <= low and high <= 1.0:
        low, high = 0.0, 1.0
    np.nan_to_num(samples, copy=False, nan=low, posinf=high, neginf=low)
    if high <= low:
        return np.zeros(samples.shape, dtype=np.uint8)
    # Scaling before the shift keeps a range near the float32 limits finite.
    scale = 255.0 / (high - low)
    samples *= scale
    samples -= low * scale
    np.rint(samples, out=samples)
    np.clip(samples, 0, 255, out=samples)
    return samples.astype(np.uint8)


def _to_array(image: Image.Image) -> NDArray[np.uint8]:
    mode = image.mode
    if mode in ("I;16", "I;16L", "I;16B", "I"):
        # Mode I is signed 32-bit; negative samples would wrap as unsigned.
        samples = np.clip(np.asarray(image, dtype=np.int64), 0, _MAX_SAMPLE_16)
        return ((samples * 255 + 32767) // _MAX_SAMPLE_16).astype(np.uint8)
    if mode == "F":
        return _float_to_gray(np.array(image, dtype=np.float32))
    if mode in ("1", "L", "LA"):
        return np.asarray(image.convert("L"), dtype=np.uint8)
    if mode in ("P", "PA") and _has_gray_palette(image):
        return np.asarray(image.convert("L"), dtype=np.uint8)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def load_image(
    image_id: ImageId, dpi: Dpi | None = None
) -> tuple[NDArray[np.uint8], ImageMetadata]:
    """Decode one page of an image file, upright according to its EXIF orientation.

    A 90 or 270 degree orientation swaps the width and height, and the
    horizontal and vertical DPI, of the stored image.

    Args:
        image_id (ImageId): File and page; ``page`` is 1-based for multi-page
            TIFFs and 0 (or 1) for single-page files.
        dpi (Dpi | None): Resolution that replaces the one in the file; ignored
            when None or null. Given for the upright image.

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
            metadata = _metadata_of(image)
            pixels = _to_array(_upright(image))
        except (
            OSError,
            SyntaxError,
            EOFError,
            ValueError,
            Image.DecompressionBombError,
        ) as exc:
            raise ImageLoadError(f"Cannot decode {path}: {exc}") from exc
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


def _colour_image(bgr: NDArray[np.uint8], *, palette: bool) -> Image.Image:
    """Make the Pillow image a BGR page is written from.

    With ``palette``, a page of at most 256 colours, such as posterized output,
    becomes a palette image, and one whose colours are all grey becomes a grey
    image, as C++ writes an indexed QImage. Otherwise it stays 24-bit RGB.

    Args:
        bgr (NDArray[np.uint8]): (H, W, 3) BGR page. Not modified.
        palette (bool): Whether a palette image may be written; libtiff's JPEG
            codec takes no palette images.

    Returns:
        Image.Image: A mode ``P``, ``L`` or ``RGB`` image of the same pixels.
    """
    indexed = to_indexed(bgr, _MAX_PALETTE_COLOURS) if palette else None
    if indexed is None:
        return Image.fromarray(np.ascontiguousarray(bgr[..., ::-1]), mode="RGB")
    colours, indices = indexed
    if np.all(colours == colours[:, :1]):
        return Image.fromarray(np.ascontiguousarray(bgr[..., 0]), mode="L")
    image = Image.fromarray(indices, mode="P")
    image.putpalette(colours.tobytes())
    return image


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
    follow the colour method. A BGR TIFF of at most 256 colours is written as a
    palette image, or as grey when every colour is grey, unless the colour method
    is JPEG. ``(H, W, 4)`` ``uint8`` arrays are BGRA and keep
    their alpha as an extra sample, as C++ does for images with an alpha channel.
    16-bit RGB is not written, since C++ writes none either; reduce it to 8 bits.

    The file is written under a temporary name beside ``path`` and renamed into
    place, so a failure leaves any earlier file at ``path`` untouched.

    Args:
        path (Path): Destination; suffix must be .tif, .tiff or .png.
        pixels (NDArray): ``bool`` or ``uint8`` array as described above. Not
            modified.
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
        image = _colour_image(
            pixels, palette=is_tiff and chosen.color != TiffCompression.JPEG
        )
        compression = _PILLOW_COMPRESSION[chosen.color]
    elif pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 4:
        bgra_to_rgba = pixels[..., [2, 1, 0, 3]]
        image = Image.fromarray(np.ascontiguousarray(bgra_to_rgba), mode="RGBA")
        compression = _PILLOW_COMPRESSION[chosen.color]
        if compression == _PILLOW_COMPRESSION[TiffCompression.JPEG]:
            # libtiff's JPEG codec has no place for an alpha sample.
            compression = _PILLOW_COMPRESSION[TiffCompression.DEFLATE]
    elif pixels.dtype == np.uint16 and pixels.ndim == 3:
        raise ValueError(
            "16-bit colour images cannot be written; convert them to 8 bits first"
        )
    else:
        raise ValueError(f"Unsupported array dtype={pixels.dtype} shape={pixels.shape}")

    options: dict = {}
    if dpi is not None and not dpi.is_null():
        options["dpi"] = (dpi.horizontal, dpi.vertical)
    if is_tiff:
        options["compression"] = compression
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    try:
        image.save(temporary, format="TIFF" if is_tiff else "PNG", **options)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
