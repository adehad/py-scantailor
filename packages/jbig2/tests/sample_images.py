"""Deterministic bilevel test images; ``True`` is black."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:"


def checkerboard(height: int, width: int, cell: int) -> np.ndarray:
    """Checkerboard with square cells.

    Args:
        height (int): Rows.
        width (int): Columns.
        cell (int): Cell side in pixels.

    Returns:
        np.ndarray: ``bool`` image.
    """
    y, x = np.indices((height, width))
    return ((y // cell + x // cell) % 2).astype(bool)


def noise(height: int, width: int, density: float, seed: int) -> np.ndarray:
    """Independent random pixels.

    Args:
        height (int): Rows.
        width (int): Columns.
        density (float): Probability of a black pixel.
        seed (int): RNG seed.

    Returns:
        np.ndarray: ``bool`` image.
    """
    return np.random.default_rng(seed).random((height, width)) < density


def _render_text(width: int, height: int, seed: int, font_size: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    page = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=font_size)
    margin = width // 14
    line_pitch = int(font_size * 1.5)
    for top in range(margin, height - margin - line_pitch, line_pitch):
        if rng.random() < 0.08:
            continue  # paragraph break
        words = []
        while sum(len(w) + 1 for w in words) < (width - 2 * margin) / (font_size * 0.55):
            words.append("".join(rng.choice(list(_LETTERS), size=rng.integers(1, 10))))
        line = " ".join(words)
        while draw.textlength(line, font=font) > width - 2 * margin:
            line = line[: line.rfind(" ")]
        draw.text((margin, top), line, fill=0, font=font)
    return page


def text_page(width: int = 2100, height: int = 3000, seed: int = 0) -> np.ndarray:
    """Clean synthetic page of rendered text, A4 at roughly 250 dpi by default.

    Args:
        width (int): Columns.
        height (int): Rows.
        seed (int): RNG seed for the word sequence.

    Returns:
        np.ndarray: ``bool`` image.
    """
    return np.asarray(_render_text(width, height, seed, font_size=34)) < 128


def scanned_page(width: int = 1700, height: int = 2200, seed: int = 1) -> np.ndarray:
    """Page with the defects of a thresholded scan.

    Slightly rotated text with ragged edges from sensor noise, salt-and-pepper
    specks, a dark scanner-lid band down one side and an ordered-dither
    photograph region.

    Args:
        width (int): Columns.
        height (int): Rows.
        seed (int): RNG seed.

    Returns:
        np.ndarray: ``bool`` image.
    """
    rng = np.random.default_rng(seed)
    gray = _render_text(width, height, seed, font_size=28).rotate(
        0.6, resample=Image.Resampling.BILINEAR, fillcolor=255
    )
    pixels = np.asarray(gray, dtype=np.float32) + rng.normal(0, 25, (height, width))
    black = pixels < 128
    black |= rng.random((height, width)) < 0.0005
    black[:, : width // 60] = rng.random((height, width // 60)) < 0.9
    top, left = height // 3, width // 2
    photo_h, photo_w = height // 5, width // 3
    y, x = np.indices((photo_h, photo_w))
    shade = 0.5 + 0.5 * np.sin(x / 37.0) * np.cos(y / 23.0)
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0
    black[top : top + photo_h, left : left + photo_w] = (
        shade > np.tile(bayer, (photo_h // 4 + 1, photo_w // 4 + 1))[:photo_h, :photo_w]
    )
    return black


def fuzz_images(count: int, seed: int) -> list[np.ndarray]:
    """Small random images mixing noise, solid blocks and repeated rows.

    Args:
        count (int): Number of images.
        seed (int): RNG seed.

    Returns:
        list[np.ndarray]: ``bool`` images of varied shapes from 1x1 to 48x80.
    """
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(count):
        height, width = int(rng.integers(1, 49)), int(rng.integers(1, 81))
        image = rng.random((height, width)) < rng.random()
        if rng.random() < 0.3:
            image[rng.random(height) < 0.6] = False
        if rng.random() < 0.3 and height > 1:
            repeat_from = rng.integers(0, height)
            image[repeat_from:] = image[repeat_from]
        if rng.random() < 0.2:
            image = ~image
        images.append(image)
    return images
