"""Debug images the Output stage records, with upstream's labels, on generated pages."""

import cv2
import numpy as np
import pytest

from scantailor.core.black_on_white_estimator import is_black_on_white_refining
from scantailor.core.debug_images import DebugImages
from scantailor.core.estimate_background import normalize_illumination_gray
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.output.binarization import (
    BinarizationOptions,
    ColorSegmenterOptions,
)
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.color_options import (
    ColorCommonOptions,
    PosterizationOptions,
)
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.generator import generate_output
from scantailor.filters.output.params import Params
from scantailor.filters.output.picture_shape import PictureShape, PictureShapeOptions
from scantailor.imageproc.despeckle import despeckle

_H, _W = 400, 300
_ESTIMATION = [
    "downscaled",
    "preproc_method1",
    "preproc_method2",
    "raw_diff",
    "approx_diff",
    "raw_vs_approx_diff",
    "use_method",
    "areaToConsider",
    "mask",
    "eroded",
    "lines_extended",
]
_NORMALIZATION = [
    "toBeNormalized",
    *_ESTIMATION,
    "background",
    "normalized_illumination",
]
_PICTURES = [
    "stretched",
    "eroded",
    "dilated",
    "grayGradient",
    "marker",
    "reconstructed",
    "reconstructed_inverted",
    "holesFilled",
]


def _page(color: bool = False) -> np.ndarray:
    """Unevenly lit page with text lines, specks and a photo-like block."""
    rows = np.linspace(225, 250, _H, dtype=np.float64)[:, None]
    page = np.repeat(rows, _W, axis=1).astype(np.uint8)
    for top in range(30, 160, 20):
        page[top : top + 8, 20:280] = 30
    for x in range(30, 270, 40):
        page[190, x] = 20
    rng = np.random.default_rng(7)
    page[230:360, 60:240] = rng.integers(40, 200, (130, 180), dtype=np.uint8)
    if not color:
        return page
    bgr = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    bgr[30:38, 20:280] = (0, 0, 200)
    return bgr


def _params(color_mode: ColorMode, **overrides) -> Params:
    options = {
        "color_mode": color_mode,
        "output_dpi": Dpi.uniform(300),
        "despeckle_level": DespeckleLevel.NORMAL,
        "binarization": BinarizationOptions(
            normalize_illumination=True,
            savitzky_golay_smoothing=True,
            morphological_smoothing=True,
        ),
        "picture_shape": PictureShapeOptions(shape=PictureShape.FREE),
    }
    options.update(overrides)
    return Params(**options)


def _labels(dbg: DebugImages) -> list[str]:
    """Recorded labels, with the background method choice folded into one name."""
    return [
        "use_method" if image.label.startswith("use_method") else image.label
        for image in dbg.images
    ]


def _generate(page: np.ndarray, params: Params) -> tuple[np.ndarray, list[str]]:
    with DebugImages() as dbg:
        image = generate_output(page, params, dbg=dbg).image
        assert all(item.path.is_file() for item in dbg.images)
        return image, _labels(dbg)


def _despeckle_labels(labels: list[str]) -> list[str]:
    start = labels.index("big_components_unified")
    return labels[start : labels.index("despeckled") + 1]


class TestBlackAndWhite:
    def test_records_upstream_steps_in_processing_order(self):
        _, labels = _generate(_page(), _params(ColorMode.BLACK_AND_WHITE))

        despeckling = _despeckle_labels(labels)
        assert labels == [
            *_NORMALIZATION,
            "maybeNormalized",
            "smoothed",
            "binarized_and_cropped",
            "edges_smoothed",
            *despeckling,
        ]
        assert despeckling[:2] == ["big_components_unified", "voronoi"]
        assert despeckling[-1] == "despeckled"

    def test_colour_segmentation_records_segmented_page_with_fill_zones(self):
        binarization = BinarizationOptions(
            normalize_illumination=False,
            savitzky_golay_smoothing=False,
            morphological_smoothing=False,
            color_segmenter=ColorSegmenterOptions(enabled=True, noise_reduction=0),
        )
        params = _params(
            ColorMode.BLACK_AND_WHITE,
            binarization=binarization,
            despeckle_level=DespeckleLevel.OFF,
        )

        _, labels = _generate(_page(color=True), params)

        assert labels == [
            "maybeNormalized",
            "binarized_and_cropped",
            "segmented",
            "segmented_with_fill_zones",
        ]


class TestMixed:
    def test_records_picture_detection_masks_and_composed_page(self):
        _, labels = _generate(_page(), _params(ColorMode.MIXED))

        despeckling = _despeckle_labels(labels)
        assert labels == [
            *_NORMALIZATION,
            "maybeNormalized",
            "smoothed",
            "binarized_and_cropped",
            "edges_smoothed",
            *_PICTURES,
            "bwMask",
            "bwMask with zones",
            *despeckling,
            "combined",
            "fillZones",
        ]

    def test_higher_search_sensitivity_adds_second_stretch(self):
        shape = PictureShapeOptions(
            shape=PictureShape.FREE, higher_search_sensitivity=True
        )

        _, labels = _generate(_page(), _params(ColorMode.MIXED, picture_shape=shape))

        assert labels[labels.index("holesFilled") + 1] == "stretched2"


class TestColour:
    def test_records_normalised_page_fill_zones_and_posterized(self):
        params = _params(
            ColorMode.COLOR_GRAYSCALE,
            color_common=ColorCommonOptions(
                normalize_illumination=True,
                posterization=PosterizationOptions(enabled=True, level=4),
            ),
        )

        _, labels = _generate(_page(color=True), params)

        assert labels == [*_NORMALIZATION, "maybeNormalized", "fillZones", "posterized"]


@pytest.mark.parametrize(
    ("color_mode", "color"),
    [
        (ColorMode.BLACK_AND_WHITE, False),
        (ColorMode.MIXED, False),
        (ColorMode.COLOR_GRAYSCALE, True),
    ],
)
def test_output_pixels_are_the_same_with_and_without_debug_images(color_mode, color):
    page = _page(color)
    params = _params(color_mode)

    debugged, _ = _generate(page, params)

    np.testing.assert_array_equal(debugged, generate_output(page, params).image)


def test_illumination_normalisation_records_estimation_and_keeps_its_result():
    gray = _page()

    with DebugImages() as dbg:
        normalized = normalize_illumination_gray(gray, dbg=dbg)
        labels = _labels(dbg)

    assert labels == _NORMALIZATION
    np.testing.assert_array_equal(normalized, normalize_illumination_gray(gray))


def test_despeckle_records_component_maps_in_colour_and_keeps_its_result():
    binary = np.full((200, 200), 255, dtype=np.uint8)
    binary[50:70, 20:180] = 0
    binary[80, 100] = 0
    binary[150, 30] = 0
    binary[150, 33] = 0

    with DebugImages() as dbg:
        despeckled = despeckle(binary, 300, 2.0, dbg)
        recorded = {image.label: cv2.imread(str(image.path)) for image in dbg.images}

    assert list(recorded)[:2] == ["big_components_unified", "voronoi"]
    voronoi = recorded["voronoi"]
    assert voronoi is not None
    assert voronoi.shape == (200, 200, 3)
    assert len(np.unique(voronoi.reshape(-1, 3), axis=0)) > 2
    np.testing.assert_array_equal(despeckled, despeckle(binary, 300, 2.0))


def test_dark_page_records_refining_pass_images():
    page = np.full((600, 450), 20, dtype=np.uint8)
    for top in range(60, 540, 30):
        page[top : top + 10, 40:410] = 230
    xform = ImageTransformation.from_size(450, 600, Dpi.uniform(300))

    with DebugImages() as dbg:
        is_black_on_white_refining(page, xform, dbg)
        labels = _labels(dbg)

    assert labels == ["bw150", "content_mask"]
