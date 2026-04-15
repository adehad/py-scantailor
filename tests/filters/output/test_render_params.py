"""Tests for RenderParams."""

from scantailor.filters.output.binarization import BinarizationOptions
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.params import Params
from scantailor.filters.output.render_params import RenderParams, SplittingOptions


class TestSplittingOptions:
    """Tests for SplittingOptions class."""

    def test_default_values(self) -> None:
        """Default values are all False."""
        options = SplittingOptions()
        assert options.split_output is False
        assert options.original_background is False
        assert options.color_foreground is False

    def test_custom_values(self) -> None:
        """Can create with custom values."""
        options = SplittingOptions(
            split_output=True,
            original_background=True,
            color_foreground=False,
        )
        assert options.split_output is True
        assert options.original_background is True
        assert options.color_foreground is False


class TestRenderParams:
    """Tests for RenderParams class."""

    def test_from_bw_params(self) -> None:
        """B&W mode requires binarization."""
        params = Params(color_mode=ColorMode.BLACK_AND_WHITE)
        render = RenderParams.from_params(params)

        assert render.need_binarization is True
        assert render.mixed_output is False
        assert render.binary_output is True

    def test_from_grayscale_params(self) -> None:
        """Grayscale mode does not require binarization."""
        params = Params(color_mode=ColorMode.COLOR_GRAYSCALE)
        render = RenderParams.from_params(params)

        assert render.need_binarization is False
        assert render.mixed_output is False
        assert render.binary_output is False

    def test_from_mixed_params(self) -> None:
        """Mixed mode requires binarization but is not purely binary."""
        params = Params(color_mode=ColorMode.MIXED)
        render = RenderParams.from_params(params)

        assert render.need_binarization is True
        assert render.mixed_output is True
        assert render.binary_output is False

    def test_morphological_smoothing_from_binarization(self) -> None:
        """Morphological smoothing comes from binarization options."""
        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(morphological_smoothing=True),
        )
        render = RenderParams.from_params(params)

        assert render.morphological_smoothing is True

    def test_morphological_smoothing_disabled(self) -> None:
        """Morphological smoothing can be disabled."""
        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(morphological_smoothing=False),
        )
        render = RenderParams.from_params(params)

        assert render.morphological_smoothing is False

    def test_normalize_illumination_from_binarization(self) -> None:
        """Illumination normalization comes from binarization options."""
        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(normalize_illumination=True),
        )
        render = RenderParams.from_params(params)

        assert render.normalize_illumination is True

    def test_fill_margins_default(self) -> None:
        """Fill margins is enabled by default."""
        params = Params()
        render = RenderParams.from_params(params)

        assert render.fill_margins is True

    def test_split_output_with_mixed_mode(self) -> None:
        """Split output is enabled only in mixed mode with splitting options."""
        params = Params(color_mode=ColorMode.MIXED)
        splitting = SplittingOptions(split_output=True)
        render = RenderParams.from_params(params, splitting)

        assert render.split_output is True
        assert render.mixed_output is True

    def test_split_output_ignored_for_bw(self) -> None:
        """Split output is ignored for pure B&W mode."""
        params = Params(color_mode=ColorMode.BLACK_AND_WHITE)
        splitting = SplittingOptions(split_output=True)
        render = RenderParams.from_params(params, splitting)

        assert render.split_output is False  # Not mixed mode

    def test_original_background_with_split(self) -> None:
        """Original background enabled when splitting with binarization."""
        params = Params(color_mode=ColorMode.MIXED)
        splitting = SplittingOptions(
            split_output=True,
            original_background=True,
        )
        render = RenderParams.from_params(params, splitting)

        assert render.original_background is True

    def test_color_foreground_disables_binarization(self) -> None:
        """Color foreground mode disables binarization."""
        params = Params(color_mode=ColorMode.MIXED)
        splitting = SplittingOptions(
            split_output=True,
            color_foreground=True,
        )
        render = RenderParams.from_params(params, splitting)

        assert render.need_binarization is False
        assert render.split_output is True

    def test_needs_smoothing(self) -> None:
        """needs_smoothing returns True if any smoothing enabled."""
        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(morphological_smoothing=True),
        )
        render = RenderParams.from_params(params)

        assert render.needs_smoothing() is True

    def test_needs_smoothing_false(self) -> None:
        """needs_smoothing returns False if no smoothing enabled."""
        params = Params(
            color_mode=ColorMode.BLACK_AND_WHITE,
            binarization=BinarizationOptions(morphological_smoothing=False),
        )
        render = RenderParams.from_params(params)

        assert render.needs_smoothing() is False

    def test_grayscale_no_binarization_flags(self) -> None:
        """Grayscale mode doesn't set binarization-related flags."""
        params = Params(
            color_mode=ColorMode.COLOR_GRAYSCALE,
            binarization=BinarizationOptions(
                morphological_smoothing=True,
                normalize_illumination=True,
            ),
        )
        render = RenderParams.from_params(params)

        # These should not be set since no binarization needed
        assert render.morphological_smoothing is False
        assert render.normalize_illumination is False
