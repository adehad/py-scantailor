"""Render parameters derived from output settings.

This module provides the RenderParams class which computes rendering flags
from the stored output parameters. These flags control various aspects of
the output generation process.
"""

from dataclasses import dataclass
from typing import Self

from .color_mode import ColorMode
from .params import Params
from .splitting_options import SplittingOptions


@dataclass
class RenderParams:
    """Computed rendering parameters derived from output settings.

    This class computes various flags that control the rendering process
    based on the output parameters (color mode, binarization options, etc.).

    The flags are computed once at construction and cached for efficiency.

    Example:
        >>> from scantailor.filters.output import Params, RenderParams
        >>> params = Params(color_mode=ColorMode.BLACK_AND_WHITE)
        >>> render = RenderParams.from_params(params)
        >>> render.need_binarization
        True
        >>> render.binary_output
        True
    """

    # Core rendering flags
    fill_margins: bool = False
    fill_offcut: bool = False
    normalize_illumination: bool = False
    normalize_illumination_color: bool = False
    need_binarization: bool = False
    mixed_output: bool = False
    savitzky_golay_smoothing: bool = False
    morphological_smoothing: bool = False
    split_output: bool = False
    original_background: bool = False
    color_segmentation: bool = False
    posterize: bool = False

    @classmethod
    def from_params(
        cls,
        params: Params,
        splitting_options: SplittingOptions | None = None,
    ) -> Self:
        """Create RenderParams from output Params.

        Args:
            params: The output parameters to derive rendering flags from.
            splitting_options: Splitting configuration; defaults to
                ``params.splitting``.

        Returns:
            A new RenderParams instance with computed flags.
        """
        if splitting_options is None:
            splitting_options = params.splitting

        # Start with default values
        render = cls()

        # Determine if binarization is needed
        color_mode = params.color_mode
        if color_mode in (ColorMode.BLACK_AND_WHITE, ColorMode.MIXED):
            render.need_binarization = True
            if color_mode == ColorMode.MIXED:
                render.mixed_output = True

            # Handle split output for mixed mode
            if render.mixed_output and splitting_options.split_output:
                render.split_output = True
                if splitting_options.color_foreground:
                    render.need_binarization = False
                if render.need_binarization and splitting_options.original_background:
                    render.original_background = True

        color_common = params.color_common
        if render.need_binarization:
            binarization = params.binarization
            render.savitzky_golay_smoothing = binarization.savitzky_golay_smoothing
            render.morphological_smoothing = binarization.morphological_smoothing
            render.normalize_illumination = binarization.normalize_illumination
            if binarization.color_segmenter.enabled:
                render.color_segmentation = True
                render.posterize = color_common.posterization.enabled
        else:
            render.normalize_illumination = color_common.normalize_illumination
            render.posterize = color_common.posterization.enabled

        render.fill_margins = color_common.fill_margins
        render.fill_offcut = color_common.fill_offcut
        render.normalize_illumination_color = color_common.normalize_illumination

        return render

    @property
    def binary_output(self) -> bool:
        """True if output is purely binary (not mixed mode)."""
        return self.need_binarization and not self.mixed_output

    def needs_smoothing(self) -> bool:
        """True if any smoothing is enabled."""
        return self.savitzky_golay_smoothing or self.morphological_smoothing
