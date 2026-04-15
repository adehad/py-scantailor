"""Output filter implementation.

This filter is the final stage in the processing pipeline. It generates
the final output images with binarization, despeckle, and other processing
applied.
"""

import numpy as np
from numpy.typing import NDArray

from scantailor.core.models import Dpi, PageId

from .binarization import BinarizationMethod, BinarizationOptions
from .color_mode import ColorMode
from .despeckle import DespeckleLevel
from .generator import OutputResult, generate_output
from .params import Params
from .settings import Settings


class Filter:
    """Output filter.

    This filter handles the final processing of scanned images, including:
    - Binarization (conversion to black and white)
    - Despeckling (noise removal)
    - Color mode selection

    Example:
        >>> from scantailor.filters.output import Filter, Settings
        >>> from scantailor.core import ImageId, PageId, SubPage, Dpi
        >>> from pathlib import Path
        >>> import numpy as np
        >>>
        >>> settings = Settings()
        >>> filter = Filter(settings)
        >>>
        >>> # Process an image
        >>> image = np.zeros((100, 100), dtype=np.uint8)
        >>> page_id = PageId(
        ...     image_id=ImageId(file_path=Path("/scan.tiff")),
        ...     sub_page=SubPage.SINGLE_PAGE,
        ... )
        >>> result = filter.process(image, page_id)
    """

    name: str = "Output"

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the filter.

        Args:
            settings: Settings storage for per-page parameters.
                     If None, a new Settings instance is created.
        """
        self._settings = settings if settings is not None else Settings()

    @property
    def settings(self) -> Settings:
        """Get the settings storage."""
        return self._settings

    def get_params(self, page_id: "PageId") -> Params:
        """Get the current parameters for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The stored parameters or defaults if not set.
        """
        return self._settings.get_params(page_id)

    def set_params(self, page_id: "PageId", params: Params) -> None:
        """Set the parameters for a page.

        Args:
            page_id: The page to set parameters for.
            params: The parameters to store.
        """
        self._settings.set_params(page_id, params)

    def is_params_set(self, page_id: "PageId") -> bool:
        """Check if parameters have been set for a page."""
        return self._settings.is_params_set(page_id)

    def get_output_dpi(self, page_id: "PageId") -> Dpi:
        """Get the output DPI for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The output DPI.
        """
        return self.get_params(page_id).output_dpi

    def set_output_dpi(self, page_id: "PageId", dpi: Dpi) -> Params:
        """Set the output DPI for a page.

        Args:
            page_id: The page to update.
            dpi: The new output DPI.

        Returns:
            The updated Params.
        """
        params = self.get_params(page_id).with_output_dpi(dpi)
        self._settings.set_params(page_id, params)
        return params

    def get_color_mode(self, page_id: "PageId") -> ColorMode:
        """Get the color mode for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The color mode.
        """
        return self.get_params(page_id).color_mode

    def set_color_mode(self, page_id: "PageId", mode: ColorMode) -> Params:
        """Set the color mode for a page.

        Args:
            page_id: The page to update.
            mode: The new color mode.

        Returns:
            The updated Params.
        """
        params = self.get_params(page_id).with_color_mode(mode)
        self._settings.set_params(page_id, params)
        return params

    def get_despeckle_level(self, page_id: "PageId") -> DespeckleLevel:
        """Get the despeckle level for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The despeckle level.
        """
        return self.get_params(page_id).despeckle_level

    def set_despeckle_level(self, page_id: "PageId", level: DespeckleLevel) -> Params:
        """Set the despeckle level for a page.

        Args:
            page_id: The page to update.
            level: The new despeckle level.

        Returns:
            The updated Params.
        """
        params = self.get_params(page_id).with_despeckle_level(level)
        self._settings.set_params(page_id, params)
        return params

    def get_binarization_method(self, page_id: "PageId") -> BinarizationMethod:
        """Get the binarization method for a page.

        Args:
            page_id: The page to look up.

        Returns:
            The binarization method.
        """
        return self.get_params(page_id).binarization.method

    def set_binarization_method(
        self, page_id: "PageId", method: BinarizationMethod
    ) -> Params:
        """Set the binarization method for a page.

        Args:
            page_id: The page to update.
            method: The new binarization method.

        Returns:
            The updated Params.
        """
        params = self.get_params(page_id).with_binarization_method(method)
        self._settings.set_params(page_id, params)
        return params

    def set_binarization_options(
        self, page_id: "PageId", options: BinarizationOptions
    ) -> Params:
        """Set the binarization options for a page.

        Args:
            page_id: The page to update.
            options: The new binarization options.

        Returns:
            The updated Params.
        """
        params = self.get_params(page_id).with_binarization(options)
        self._settings.set_params(page_id, params)
        return params

    def apply_to_pages(self, page_ids: list["PageId"], params: Params) -> None:
        """Apply parameters to multiple pages.

        Args:
            page_ids: List of pages to set parameters for.
            params: The parameters to apply.
        """
        self._settings.apply_params_to_pages(page_ids, params)

    def process(
        self,
        image: NDArray[np.uint8],
        page_id: "PageId",
    ) -> OutputResult:
        """Process an image and generate final output.

        Args:
            image: The input image (grayscale or color).
            page_id: The page ID.

        Returns:
            OutputResult with the processed image.
        """
        params = self.get_params(page_id)
        return generate_output(image, params)

    def process_with_params(
        self,
        image: NDArray[np.uint8],
        params: Params,
    ) -> OutputResult:
        """Process an image with specific parameters.

        This method processes without storing or looking up settings.

        Args:
            image: The input image.
            params: The parameters to use.

        Returns:
            OutputResult with the processed image.
        """
        return generate_output(image, params)
