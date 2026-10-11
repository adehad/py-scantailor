"""Options for saving a mixed page as separate layers."""

from pydantic import BaseModel, ConfigDict


class SplittingOptions(BaseModel):
    """Options for splitting output into layers.

    Attributes:
        split_output: Save a foreground and a background image as well.
        original_background: Keep the original page as the background, with
            only the ink drawn over it. Applies to binarised foregrounds.
        color_foreground: The foreground keeps its colours instead of being
            binarised.
    """

    model_config = ConfigDict(frozen=True)

    split_output: bool = False
    original_background: bool = False
    color_foreground: bool = False
