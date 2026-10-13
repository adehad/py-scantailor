"""What it takes to map points through the dewarp that produced a page."""

from pydantic import BaseModel, ConfigDict


class DewarpSpec(BaseModel):
    """What it takes to map points through the dewarp that produced a page.

    Attributes:
        top: Polyline along the top edge of the page model, left to right, as
            (x, y) pairs in source image pixels.
        bottom: Polyline along the bottom edge, left to right.
        depth_perception: The output parameter the dewarper was built with.
        post_deskew_angle: Rotation, in degrees, applied to the dewarped image
            about its centre; positive turns the page clockwise.
    """

    model_config = ConfigDict(frozen=True)

    top: tuple[tuple[float, float], ...]
    bottom: tuple[tuple[float, float], ...]
    depth_perception: float
    post_deskew_angle: float = 0.0
