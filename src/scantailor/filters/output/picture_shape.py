"""Options for automatic picture detection in mixed output."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PictureShape(StrEnum):
    """How detected pictures are shaped.

    OFF: No automatic detection; only manual zones make pictures.
    FREE: The detected area is used as it is.
    RECTANGULAR: Each cluster of detected area becomes its bounding rectangle.
    """

    OFF = "off"
    FREE = "free"
    RECTANGULAR = "rectangular"


class PictureShapeOptions(BaseModel):
    """Automatic picture detection settings.

    Attributes:
        shape: Detection mode.
        sensitivity: For rectangular pictures, the percentage of picture pixels
            a border line must hold to stay inside the rectangle; 100 keeps the
            full bounding box.
        higher_search_sensitivity: Detect fainter pictures at the price of more
            false positives.
    """

    model_config = ConfigDict(frozen=True)

    shape: PictureShape = PictureShape.FREE
    sensitivity: int = Field(default=100, ge=0, le=100)
    higher_search_sensitivity: bool = False
