"""Settings storage for Select Content filter."""

import threading
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, field_serializer, field_validator

from scantailor.core.models import ImageId, PageId, SubPage

from .params import Params


def _parse_key(key: object) -> object:
    """Turn a stored key into a PageId.

    Older projects keyed ``"<path>:<sub page number>"``, without the image page,
    so those entries are assigned to page 0 of the file.
    """
    if not isinstance(key, str):
        return key
    if key.startswith("{"):
        return PageId.model_validate_json(key)
    path, _, sub_page = key.rpartition(":")
    return PageId(
        image_id=ImageId(file_path=path), sub_page=SubPage(int(sub_page or 0))
    )


class Settings(BaseModel):
    """Thread-safe storage for Select Content filter settings.

    Stores per-page parameters including content and page boxes,
    as well as global settings like page detection box size.
    """

    params_by_page: dict[PageId, Params] = Field(default_factory=dict)
    page_detection_box_width: float = Field(default=0.0, ge=0.0)
    page_detection_box_height: float = Field(default=0.0, ge=0.0)
    page_detection_tolerance: float = Field(default=0.1, ge=0.0, le=1.0)

    model_config = {"arbitrary_types_allowed": True}
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @field_validator("params_by_page", mode="before")
    @classmethod
    def _migrate_keys(cls, value: Any) -> Any:
        """Accept keys written as JSON PageIds or in the older path-based form."""
        if not isinstance(value, dict):
            return value
        return {_parse_key(key): params for key, params in value.items()}

    @field_serializer("params_by_page")
    def _serialize_keys(self, value: dict[PageId, Params]) -> dict[str, Params]:
        return {page_id.model_dump_json(): params for page_id, params in value.items()}

    def get_params(self, page_id: PageId) -> Params:
        """Get parameters for a page, or default if not set."""
        with self._lock:
            return self.params_by_page.get(page_id, Params())

    def set_params(self, page_id: PageId, params: Params) -> None:
        """Set parameters for a page."""
        with self._lock:
            self.params_by_page[page_id] = params

    def is_params_set(self, page_id: PageId) -> bool:
        """Check if parameters have been set for a page."""
        with self._lock:
            return page_id in self.params_by_page

    def clear_page(self, page_id: PageId) -> None:
        """Remove parameters for a specific page."""
        with self._lock:
            self.params_by_page.pop(page_id, None)

    def clear(self) -> None:
        """Remove all stored parameters."""
        with self._lock:
            self.params_by_page.clear()

    def get_page_detection_box(self) -> tuple[float, float]:
        """Get page detection box dimensions (width, height)."""
        with self._lock:
            return (self.page_detection_box_width, self.page_detection_box_height)

    def set_page_detection_box(self, width: float, height: float) -> None:
        """Set page detection box dimensions."""
        with self._lock:
            self.page_detection_box_width = max(0.0, width)
            self.page_detection_box_height = max(0.0, height)

    def get_page_detection_tolerance(self) -> float:
        """Get page detection tolerance threshold."""
        with self._lock:
            return self.page_detection_tolerance

    def set_page_detection_tolerance(self, tolerance: float) -> None:
        """Set page detection tolerance (0.0 to 1.0)."""
        with self._lock:
            self.page_detection_tolerance = max(0.0, min(1.0, tolerance))

    def apply_params_to_pages(self, page_ids: list[PageId], params: Params) -> None:
        """Apply the same parameters to multiple pages."""
        with self._lock:
            for page_id in page_ids:
                self.params_by_page[page_id] = params.model_copy()

    def get_all_content_boxes(self) -> dict[PageId, tuple[float, float, float, float]]:
        """Get all stored content boxes as a dictionary.

        Returns dict mapping page to (x, y, width, height) tuple.
        """
        with self._lock:
            return {
                key: params.content_box.to_tuple()
                for key, params in self.params_by_page.items()
                if params.content_box.is_valid()
            }
