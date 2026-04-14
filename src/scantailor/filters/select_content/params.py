"""Parameters for Select Content filter."""

from typing import Self

from pydantic import BaseModel, Field

from .content_box import ContentBox, PageBox, PhysicalSize
from .dependencies import Dependencies
from .detection import ContentDetectionMode, PageDetectionMode


class Params(BaseModel):
    """Per-page parameters for content and page detection.

    Stores both the detected/manual boxes and the detection modes.
    """

    content_box: ContentBox = Field(default_factory=ContentBox)
    page_box: PageBox = Field(default_factory=PageBox)
    content_size_mm: PhysicalSize = Field(default_factory=PhysicalSize)
    content_detection_mode: ContentDetectionMode = ContentDetectionMode.AUTO
    page_detection_mode: PageDetectionMode = PageDetectionMode.DISABLED
    fine_tune_corners: bool = False
    dependencies: Dependencies = Field(default_factory=Dependencies)

    def is_content_auto(self) -> bool:
        """Return True if content detection is automatic."""
        return self.content_detection_mode == ContentDetectionMode.AUTO

    def is_content_manual(self) -> bool:
        """Return True if content is manually specified."""
        return self.content_detection_mode == ContentDetectionMode.MANUAL

    def is_content_disabled(self) -> bool:
        """Return True if content detection is disabled."""
        return self.content_detection_mode == ContentDetectionMode.DISABLED

    def is_page_auto(self) -> bool:
        """Return True if page detection is automatic."""
        return self.page_detection_mode == PageDetectionMode.AUTO

    def is_page_manual(self) -> bool:
        """Return True if page box is manually specified."""
        return self.page_detection_mode == PageDetectionMode.MANUAL

    def is_page_disabled(self) -> bool:
        """Return True if page detection is disabled (full image)."""
        return self.page_detection_mode == PageDetectionMode.DISABLED

    def with_content_box(
        self,
        box: ContentBox,
        mode: ContentDetectionMode | None = None,
    ) -> Self:
        """Return new params with updated content box."""
        return self.model_copy(
            update={
                "content_box": box,
                "content_detection_mode": mode
                if mode is not None
                else self.content_detection_mode,
            }
        )

    def with_page_box(
        self,
        box: PageBox,
        mode: PageDetectionMode | None = None,
    ) -> Self:
        """Return new params with updated page box."""
        return self.model_copy(
            update={
                "page_box": box,
                "page_detection_mode": mode
                if mode is not None
                else self.page_detection_mode,
            }
        )

    def with_manual_content(self, box: ContentBox) -> Self:
        """Return new params with manually specified content box."""
        return self.with_content_box(box, ContentDetectionMode.MANUAL)

    def with_manual_page(self, box: PageBox) -> Self:
        """Return new params with manually specified page box."""
        return self.with_page_box(box, PageDetectionMode.MANUAL)

    def with_auto_content(self) -> Self:
        """Return new params with content detection set to AUTO."""
        return self.model_copy(
            update={"content_detection_mode": ContentDetectionMode.AUTO}
        )

    def with_auto_page(self) -> Self:
        """Return new params with page detection set to AUTO."""
        return self.model_copy(update={"page_detection_mode": PageDetectionMode.AUTO})

    def with_disabled_content(self) -> Self:
        """Return new params with content detection disabled."""
        return self.model_copy(
            update={"content_detection_mode": ContentDetectionMode.DISABLED}
        )

    def with_disabled_page(self) -> Self:
        """Return new params with page detection disabled."""
        return self.model_copy(
            update={"page_detection_mode": PageDetectionMode.DISABLED}
        )

    def with_fine_tune(self, enabled: bool) -> Self:
        """Return new params with fine-tune corners setting."""
        return self.model_copy(update={"fine_tune_corners": enabled})

    def ensure_content_within_page(self) -> Self:
        """Return new params with content box clipped to page box."""
        if self.page_box.is_empty():
            return self

        page_as_content = self.page_box.to_content_box()
        clipped = self.content_box.intersection(page_as_content)

        if clipped.is_empty() and not self.content_box.is_empty():
            # Content completely outside page - use page as content
            return self.model_copy(update={"content_box": page_as_content})

        return self.model_copy(update={"content_box": clipped})
