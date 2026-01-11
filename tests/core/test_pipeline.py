"""Tests for pipeline execution."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core import (
    FilterStage,
    ImageId,
    PageId,
    StageSequence,
    SubPage,
)
from scantailor.core.pipeline import (
    PipelineOptions,
    PipelineResult,
    process_batch,
    process_page,
)


@pytest.fixture
def page_id() -> PageId:
    """Create a test page ID."""
    return PageId(
        image_id=ImageId(file_path=Path("/test/scan.tiff")),
        sub_page=SubPage.SINGLE_PAGE,
    )


@pytest.fixture
def grayscale_image() -> np.ndarray:
    """Create a test grayscale image."""
    image = np.zeros((200, 300), dtype=np.uint8)
    # Add some content - dark text on light background
    image[:] = 200  # Light background
    image[50:150, 50:250] = 50  # Dark content area
    return image


@pytest.fixture
def sequence() -> StageSequence:
    """Create a stage sequence with default filters."""
    return StageSequence()


class TestPipelineOptions:
    """Tests for PipelineOptions."""

    def test_default_options(self) -> None:
        """Default options use full pipeline."""
        opts = PipelineOptions()
        assert opts.start_stage == FilterStage.FIX_ORIENTATION
        assert opts.end_stage == FilterStage.OUTPUT
        assert opts.dpi == 300.0
        assert opts.dewarper is None
        assert opts.detect_only is False

    def test_custom_start_stage(self) -> None:
        """Custom start stage is respected."""
        opts = PipelineOptions(start_stage=FilterStage.DESKEW)
        assert opts.start_stage == FilterStage.DESKEW

    def test_custom_end_stage(self) -> None:
        """Custom end stage is respected."""
        opts = PipelineOptions(end_stage=FilterStage.SELECT_CONTENT)
        assert opts.end_stage == FilterStage.SELECT_CONTENT


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_result_attributes(
        self, grayscale_image: np.ndarray, page_id: PageId
    ) -> None:
        """Result has expected attributes."""
        result = PipelineResult(
            image=grayscale_image,
            page_id=page_id,
            stages_completed=[FilterStage.FIX_ORIENTATION, FilterStage.PAGE_SPLIT],
            metadata={"test": "value"},
        )
        assert result.image is grayscale_image
        assert result.page_id == page_id
        assert len(result.stages_completed) == 2
        assert result.metadata["test"] == "value"


class TestProcessPage:
    """Tests for process_page function."""

    def test_process_full_pipeline(
        self,
        grayscale_image: np.ndarray,
        page_id: PageId,
        sequence: StageSequence,
    ) -> None:
        """Full pipeline executes all stages."""
        result = process_page(grayscale_image, page_id, sequence)

        assert isinstance(result, PipelineResult)
        assert result.page_id == page_id
        assert len(result.stages_completed) == 6
        assert FilterStage.FIX_ORIENTATION in result.stages_completed
        assert FilterStage.OUTPUT in result.stages_completed

    def test_process_partial_pipeline(
        self,
        grayscale_image: np.ndarray,
        page_id: PageId,
        sequence: StageSequence,
    ) -> None:
        """Partial pipeline executes only specified stages."""
        opts = PipelineOptions(
            start_stage=FilterStage.DESKEW,
            end_stage=FilterStage.SELECT_CONTENT,
        )
        result = process_page(grayscale_image, page_id, sequence, opts)

        assert len(result.stages_completed) == 2
        assert FilterStage.DESKEW in result.stages_completed
        assert FilterStage.SELECT_CONTENT in result.stages_completed
        assert FilterStage.FIX_ORIENTATION not in result.stages_completed
        assert FilterStage.OUTPUT not in result.stages_completed

    def test_detect_only_mode(
        self,
        grayscale_image: np.ndarray,
        page_id: PageId,
        sequence: StageSequence,
    ) -> None:
        """Detect-only mode doesn't transform image."""
        opts = PipelineOptions(
            end_stage=FilterStage.DESKEW,
            detect_only=True,
        )
        result = process_page(grayscale_image, page_id, sequence, opts)

        # Image should be unchanged (except for copy)
        np.testing.assert_array_equal(result.image, grayscale_image)
        # But metadata should still be populated
        assert "DESKEW" in result.metadata
        assert "skew_angle" in result.metadata["DESKEW"]

    def test_fix_orientation_metadata(
        self,
        grayscale_image: np.ndarray,
        page_id: PageId,
        sequence: StageSequence,
    ) -> None:
        """Fix orientation stage produces metadata."""
        opts = PipelineOptions(end_stage=FilterStage.FIX_ORIENTATION)
        result = process_page(grayscale_image, page_id, sequence, opts)

        assert "FIX_ORIENTATION" in result.metadata
        assert "rotation_degrees" in result.metadata["FIX_ORIENTATION"]

    def test_page_split_metadata(
        self,
        grayscale_image: np.ndarray,
        page_id: PageId,
        sequence: StageSequence,
    ) -> None:
        """Page split stage produces metadata."""
        opts = PipelineOptions(end_stage=FilterStage.PAGE_SPLIT)
        result = process_page(grayscale_image, page_id, sequence, opts)

        assert "PAGE_SPLIT" in result.metadata
        assert "layout_type" in result.metadata["PAGE_SPLIT"]

    def test_deskew_metadata(
        self,
        grayscale_image: np.ndarray,
        page_id: PageId,
        sequence: StageSequence,
    ) -> None:
        """Deskew stage produces metadata."""
        opts = PipelineOptions(end_stage=FilterStage.DESKEW)
        result = process_page(grayscale_image, page_id, sequence, opts)

        assert "DESKEW" in result.metadata
        assert "skew_angle" in result.metadata["DESKEW"]
        assert "is_manual" in result.metadata["DESKEW"]

    def test_select_content_metadata(
        self,
        grayscale_image: np.ndarray,
        page_id: PageId,
        sequence: StageSequence,
    ) -> None:
        """Select content stage produces metadata."""
        opts = PipelineOptions(end_stage=FilterStage.SELECT_CONTENT)
        result = process_page(grayscale_image, page_id, sequence, opts)

        assert "SELECT_CONTENT" in result.metadata
        assert "content_detection_mode" in result.metadata["SELECT_CONTENT"]

    def test_page_layout_metadata(
        self,
        grayscale_image: np.ndarray,
        page_id: PageId,
        sequence: StageSequence,
    ) -> None:
        """Page layout stage produces metadata."""
        opts = PipelineOptions(end_stage=FilterStage.PAGE_LAYOUT)
        result = process_page(grayscale_image, page_id, sequence, opts)

        assert "PAGE_LAYOUT" in result.metadata
        assert "margins" in result.metadata["PAGE_LAYOUT"]
        assert "alignment" in result.metadata["PAGE_LAYOUT"]

    def test_output_metadata(
        self,
        grayscale_image: np.ndarray,
        page_id: PageId,
        sequence: StageSequence,
    ) -> None:
        """Output stage produces metadata."""
        result = process_page(grayscale_image, page_id, sequence)

        assert "OUTPUT" in result.metadata
        assert "color_mode" in result.metadata["OUTPUT"]
        assert "is_binary" in result.metadata["OUTPUT"]

    def test_output_produces_binary(
        self,
        grayscale_image: np.ndarray,
        page_id: PageId,
        sequence: StageSequence,
    ) -> None:
        """Output stage produces binary image by default."""
        result = process_page(grayscale_image, page_id, sequence)

        # Default is black and white mode
        unique_values = np.unique(result.image)
        assert len(unique_values) <= 2  # Binary


class TestProcessBatch:
    """Tests for process_batch function."""

    def test_batch_processes_all_images(
        self,
        grayscale_image: np.ndarray,
        sequence: StageSequence,
    ) -> None:
        """Batch processing handles multiple images."""
        page_id1 = PageId(
            image_id=ImageId(file_path=Path("/test/scan1.tiff")),
            sub_page=SubPage.SINGLE_PAGE,
        )
        page_id2 = PageId(
            image_id=ImageId(file_path=Path("/test/scan2.tiff")),
            sub_page=SubPage.SINGLE_PAGE,
        )

        images = [
            (grayscale_image, page_id1),
            (grayscale_image.copy(), page_id2),
        ]

        results = process_batch(images, sequence)

        assert len(results) == 2
        assert results[0].page_id == page_id1
        assert results[1].page_id == page_id2

    def test_batch_with_options(
        self,
        grayscale_image: np.ndarray,
        sequence: StageSequence,
    ) -> None:
        """Batch processing respects options."""
        page_id1 = PageId(
            image_id=ImageId(file_path=Path("/test/scan1.tiff")),
            sub_page=SubPage.SINGLE_PAGE,
        )

        images = [(grayscale_image, page_id1)]
        opts = PipelineOptions(end_stage=FilterStage.DESKEW)

        results = process_batch(images, sequence, opts)

        assert (
            len(results[0].stages_completed) == 3
        )  # FIX_ORIENTATION, PAGE_SPLIT, DESKEW

    def test_empty_batch(self, sequence: StageSequence) -> None:
        """Empty batch returns empty list."""
        results = process_batch([], sequence)
        assert results == []
