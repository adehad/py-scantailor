"""Stage-wide values and the post-dewarp rotation survive batch workers."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from scantailor.core import batch
from scantailor.core.batch import (
    PageDone,
    apply_stage_defaults,
    collect_stage_defaults,
    run_batch,
)
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineResult
from scantailor.core.stage_sequence import StageSequence
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.output.params import Params
from scantailor.filters.page_layout.guide import Guide


def _page(path: Path) -> PageId:
    return PageId(image_id=ImageId(file_path=path), sub_page=SubPage.SINGLE_PAGE)


class TestStageWideValuesReachWorkers:
    def test_the_middle_rectangle_flag_round_trips(self):
        stages = StageSequence()
        stages.page_layout_filter.settings.set_middle_rect_shown(False)

        fresh = StageSequence()
        apply_stage_defaults(fresh, collect_stage_defaults(stages))

        assert not fresh.page_layout_filter.settings.is_middle_rect_shown()

    def test_the_alignment_guides_round_trip_in_order(self):
        stages = StageSequence()
        guides = (
            Guide(position_mm=10.0, is_horizontal=True),
            Guide(position_mm=-5.0, is_horizontal=False),
        )
        stages.page_layout_filter.settings.set_guides(guides)

        fresh = StageSequence()
        apply_stage_defaults(fresh, collect_stage_defaults(stages))

        assert fresh.page_layout_filter.settings.get_guides() == guides

    def test_the_deskew_algorithm_flag_round_trips(self):
        stages = StageSequence()
        stages.deskew_filter.content_based = False

        fresh = StageSequence()
        apply_stage_defaults(fresh, collect_stage_defaults(stages))

        assert fresh.deskew_filter.content_based is False


class TestPostDeskewAngleComesBackFromWorkers:
    @staticmethod
    def _render_with_measured_angle(monkeypatch, tmp_path, angle: float):
        source = tmp_path / "in" / "page.tif"
        source.parent.mkdir()
        write_image(source, np.full((60, 40), 200, np.uint8), Dpi.uniform(300))
        page = _page(source)

        def fake_process_page(pixels, page_id, stages, options):
            return PipelineResult(
                image=np.full((60, 40), 255, np.uint8),
                page_id=page_id,
                metadata={"OUTPUT": {"was_dewarped": True, "post_deskew_angle": angle}},
                dpi=Dpi.uniform(300),
                transformation=ImageTransformation.from_size(40, 60, Dpi.uniform(300)),
            )

        monkeypatch.setattr(batch, "process_page", fake_process_page)
        stages = StageSequence()
        stored = Params(dewarping=DewarpingOptions(mode=DewarpingMode.MANUAL))
        stages.output_filter.settings.set_params(page, stored)
        events: list = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            run_batch(
                stages,
                [page],
                tmp_path / "out",
                max_workers=1,
                executor=pool,  # type: ignore[arg-type]
                on_progress=events.append,
            )
        return stages, page, events

    def test_the_measured_angle_is_stored_in_the_projects_parameters(
        self, monkeypatch, tmp_path
    ):
        stages, page, _events = self._render_with_measured_angle(
            monkeypatch, tmp_path, 1.25
        )

        stored = stages.output_filter.get_params(page).dewarping.post_deskew_angle
        assert stored == pytest.approx(1.25)

    def test_the_render_info_carries_the_measured_angle(self, monkeypatch, tmp_path):
        _stages, page, events = self._render_with_measured_angle(
            monkeypatch, tmp_path, -0.75
        )

        (done,) = [e for e in events if isinstance(e, PageDone) and e.page == page]
        assert done.info is not None
        assert done.info.post_deskew_angle == pytest.approx(-0.75)
