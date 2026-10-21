"""A render task logs on a pool thread under the name of the page it renders."""

from pathlib import Path
from typing import Any

from scantailor.app.ui.page_renderer import PageRenderer
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.stage_sequence import FilterStage, StageSequence


def test_a_failed_render_is_logged_with_the_page_name(
    qtbot: Any, tmp_path: Path, log_records: list[dict[str, Any]]
) -> None:
    renderer = PageRenderer()
    page = PageId(
        image_id=ImageId(file_path=tmp_path / "missing.tif"),
        sub_page=SubPage.RIGHT_PAGE,
    )

    with qtbot.waitSignal(renderer.render_failed, timeout=10_000):
        renderer.request(page, StageSequence(), FilterStage.DESKEW)
    renderer.wait_for_done()

    [failure] = [r for r in log_records if r["message"].startswith("Cannot render")]
    assert failure["level"] == "WARNING"
    assert failure["extra"]["page"] == "missing.tif/right_page"
