"""A batch past Page Split carries a detected spread on as its two halves."""

import inspect
import shutil
from pathlib import Path
from typing import Any

import pytest

import scantailor.core.batch as batch_module
from scantailor.core.batch import BatchEvent, BatchResult, PagesListed, run_batch
from scantailor.core.image_io import list_image_pages
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.project import ImageMetadata, Project
from scantailor.core.propagation import ProjectPageLister
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.page_split.layout_type import LayoutType

SPREAD = Path(__file__).parents[1] / "fixtures/xml_projects/split_and_mixed/scans"


@pytest.fixture
def spread_project(tmp_path: Path) -> Project:
    """A project holding the fixture spread as one page, before Page Split ran."""
    source = tmp_path / "scans" / "spread.tif"
    source.parent.mkdir()
    shutil.copy(SPREAD / "spread.tif", source)
    _, metadata = list_image_pages(source)[0]
    project = Project()
    project.add_image(
        ImageId(file_path=source),
        ImageMetadata(width=metadata.width, height=metadata.height, dpi=metadata.dpi),
    )
    return project


def _halves(project: Project) -> list[PageId]:
    image_id = project.images[0].id
    return [
        PageId(image_id=image_id, sub_page=SubPage.LEFT_PAGE),
        PageId(image_id=image_id, sub_page=SubPage.RIGHT_PAGE),
    ]


def test_a_detected_spread_is_written_as_its_two_halves(
    spread_project: Project, tmp_path: Path
) -> None:
    stages = StageSequence()
    events: list[BatchEvent] = []

    result = run_batch(
        stages,
        spread_project.get_pages(),
        tmp_path / "out",
        max_workers=1,
        on_progress=events.append,
        page_lister=ProjectPageLister(spread_project, stages),
    )

    assert sorted(path.name for path in result.written.values()) == [
        "spread_1L.tif",
        "spread_2R.tif",
    ]
    assert [e.pages for e in events if isinstance(e, PagesListed)] == [
        tuple(_halves(spread_project))
    ]


@pytest.mark.slow
def test_without_a_page_lister_the_spread_stays_one_page(
    spread_project: Project, tmp_path: Path
) -> None:
    result = run_batch(
        StageSequence(), spread_project.get_pages(), tmp_path / "out", max_workers=1
    )

    assert [path.name for path in result.written.values()] == ["spread.tif"]


@pytest.mark.slow
def test_a_batch_to_page_split_lists_no_new_pages(
    spread_project: Project, tmp_path: Path
) -> None:
    stages = StageSequence()
    events: list[BatchEvent] = []

    result = run_batch(
        stages,
        spread_project.get_pages(),
        tmp_path / "out",
        max_workers=1,
        on_progress=events.append,
        end_stage=FilterStage.PAGE_SPLIT,
        page_lister=ProjectPageLister(spread_project, stages),
    )

    assert result.detected == spread_project.get_pages()
    assert not any(isinstance(event, PagesListed) for event in events)


@pytest.mark.slow
def test_an_unreadable_two_page_image_fails_both_of_its_pages(
    spread_project: Project, tmp_path: Path
) -> None:
    spread_project.images[0].num_sub_pages = 2
    spread_project.images[0].id.file_path.write_bytes(b"not an image")
    stages = StageSequence()

    result = run_batch(
        stages,
        spread_project.get_pages(),
        tmp_path / "out",
        max_workers=1,
        page_lister=ProjectPageLister(spread_project, stages),
    )

    assert sorted(result.failed) == sorted(_halves(spread_project))
    assert result.written == {}


def test_reached_images_name_each_image_once_measured_first(
    spread_project: Project,
) -> None:
    left, right = _halves(spread_project)
    other = PageId(image_id=ImageId(file_path=Path("other.png")))
    result = BatchResult(detected=[left, right], failed={other: "unreadable"})

    assert result.reached_images() == [left.image_id, other.image_id]


@pytest.mark.slow
def test_a_stale_whole_spread_output_is_deleted_and_logged(
    spread_project: Project, tmp_path: Path, log_messages: list[str]
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "spread.tif").write_bytes(b"stale")
    stages = StageSequence()

    result = run_batch(
        stages,
        spread_project.get_pages(),
        out_dir,
        max_workers=1,
        page_lister=ProjectPageLister(spread_project, stages),
    )

    assert result.replaced == spread_project.get_pages()
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "spread_1L.tif",
        "spread_2R.tif",
    ]
    assert [line for line in log_messages if line.startswith("deleted")] == [
        f"deleted {out_dir / 'spread.tif'}: Page Split replaced its page"
    ]


@pytest.mark.slow
def test_a_source_named_like_the_replaced_output_is_never_deleted(
    spread_project: Project,
) -> None:
    source = spread_project.images[0].id.file_path
    stages = StageSequence()

    run_batch(
        stages,
        spread_project.get_pages(),
        source.parent,
        max_workers=1,
        page_lister=ProjectPageLister(spread_project, stages),
    )

    assert source.is_file()
    assert (source.parent / "spread_1L.tif").is_file()


@pytest.fixture
def split_pass_images(monkeypatch: pytest.MonkeyPatch) -> list[list[ImageId]]:
    """Record the images each batch sends through the Page Split pass."""
    sent: list[list[ImageId]] = []
    detect = batch_module._detect_geometry

    signature = inspect.signature(detect)

    def spy(*args: Any, **kwargs: Any) -> list[PageId]:
        call = signature.bind(*args, **kwargs).arguments
        if call["end_stage"] == FilterStage.PAGE_SPLIT:
            sent.append([page.image_id for page in call["pages"]])
        return detect(*args, **kwargs)

    monkeypatch.setattr(batch_module, "_detect_geometry", spy)
    return sent


def _run(project: Project, stages: StageSequence, out_dir: Path) -> None:
    run_batch(
        stages,
        project.get_pages(),
        out_dir,
        max_workers=1,
        page_lister=ProjectPageLister(project, stages),
    )


@pytest.mark.slow
def test_a_rerun_decodes_no_scan_for_page_split_when_every_split_is_stored(
    spread_project: Project, tmp_path: Path, split_pass_images: list[list[ImageId]]
) -> None:
    stages = StageSequence()
    _run(spread_project, stages, tmp_path / "out")
    spread_project.images[0].num_sub_pages = 2

    _run(spread_project, stages, tmp_path / "out")

    assert split_pass_images == [[spread_project.images[0].id], []]


@pytest.mark.slow
def test_a_scan_whose_split_changed_since_it_was_stored_goes_through_the_pass(
    spread_project: Project, tmp_path: Path, split_pass_images: list[list[ImageId]]
) -> None:
    stages = StageSequence()
    _run(spread_project, stages, tmp_path / "out")
    spread_project.images[0].num_sub_pages = 2
    stages.page_split_filter.set_layout_type(
        PageId(image_id=spread_project.images[0].id), LayoutType.SINGLE_PAGE_UNCUT
    )

    _run(spread_project, stages, tmp_path / "out")

    image_id = spread_project.images[0].id
    assert split_pass_images == [[image_id], [image_id]]
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == ["spread.tif"]


@pytest.mark.slow
def test_a_same_named_source_that_fails_page_split_keeps_the_labels_of_the_others(
    spread_project: Project, tmp_path: Path
) -> None:
    unreadable = tmp_path / "earlier" / "spread.tif"
    unreadable.parent.mkdir()
    unreadable.write_bytes(b"not an image")
    spread_project.images.insert(
        0,
        spread_project.images[0].model_copy(
            update={"id": ImageId(file_path=unreadable)}
        ),
    )
    stages = StageSequence()

    result = run_batch(
        stages,
        spread_project.get_pages(),
        tmp_path / "out",
        max_workers=1,
        page_lister=ProjectPageLister(spread_project, stages),
    )

    assert sorted(path.name for path in result.written.values()) == [
        "spread(1)_1L.tif",
        "spread(1)_2R.tif",
    ]
