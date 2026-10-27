"""Tests for the record that lets a batch keep a page's unchanged output files."""

import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scantailor.core.batch import PageDone, run_batch, spawn_pool
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, OrthogonalRotation, PageId
from scantailor.core.output_reuse import build_record, is_reusable, output_params_hash
from scantailor.core.project import FileStamp, ImageInfo, PageOutputRecord, Project
from scantailor.core.settings import TiffCompression, TiffCompressions
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.page_layout.alignment import Alignment
from scantailor.filters.page_layout.params import ContentSize, Params
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox
from tests.dewarping.synthetic import PAGE_DPI, curved_text_page

COMPRESSIONS = TiffCompressions()


def _make_source(path: Path, shade: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((400, 300), 220, dtype=np.uint8)
    pixels[100:300, 80:220] = shade
    write_image(path, pixels, Dpi.uniform(300))


def _configure(stages: StageSequence, page: PageId) -> None:
    stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    stages.deskew_filter.set_manual_angle(page, 0.0)
    stages.select_content_filter.set_manual_content(
        page, ContentBox(x=80, y=100, width=140, height=200)
    )


def _hash(
    stages: StageSequence,
    page: PageId,
    dpi_override: Dpi | None = None,
    compressions: TiffCompressions = COMPRESSIONS,
) -> str:
    return output_params_hash(
        stages,
        page,
        dpi_override=dpi_override,
        default_dpi=300.0,
        tiff_compressions=compressions,
        aggregate_mm=stages.page_layout_filter.settings.get_aggregate_hard_size_mm(),
    )


@pytest.fixture
def page(tmp_path: Path) -> PageId:
    path = tmp_path / "in" / "scan.tif"
    _make_source(path)
    return PageId(image_id=ImageId(file_path=path))


@pytest.fixture
def stages(page: PageId) -> StageSequence:
    sequence = StageSequence()
    _configure(sequence, page)
    return sequence


class TestOutputParamsHash:
    def test_same_settings_give_the_same_hash(self, stages, page):
        assert _hash(stages, page) == _hash(stages, page)

    @pytest.mark.parametrize(
        "edit",
        [
            pytest.param(
                lambda s, p: s.fix_orientation_filter.set_rotation(
                    p.image_id, OrthogonalRotation().rotate_clockwise()
                ),
                id="fix-orientation",
            ),
            pytest.param(
                lambda s, p: s.page_split_filter.set_layout_type(
                    p, LayoutType.PAGE_PLUS_OFFCUT
                ),
                id="page-split",
            ),
            pytest.param(
                lambda s, p: s.deskew_filter.set_manual_angle(p, 1.5), id="deskew"
            ),
            pytest.param(
                lambda s, p: s.select_content_filter.set_manual_content(
                    p, ContentBox(x=70, y=100, width=150, height=200)
                ),
                id="select-content",
            ),
            pytest.param(
                lambda s, p: s.page_layout_filter.set_uniform_margins(p, 12.0),
                id="margins",
            ),
            pytest.param(
                lambda s, p: s.output_filter.set_color_mode(
                    p, ColorMode.COLOR_GRAYSCALE
                ),
                id="output",
            ),
            pytest.param(
                lambda s, _p: s.output_filter.settings.set_default_dpi(
                    Dpi.uniform(400)
                ),
                id="stage-default",
            ),
        ],
    )
    def test_an_edit_at_any_stage_changes_the_hash(
        self,
        stages: StageSequence,
        page: PageId,
        edit: Callable[[StageSequence, PageId], None],
    ):
        before = _hash(stages, page)

        edit(stages, page)

        assert _hash(stages, page) != before

    def test_a_project_dpi_override_changes_the_hash(self, stages, page):
        assert _hash(stages, page, Dpi.uniform(200)) != _hash(stages, page)

    def test_tiff_compression_changes_the_hash(self, stages, page):
        other = TiffCompressions(color=TiffCompression.DEFLATE)

        assert _hash(stages, page, compressions=other) != _hash(stages, page)

    def test_aligned_page_follows_the_aggregate_size_of_other_pages(
        self, tmp_path, stages, page
    ):
        other = PageId(image_id=ImageId(file_path=tmp_path / "in" / "other.tif"))
        before = _hash(stages, page)

        _grow_content(stages, other, width_mm=500.0)

        assert _hash(stages, page) != before

    def test_page_not_aligned_with_others_ignores_their_size(
        self, tmp_path, stages, page
    ):
        stages.page_layout_filter.set_alignment(page, Alignment(is_null=True))
        other = PageId(image_id=ImageId(file_path=tmp_path / "in" / "other.tif"))
        before = _hash(stages, page)

        _grow_content(stages, other, width_mm=500.0)

        assert _hash(stages, page) == before

    def test_view_only_defaults_leave_the_hash_alone(self, stages, page):
        layout = stages.page_layout_filter.settings
        before = _hash(stages, page)

        layout.set_middle_rect_shown(not layout.is_middle_rect_shown())

        assert _hash(stages, page) == before


def _grow_content(stages: StageSequence, page: PageId, width_mm: float) -> None:
    params = Params().model_copy(
        update={"content_size_mm": ContentSize(width_mm=width_mm, height_mm=10.0)}
    )
    stages.page_layout_filter.set_params(page, params)


def _write_output(out: Path, name: str = "scan.tif") -> Path:
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_image(path, np.full((10, 10), 255, dtype=np.uint8), Dpi.uniform(600))
    return path


def _touch_later(path: Path) -> None:
    status = path.stat()
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 5_000_000_000))


class TestIsReusable:
    def _record(self, page: PageId, out: Path) -> PageOutputRecord:
        written = [_write_output(out), _write_output(out, "foreground/scan.tif")]
        record = build_record(page, "digest", out, written)
        assert record is not None
        return record

    def test_matching_record_and_files_are_reusable(self, tmp_path, page):
        record = self._record(page, tmp_path / "out")

        assert is_reusable(record, "digest", tmp_path / "out", "scan.tif")

    def test_record_keys_files_by_posix_path_inside_the_output_folder(
        self, tmp_path, page
    ):
        record = self._record(page, tmp_path / "out")

        assert set(record.files) == {"scan.tif", "foreground/scan.tif"}

    def test_changed_settings_are_not_reusable(self, tmp_path, page):
        record = self._record(page, tmp_path / "out")

        assert not is_reusable(record, "other", tmp_path / "out", "scan.tif")

    def test_replaced_source_file_is_not_reusable(self, tmp_path, page):
        record = self._record(page, tmp_path / "out")

        _make_source(page.image_id.file_path, shade=90)
        _touch_later(page.image_id.file_path)

        assert not is_reusable(record, "digest", tmp_path / "out", "scan.tif")

    def test_edited_output_file_is_not_reusable(self, tmp_path, page):
        out = tmp_path / "out"
        record = self._record(page, out)

        _touch_later(out / "scan.tif")

        assert not is_reusable(record, "digest", out, "scan.tif")

    def test_deleted_output_file_is_not_reusable(self, tmp_path, page):
        out = tmp_path / "out"
        record = self._record(page, out)

        (out / "scan.tif").unlink()

        assert not is_reusable(record, "digest", out, "scan.tif")

    def test_deleted_layer_file_is_not_reusable(self, tmp_path, page):
        out = tmp_path / "out"
        record = self._record(page, out)

        (out / "foreground" / "scan.tif").unlink()

        assert not is_reusable(record, "digest", out, "scan.tif")

    def test_renamed_output_file_is_not_reusable(self, tmp_path, page):
        record = self._record(page, tmp_path / "out")

        assert not is_reusable(record, "digest", tmp_path / "out", "scan_1L.tif")

    def test_other_output_folder_is_not_reusable(self, tmp_path, page):
        record = self._record(page, tmp_path / "out")

        assert not is_reusable(record, "digest", tmp_path / "elsewhere", "scan.tif")

    def test_no_record_is_built_when_a_written_file_is_missing(self, tmp_path, page):
        out = tmp_path / "out"

        assert build_record(page, "digest", out, [out / "scan.tif"]) is None


class TestProjectOutputRecords:
    def _project(self, page: PageId, tmp_path: Path) -> Project:
        project = Project(
            output_directory=tmp_path / "out", images=[ImageInfo(id=page.image_id)]
        )
        record = PageOutputRecord(
            page=page,
            params_hash="abc",
            source=FileStamp(size=10, mtime_ns=20),
            files={"scan.tif": FileStamp(size=30, mtime_ns=40)},
        )
        project.set_output_records({page: record})
        return project

    def test_records_survive_a_save_and_load(self, tmp_path, page):
        project = self._project(page, tmp_path)
        path = tmp_path / "project.json"

        project.save(path)
        loaded = Project.load(path)

        assert loaded.output_record_map() == project.output_record_map()

    def test_saved_record_stores_the_page_path_relative_to_the_project(
        self, tmp_path, page
    ):
        path = tmp_path / "project.json"

        self._project(page, tmp_path).save(path)

        stored = json.loads(path.read_text(encoding="utf-8"))
        file_path = stored["output_records"][0]["page"]["image_id"]["file_path"]
        assert not Path(file_path).is_absolute()

    def test_project_saved_without_records_loads_with_none(self, tmp_path, page):
        path = tmp_path / "project.json"
        self._project(page, tmp_path).save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["output_records"]
        path.write_text(json.dumps(data), encoding="utf-8")

        assert Project.load(path).output_records == []

    def test_new_records_replace_those_of_the_same_page(self, tmp_path, page):
        project = self._project(page, tmp_path)
        newer = project.output_records[0].model_copy(update={"params_hash": "new"})

        project.set_output_records({page: newer})

        assert [r.params_hash for r in project.output_records] == ["new"]

    def test_removing_a_page_drops_its_record(self, tmp_path, page):
        project = self._project(page, tmp_path)

        project.remove_pages([page])

        assert project.output_records == []

    def test_relinking_moves_the_record_to_the_new_path(self, tmp_path, page):
        project = self._project(page, tmp_path)
        moved = tmp_path / "moved" / "scan.tif"

        project.remap_image_paths(lambda _path: moved)

        assert project.output_records[0].page.image_id.file_path == moved


@pytest.fixture(scope="module")
def pool():
    with spawn_pool(2) as executor:
        yield executor


@pytest.mark.slow
class TestBatchKeepsUnchangedOutput:
    def _run(
        self,
        stages: StageSequence,
        pages: list[PageId],
        out: Path,
        pool: ProcessPoolExecutor,
        records: dict[PageId, PageOutputRecord] | None,
    ):
        events: list[object] = []
        result = run_batch(
            stages,
            pages,
            out,
            max_workers=2,
            executor=pool,
            on_progress=events.append,
            output_records=records,
            tiff_compressions=COMPRESSIONS,
        )
        assert result.failed == {}
        return result, events

    @pytest.fixture
    def pages(self, tmp_path: Path, stages: StageSequence, page: PageId):
        other = PageId(image_id=ImageId(file_path=tmp_path / "in" / "other.tif"))
        _make_source(other.image_id.file_path, shade=80)
        _configure(stages, other)
        return [page, other]

    def test_second_batch_keeps_every_unchanged_page(
        self, tmp_path, stages, pages, pool
    ):
        out = tmp_path / "out"
        first, _ = self._run(stages, pages, out, pool, {})
        stamps = {p: FileStamp.of(path) for p, path in first.written.items()}

        second, events = self._run(stages, pages, out, pool, first.output_records)

        assert sorted(second.reused) == sorted(pages)
        assert second.render_info == {}
        assert second.output_records == first.output_records
        assert {p: FileStamp.of(path) for p, path in second.written.items()} == stamps
        done = [e for e in events if isinstance(e, PageDone)]
        assert len(done) == len(pages)
        assert all(e.reused for e in done)

    def test_edited_page_is_rendered_and_the_other_kept(
        self, tmp_path, stages, pages, pool
    ):
        out = tmp_path / "out"
        first, _ = self._run(stages, pages, out, pool, {})

        stages.output_filter.set_color_mode(pages[0], ColorMode.COLOR_GRAYSCALE)
        second, _ = self._run(stages, pages, out, pool, first.output_records)

        assert second.reused == [pages[1]]
        assert set(second.render_info) == {pages[0]}
        assert second.output_records[pages[0]] != first.output_records[pages[0]]

    def test_edit_that_moves_the_shared_page_size_renders_every_aligned_page(
        self, tmp_path, stages, pages, pool
    ):
        out = tmp_path / "out"
        first, _ = self._run(stages, pages, out, pool, {})

        stages.select_content_filter.set_manual_content(
            pages[0], ContentBox(x=20, y=20, width=260, height=360)
        )
        second, _ = self._run(stages, pages, out, pool, first.output_records)

        assert second.reused == []

    def test_page_rendered_again_keeps_the_shared_page_size(
        self, tmp_path, stages, pages, pool
    ):
        out = tmp_path / "out"
        first, _ = self._run(stages, pages, out, pool, {})

        # Smaller than the other page's box, so the shared page size stays.
        stages.select_content_filter.set_manual_content(
            pages[0], ContentBox(x=100, y=150, width=60, height=50)
        )
        second, _ = self._run(stages, pages, out, pool, first.output_records)

        assert second.reused == [pages[1]]
        sizes = {p: Image.open(path).size for p, path in second.written.items()}
        assert sizes[pages[0]] == sizes[pages[1]]

    def test_deleted_output_file_is_rendered_again(self, tmp_path, stages, pages, pool):
        out = tmp_path / "out"
        first, _ = self._run(stages, pages, out, pool, {})

        first.written[pages[1]].unlink()
        second, _ = self._run(stages, pages, out, pool, first.output_records)

        assert second.reused == [pages[0]]
        assert second.written[pages[1]].is_file()

    def test_batch_without_records_renders_every_page(
        self, tmp_path, stages, pages, pool
    ):
        out = tmp_path / "out"
        first, _ = self._run(stages, pages, out, pool, {})

        second, _ = self._run(stages, pages, out, pool, None)

        assert second.reused == []
        assert set(second.output_records) == set(first.output_records)


@pytest.mark.slow
def test_auto_dewarped_page_is_kept_after_its_found_model_is_stored(
    tmp_path: Path, pool: ProcessPoolExecutor
):
    path = tmp_path / "in" / "curved.tif"
    path.parent.mkdir(parents=True)
    write_image(path, curved_text_page(), Dpi.uniform(PAGE_DPI))
    page = PageId(image_id=ImageId(file_path=path))
    stages = StageSequence()
    stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    stages.deskew_filter.set_manual_angle(page, 0.0)
    params = stages.output_filter.get_params(page).model_copy(
        update={
            "color_mode": ColorMode.COLOR_GRAYSCALE,
            "output_dpi": Dpi.uniform(PAGE_DPI),
            "dewarping": DewarpingOptions(mode=DewarpingMode.AUTO, post_deskew=True),
        }
    )
    stages.output_filter.set_params(page, params)
    out = tmp_path / "out"

    first = run_batch(stages, [page], out, max_workers=2, executor=pool)
    second = run_batch(
        stages,
        [page],
        out,
        max_workers=2,
        executor=pool,
        output_records=first.output_records,
    )

    assert first.render_info[page].distortion_model is not None
    assert second.reused == [page]
