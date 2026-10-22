"""Inserting and removing pages in a project, and erasing their output files."""

from pathlib import Path

import pytest

from scantailor.core.batch import erase_output_files, output_file_names
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.project import (
    ImageInfo,
    ImageMetadata,
    Project,
    StageParamsEntry,
)


def image(name: str, num_sub_pages: int = 1, page: int = 0) -> ImageInfo:
    return ImageInfo(
        id=ImageId(file_path=Path(name), page=page),
        metadata=ImageMetadata(width=100, height=100),
        num_sub_pages=num_sub_pages,
    )


def single(name: str, page: int = 0) -> PageId:
    return PageId(image_id=ImageId(file_path=Path(name), page=page))


def half(name: str, sub_page: SubPage) -> PageId:
    return PageId(image_id=ImageId(file_path=Path(name)), sub_page=sub_page)


@pytest.fixture
def project() -> Project:
    project = Project()
    project.images = [image("a.png"), image("b.png", 2), image("c.png")]
    return project


class TestInsert:
    def test_before_a_single_image_goes_directly_ahead_of_it(self, project: Project):
        added = project.insert_image(
            image("n.png"), ImageId(file_path=Path("c.png")), before=True
        )

        assert added == [single("n.png")]
        assert [p.image_id.file_path.name for p in project.get_pages()] == [
            "a.png",
            "b.png",
            "b.png",
            "n.png",
            "c.png",
        ]

    def test_after_a_split_image_goes_behind_both_halves(self, project: Project):
        project.insert_image(
            image("n.png"), ImageId(file_path=Path("b.png")), before=False
        )

        names = [p.image_id.file_path.name for p in project.get_pages()]
        assert names == ["a.png", "b.png", "b.png", "n.png", "c.png"]

    def test_before_a_split_image_goes_ahead_of_both_halves(self, project: Project):
        project.insert_image(
            image("n.png"), ImageId(file_path=Path("b.png")), before=True
        )

        names = [p.image_id.file_path.name for p in project.get_pages()]
        assert names == ["a.png", "n.png", "b.png", "b.png", "c.png"]

    def test_inserting_a_split_image_reports_both_halves_in_layout_order(
        self, project: Project
    ):
        project.layout_direction = "RTL"

        added = project.insert_image(image("n.png", 2), None, before=True)

        assert added == [
            half("n.png", SubPage.RIGHT_PAGE),
            half("n.png", SubPage.LEFT_PAGE),
        ]
        assert project.images[-1].id.file_path.name == "n.png"

    def test_before_nothing_appends_at_the_end_of_an_empty_project(self):
        empty = Project()

        added = empty.insert_image(image("n.png"), None, before=True)

        assert added == [single("n.png")]
        assert empty.get_pages() == [single("n.png")]

    def test_after_nothing_inserts_nothing(self, project: Project):
        assert project.insert_image(image("n.png"), None, before=False) == []
        assert len(project.images) == 3

    def test_beside_an_unknown_image_inserts_nothing(self, project: Project):
        unknown = ImageId(file_path=Path("zzz.png"))

        assert project.insert_image(image("n.png"), unknown, before=True) == []
        assert len(project.images) == 3

    def test_both_halves_removed_on_a_one_page_image_are_reset(self, project: Project):
        info = image("n.png").model_copy(
            update={"left_half_removed": True, "right_half_removed": True}
        )

        added = project.insert_image(info, None, before=True)

        assert added == [single("n.png")]

    def test_a_one_page_image_with_a_removed_half_keeps_the_other_half(
        self, project: Project
    ):
        info = image("n.png").model_copy(update={"right_half_removed": True})

        added = project.insert_image(info, None, before=True)

        assert added == [half("n.png", SubPage.LEFT_PAGE)]

    def test_multi_page_tiff_pages_stay_in_order_after_an_image(self, project: Project):
        tiff = [image("t.tif", page=n) for n in (1, 2, 3)]

        added = project.insert_images(
            tiff, ImageId(file_path=Path("a.png")), before=False
        )

        assert added == [single("t.tif", 1), single("t.tif", 2), single("t.tif", 3)]
        assert project.get_pages()[1:4] == added

    def test_multi_page_tiff_pages_stay_in_order_before_an_image(
        self, project: Project
    ):
        tiff = [image("t.tif", page=n) for n in (1, 2, 3)]

        added = project.insert_images(
            tiff, ImageId(file_path=Path("c.png")), before=True
        )

        assert project.get_pages()[-4:-1] == added
        assert project.get_pages()[-1] == single("c.png")

    def test_inserted_images_keep_the_settings_of_existing_pages(
        self, project: Project
    ):
        entry = StageParamsEntry(page=single("c.png"), params={"x": 1})
        project.stage_params = {"deskew": [entry]}

        project.insert_image(image("n.png"), None, before=True)

        assert project.stage_params == {"deskew": [entry]}


class TestRemove:
    def test_removing_a_single_page_removes_its_image(self, project: Project):
        removed = project.remove_pages([single("a.png")])

        assert removed == [single("a.png")]
        assert [info.id.file_path.name for info in project.images] == ["b.png", "c.png"]

    def test_removing_one_half_keeps_the_other_as_a_one_page_image(
        self, project: Project
    ):
        removed = project.remove_pages([half("b.png", SubPage.LEFT_PAGE)])

        assert removed == [half("b.png", SubPage.LEFT_PAGE)]
        assert project.get_pages() == [
            single("a.png"),
            half("b.png", SubPage.RIGHT_PAGE),
            single("c.png"),
        ]
        info = project.images[1]
        assert info.num_sub_pages == 1
        assert info.left_half_removed
        assert not info.right_half_removed

    def test_removing_the_right_half_keeps_the_left(self, project: Project):
        project.remove_pages([half("b.png", SubPage.RIGHT_PAGE)])

        assert half("b.png", SubPage.LEFT_PAGE) in project.get_pages()
        assert project.images[1].right_half_removed

    def test_removing_both_halves_removes_the_whole_image(self, project: Project):
        project.remove_pages(
            [half("b.png", SubPage.LEFT_PAGE), half("b.png", SubPage.RIGHT_PAGE)]
        )

        assert [info.id.file_path.name for info in project.images] == ["a.png", "c.png"]

    def test_removing_the_last_remaining_half_removes_the_image(self, project: Project):
        project.remove_pages([half("b.png", SubPage.LEFT_PAGE)])

        project.remove_pages([half("b.png", SubPage.RIGHT_PAGE)])

        assert [info.id.file_path.name for info in project.images] == ["a.png", "c.png"]

    def test_pages_not_in_the_project_are_ignored(self, project: Project):
        assert project.remove_pages([single("nope.png")]) == []
        assert project.remove_pages([half("a.png", SubPage.LEFT_PAGE)]) == []
        assert len(project.images) == 3

    def test_settings_of_remaining_pages_survive_and_removed_ones_go(
        self, project: Project
    ):
        kept = StageParamsEntry(page=single("c.png"), params={"x": 1})
        kept_half = StageParamsEntry(
            page=half("b.png", SubPage.RIGHT_PAGE), params={"y": 2}
        )
        dropped = StageParamsEntry(page=single("a.png"), params={"z": 3})
        dropped_half = StageParamsEntry(
            page=half("b.png", SubPage.LEFT_PAGE), params={"w": 4}
        )
        orientation = StageParamsEntry(page=single("b.png"), params={"r": 5})
        project.stage_params = {
            "deskew": [dropped, dropped_half, kept_half, kept],
            "fix_orientation": [orientation],
            "margins": [dropped],
        }

        project.remove_pages([single("a.png"), half("b.png", SubPage.LEFT_PAGE)])

        assert project.stage_params == {
            "deskew": [kept_half, kept],
            "fix_orientation": [orientation],
        }

    def test_the_selected_page_is_cleared_when_it_is_removed(self, project: Project):
        project.selected_page = single("a.png")

        project.remove_pages([single("a.png")])

        assert project.selected_page is None

    def test_the_selected_page_survives_when_it_remains(self, project: Project):
        project.selected_page = single("c.png")

        project.remove_pages([single("a.png")])

        assert project.selected_page == single("c.png")


class TestEraseOutputFiles:
    @pytest.fixture
    def output(self, tmp_path: Path) -> Path:
        folder = tmp_path / "out"
        folder.mkdir()
        return folder

    def touch(self, folder: Path, name: str) -> Path:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return path

    def test_a_single_page_removal_erases_its_file_and_split_variants(
        self, project: Project, output: Path
    ):
        pages = project.get_pages()
        names = output_file_names(
            [
                *pages,
                half("a.png", SubPage.LEFT_PAGE),
                half("a.png", SubPage.RIGHT_PAGE),
            ]
        )
        plain = self.touch(output, names[single("a.png")])
        left = self.touch(output, names[half("a.png", SubPage.LEFT_PAGE)])
        right = self.touch(output, names[half("a.png", SubPage.RIGHT_PAGE)])
        other = self.touch(output, names[single("c.png")])

        deleted = erase_output_files([single("a.png")], output, pages)

        assert set(deleted) == {plain, left, right}
        assert not plain.exists() and not left.exists() and not right.exists()
        assert other.exists()

    def test_removing_one_half_leaves_the_other_halfs_file(
        self, project: Project, output: Path
    ):
        pages = project.get_pages()
        names = output_file_names(pages)
        left = self.touch(output, names[half("b.png", SubPage.LEFT_PAGE)])
        right = self.touch(output, names[half("b.png", SubPage.RIGHT_PAGE)])

        erase_output_files([half("b.png", SubPage.LEFT_PAGE)], output, pages)

        assert not left.exists()
        assert right.exists()

    def test_split_layer_copies_are_erased_with_the_page(
        self, project: Project, output: Path
    ):
        pages = project.get_pages()
        name = output_file_names(pages)[single("a.png")]
        layers = [
            self.touch(output / folder, name)
            for folder in ("foreground", "background", "original_background")
        ]

        erase_output_files([single("a.png")], output, pages)

        assert not any(layer.exists() for layer in layers)

    def test_labels_of_same_named_sources_come_from_the_full_page_list(
        self, tmp_path: Path, output: Path
    ):
        first = single("one/p.png")
        second = single("two/p.png")
        pages = [first, second]
        names = output_file_names(pages)
        keep = self.touch(output, names[first])
        gone = self.touch(output, names[second])

        erase_output_files([second], output, pages)

        assert keep.exists()
        assert not gone.exists()

    def test_stored_labels_decide_which_same_named_file_is_erased(self, output: Path):
        first = single("one/p.png")
        second = single("two/p.png")
        labels = {first.image_id.file_path: 1, second.image_id.file_path: 0}
        names = output_file_names([first, second], file_labels=labels)
        kept = self.touch(output, names[first])
        gone = self.touch(output, names[second])

        erase_output_files([second], output, [first, second], file_labels=labels)

        assert kept.exists()
        assert not gone.exists()

    def test_a_source_image_in_the_output_folder_is_never_erased(self, output: Path):
        source = output / "a.tif"
        source.write_bytes(b"x")
        page = single(str(source))

        deleted = erase_output_files([page], output, [page], keep=[source])

        assert deleted == []
        assert source.exists()

    def test_files_outside_the_output_folder_are_untouched(
        self, project: Project, tmp_path: Path, output: Path
    ):
        pages = project.get_pages()
        name = output_file_names(pages)[single("a.png")]
        outside = self.touch(tmp_path, name)

        erase_output_files([single("a.png")], output, pages)

        assert outside.exists()

    def test_unrelated_files_and_folders_in_the_output_folder_are_untouched(
        self, project: Project, output: Path
    ):
        pages = project.get_pages()
        stray = self.touch(output, "notes.txt")
        folder_named_like_output = output / output_file_names(pages)[single("a.png")]
        folder_named_like_output.mkdir()

        deleted = erase_output_files([single("a.png")], output, pages)

        assert deleted == []
        assert stray.exists()
        assert folder_named_like_output.is_dir()

    def test_a_missing_output_folder_is_not_an_error(
        self, project: Project, tmp_path: Path
    ):
        pages = project.get_pages()

        assert erase_output_files([single("a.png")], tmp_path / "nope", pages) == []


class TestImagePages:
    def test_a_two_page_image_is_listed_once_by_its_left_page(
        self, project: Project
    ) -> None:
        assert project.get_image_pages() == [
            single("a.png"),
            half("b.png", SubPage.LEFT_PAGE),
            single("c.png"),
        ]

    def test_right_to_left_reading_lists_a_spread_by_its_right_page(
        self, project: Project
    ) -> None:
        project.layout_direction = "RTL"

        assert project.get_image_pages()[1] == half("b.png", SubPage.RIGHT_PAGE)

    def test_an_image_with_a_removed_half_is_listed_by_the_half_that_stays(
        self, project: Project
    ) -> None:
        project.images[1] = ImageInfo(
            id=ImageId(file_path=Path("b.png")), num_sub_pages=1, left_half_removed=True
        )

        assert project.get_image_pages()[1] == half("b.png", SubPage.RIGHT_PAGE)
