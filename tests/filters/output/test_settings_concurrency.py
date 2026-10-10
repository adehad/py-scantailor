"""Thread-safety tests for the output Settings store."""

import threading
from pathlib import Path

from scantailor.core.models import Dpi, ImageId, PageId
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.params import Params
from scantailor.filters.output.settings import Settings


def _page_id(index: int) -> PageId:
    return PageId(image_id=ImageId(file_path=Path(f"/page_{index}.png")))


def _run_threads(workers: list) -> None:
    threads = [threading.Thread(target=worker) for worker in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)


class TestSettingsConcurrency:
    """Parallel readers and writers see consistent state."""

    def test_concurrent_writers_lose_no_pages(self):
        settings = Settings()
        params = Params(color_mode=ColorMode.COLOR_GRAYSCALE)

        def writer(offset: int):
            def work() -> None:
                for i in range(200):
                    settings.set_params(_page_id(offset * 1000 + i), params)

            return work

        _run_threads([writer(n) for n in range(8)])

        assert all(
            settings.is_params_set(_page_id(n * 1000 + i))
            for n in range(8)
            for i in range(200)
        )

    def test_readers_never_observe_a_torn_default_during_updates(self):
        settings = Settings()
        seen: list[Params] = []
        errors: list[Exception] = []

        def writer() -> None:
            for i in range(300):
                settings.set_default_dpi(Dpi.uniform(300 + i % 2 * 300))
                settings.set_default_color_mode(
                    ColorMode.MIXED if i % 2 else ColorMode.BLACK_AND_WHITE
                )
                settings.apply_params_to_pages([_page_id(i)], Params())
                settings.clear_params(_page_id(i - 1))

        def reader() -> None:
            try:
                for i in range(300):
                    seen.append(settings.get_params(_page_id(i)))
            except Exception as exc:
                errors.append(exc)

        _run_threads([writer, reader, reader, reader])

        assert not errors
        assert all(p.output_dpi in (Dpi.uniform(300), Dpi.uniform(600)) for p in seen)

    def test_clear_all_racing_with_writes_leaves_a_consistent_store(self):
        settings = Settings()

        def writer() -> None:
            for i in range(500):
                settings.set_params(_page_id(i), Params())

        def clearer() -> None:
            for _ in range(100):
                settings.clear_all()

        _run_threads([writer, clearer])
        settings.clear_all()

        assert not settings.is_params_set(_page_id(0))
