import os
from pathlib import Path

from brutus.store import load_references, write_atomic
from tests.images import jpeg_bytes


class TestLoadReferences:
    def test_given_reference_photos_when_loaded_then_named_by_file_and_sorted(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "Spock.png").write_bytes(jpeg_bytes())
        (tmp_path / "Captain Kirk.jpg").write_bytes(jpeg_bytes())
        (tmp_path / "notes.txt").write_text("not an image")

        references = load_references(tmp_path)

        assert [r.name for r in references] == ["Captain Kirk", "Spock"]
        assert all(r.jpeg[:2] == b"\xff\xd8" for r in references)

    def test_given_missing_directory_when_loaded_then_empty(
        self, tmp_path: Path
    ) -> None:
        assert load_references(tmp_path / "missing") == ()

    def test_given_unchanged_files_when_loaded_twice_then_not_processed_again(
        self, tmp_path: Path
    ) -> None:
        photo = tmp_path / "Kirk.jpg"
        photo.write_bytes(jpeg_bytes())

        first = load_references(tmp_path)
        second = load_references(tmp_path)

        assert first[0] is second[0]

    def test_given_changed_file_when_loaded_again_then_reprocessed(
        self, tmp_path: Path
    ) -> None:
        photo = tmp_path / "Kirk.jpg"
        photo.write_bytes(jpeg_bytes(color="white"))
        first = load_references(tmp_path)

        photo.write_bytes(jpeg_bytes(color="black"))
        stat = photo.stat()
        os.utime(photo, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        assert load_references(tmp_path)[0].jpeg != first[0].jpeg


class TestWriteAtomic:
    def test_given_data_when_written_then_file_complete_and_no_temp_left(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "out" / "analysis.json"

        write_atomic(target, b"first")
        write_atomic(target, b"second")

        assert target.read_bytes() == b"second"
        assert [p.name for p in target.parent.iterdir()] == ["analysis.json"]
