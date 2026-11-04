from __future__ import annotations

from pathlib import Path

from tenq.jsonl import append, load_lines, load_records


def test_the_latest_record_per_key_wins(tmp_path: Path) -> None:
    path = tmp_path / "r" / "x.jsonl"
    append(path, {"key": "a", "v": 1})
    append(path, {"key": "a", "v": 2})
    append(path, {"key": "b", "v": 3})
    assert load_records(path) == {"a": {"key": "a", "v": 2}, "b": {"key": "b", "v": 3}}


def test_a_line_a_crash_cut_short_is_skipped_and_ended(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"key": "a"}\n{"key": "b", "v"')
    append(path, {"key": "c"})
    assert sorted(load_records(path)) == ["a", "c"]
    assert [r.get("key") for r in load_lines(path)] == ["a", "c"]


def test_lines_without_keys_are_kept_in_order_and_junk_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "n.jsonl"
    path.write_text('{"n": 1}\n[1, 2]\nnot json\n{"n": 2}\n')
    assert load_lines(path) == [{"n": 1}, {"n": 2}]
    assert load_records(path) == {}
    assert load_lines(tmp_path / "missing.jsonl") == []
