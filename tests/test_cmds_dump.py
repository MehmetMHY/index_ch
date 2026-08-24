"""Tests for retrieve/cmds/dump.py: build_dump, unique_path, report_skipped."""

import json
import os
import pytest

from retrieve.cmds.dump import build_dump, unique_path, report_skipped


@pytest.fixture
def chat_file(tmp_path):
    """Create a temp chat file and return its path + raw dict."""
    raw = {
        "messages": [
            {"user": "hello", "bot": "hi", "time": 1000},
            {"user": "bye", "bot": "goodbye", "time": 1100},
        ],
        "platform": "test",
        "model": "test-model",
        "base_url": "http://localhost",
        "timestamp": 1000,
    }
    f = tmp_path / "ch_session_1000.json"
    f.write_text(json.dumps(raw))
    return str(f), raw


class TestBuildDump:
    def test_single_chat(self, chat_file):
        path, raw = chat_file
        meta = {1: {"file_path": path, "archived": False}}
        merged, skipped = build_dump([1], meta)
        assert skipped == []
        assert len(merged["messages"]) == 2
        assert merged["source_files"] == ["ch_session_1000.json"]
        assert merged["platform"] == "test"

    def test_source_file_tag(self, chat_file):
        path, raw = chat_file
        meta = {1: {"file_path": path, "archived": False}}
        merged, _ = build_dump([1], meta)
        for msg in merged["messages"]:
            assert msg["source_file"] == "ch_session_1000.json"

    def test_multiple_chats_ordered_oldest_first(self, tmp_path):
        raw1 = {"messages": [{"user": "a", "bot": "b", "time": 1000}], "platform": "p"}
        raw2 = {"messages": [{"user": "c", "bot": "d", "time": 2000}], "platform": "p"}
        f1 = tmp_path / "ch_session_1000.json"
        f2 = tmp_path / "ch_session_2000.json"
        f1.write_text(json.dumps(raw1))
        f2.write_text(json.dumps(raw2))
        meta = {
            1: {"file_path": str(f1), "last_message_epoch": 1000, "archived": False},
            2: {"file_path": str(f2), "last_message_epoch": 2000, "archived": False},
        }
        merged, skipped = build_dump([2, 1], meta)  # pass in reverse order
        # should be sorted oldest first
        assert merged["source_files"] == [
            "ch_session_1000.json",
            "ch_session_2000.json",
        ]

    def test_messages_no_interleave(self, tmp_path):
        raw1 = {
            "messages": [{"user": "a1", "bot": "b1"}, {"user": "a2", "bot": "b2"}],
            "platform": "p",
        }
        raw2 = {"messages": [{"user": "c1", "bot": "d1"}], "platform": "p"}
        f1 = tmp_path / "ch_session_1000.json"
        f2 = tmp_path / "ch_session_2000.json"
        f1.write_text(json.dumps(raw1))
        f2.write_text(json.dumps(raw2))
        meta = {
            1: {"file_path": str(f1), "last_message_epoch": 1000, "archived": False},
            2: {"file_path": str(f2), "last_message_epoch": 2000, "archived": False},
        }
        merged, _ = build_dump([1, 2], meta)
        # chat 1's messages should be contiguous
        users = [m["user"] for m in merged["messages"]]
        assert users == ["a1", "a2", "c1"]

    def test_archived_skipped(self, chat_file):
        path, raw = chat_file
        meta = {1: {"file_path": path, "archived": True}}
        result = build_dump([1], meta)
        assert result is None

    def test_archived_skip_message(self, chat_file):
        path, _ = chat_file
        meta = {
            1: {"file_path": path, "archived": True},
            2: {"file_path": path, "archived": False},
        }
        merged, skipped = build_dump([1, 2], meta)
        assert len(skipped) == 1
        assert "archived" in skipped[0][1]

    def test_unreadable_file_skipped(self, tmp_path):
        meta = {1: {"file_path": str(tmp_path / "nonexistent.json"), "archived": False}}
        result = build_dump([1], meta)
        assert result is None

    def test_malformed_json_skipped(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json")
        meta = {1: {"file_path": str(f), "archived": False}}
        result = build_dump([1], meta)
        assert result is None

    def test_missing_messages_key_skipped(self, tmp_path):
        f = tmp_path / "no_msgs.json"
        f.write_text(json.dumps({"other": "data"}))
        meta = {1: {"file_path": str(f), "archived": False}}
        result = build_dump([1], meta)
        assert result is None

    def test_root_fields_from_newest(self, tmp_path):
        raw1 = {
            "messages": [{"user": "a", "bot": "b", "time": 1000}],
            "platform": "old",
            "model": "old",
        }
        raw2 = {
            "messages": [{"user": "c", "bot": "d", "time": 2000}],
            "platform": "new",
            "model": "new",
        }
        f1 = tmp_path / "ch_session_1000.json"
        f2 = tmp_path / "ch_session_2000.json"
        f1.write_text(json.dumps(raw1))
        f2.write_text(json.dumps(raw2))
        meta = {
            1: {"file_path": str(f1), "last_message_epoch": 1000, "archived": False},
            2: {"file_path": str(f2), "last_message_epoch": 2000, "archived": False},
        }
        merged, _ = build_dump([1, 2], meta)
        assert merged["platform"] == "new"
        assert merged["model"] == "new"

    def test_all_fail_returns_none(self, tmp_path, capsys):
        meta = {1: {"file_path": str(tmp_path / "nonexistent.json"), "archived": False}}
        result = build_dump([1], meta)
        assert result is None
        assert "Nothing dumped" in capsys.readouterr().out

    def test_mixed_success_and_fail(self, tmp_path):
        raw = {"messages": [{"user": "a", "bot": "b"}], "platform": "p"}
        f = tmp_path / "good.json"
        f.write_text(json.dumps(raw))
        meta = {
            1: {"file_path": str(f), "archived": False},
            2: {"file_path": str(tmp_path / "bad.json"), "archived": False},
        }
        merged, skipped = build_dump([1, 2], meta)
        assert merged is not None
        assert len(merged["messages"]) == 1
        assert len(skipped) == 1


class TestUniquePath:
    def test_non_existing(self, tmp_path):
        path = str(tmp_path / "file.json")
        assert unique_path(path) == path

    def test_collision(self, tmp_path):
        path = tmp_path / "file.json"
        path.write_text("existing")
        result = unique_path(str(path))
        assert "_1" in result
        assert result.endswith(".json")

    def test_multiple_collisions(self, tmp_path):
        base = tmp_path / "file.json"
        base.write_text("1")
        (tmp_path / "file_1.json").write_text("2")
        result = unique_path(str(base))
        assert "_2" in result

    def test_no_extension(self, tmp_path):
        path = tmp_path / "file"
        path.write_text("existing")
        result = unique_path(str(path))
        assert "_1" in result

    def test_multiple_dots(self, tmp_path):
        path = tmp_path / "file.name.json"
        path.write_text("existing")
        result = unique_path(str(path))
        assert result.endswith("_1.json")


class TestReportSkipped:
    def test_empty(self, capsys):
        report_skipped([])
        assert capsys.readouterr().out == ""

    def test_non_empty(self, capsys):
        report_skipped([("file.json", "some error")])
        out = capsys.readouterr().out
        assert "Skipped 1" in out
        assert "file.json" in out
        assert "some error" in out

    def test_multiple(self, capsys):
        report_skipped([("a.json", "err1"), ("b.json", "err2")])
        out = capsys.readouterr().out
        assert "Skipped 2" in out
        assert "a.json" in out
        assert "b.json" in out
