"""Tests for learned_ai/sentinel/db_teacher.py (graceful external DB stub)."""

from __future__ import annotations

from game.board import BoardState
from learned_ai.sentinel.config import SentinelConfig
from learned_ai.sentinel.db_teacher import ExternalSolvedDB, open_external_db


def _board():
    return BoardState.from_fen_string("BBW....B.W.W............|W|3|3")


def test_unavailable_returns_none_empty_path():
    db = ExternalSolvedDB(db_path="", enabled=True)
    assert db.is_available() is False
    assert db.query_state(_board()) is None
    assert db.query(_board()) is None
    assert db.query_move_quality(_board(), {"from": None, "to": "a4"}) is None


def test_is_available_false_when_no_path():
    db = ExternalSolvedDB("")
    assert db.is_available() is False


def test_no_crash_on_bad_path():
    # Must not raise on init even for a clearly nonexistent path.
    db = ExternalSolvedDB("/nonexistent/path/to/db")
    assert db.is_available() is False
    assert db.query_state(_board()) is None


def test_query_trajectory_length_matches_input():
    db = ExternalSolvedDB("")
    states = [_board(), _board(), _board()]
    result = db.query_trajectory(states)
    assert result == [None, None, None]
    assert len(result) == len(states)


def test_disabled_forces_unavailable(tmp_path):
    # Even if files exist, enabled=False keeps it unavailable.
    (tmp_path / "database.dat").write_bytes(b"\x00" * 16)
    (tmp_path / "preCalculatedVars.dat").write_bytes(b"\x01" * 16)
    db = ExternalSolvedDB(str(tmp_path), enabled=False)
    assert db.is_available() is False
    assert db.query_state(_board()) is None


def test_probe_records_format_metadata(tmp_path):
    # When files exist, the adapter records probe metadata for future decoding,
    # but still returns None (format undecoded) and never crashes.
    (tmp_path / "database.dat").write_bytes(b"\x00" * 1024)
    (tmp_path / "preCalculatedVars.dat").write_bytes(b"ABCD" * 8)
    db = ExternalSolvedDB(str(tmp_path), enabled=True)
    assert db.is_available() is False
    assert "vars_size_bytes" in db.format_probe
    assert db.format_probe["vars_size_bytes"] == 32
    assert db.format_probe.get("database_size_bytes") == 1024
    assert db.query_state(_board()) is None


def test_open_external_db_from_config():
    cfg = SentinelConfig(external_db_path="", external_db_enabled=False)
    db = open_external_db(cfg)
    assert isinstance(db, ExternalSolvedDB)
    assert db.is_available() is False


def test_close_is_noop():
    db = ExternalSolvedDB("")
    db.close()  # must not raise
