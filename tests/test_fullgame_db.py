"""tests/test_fullgame_db.py — Sanity tests for the full-game database."""

from __future__ import annotations

import json as _json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_ROOT = Path(__file__).resolve().parent.parent
import importlib.util as _ilu

_ai_pkg = types.ModuleType("ai")
_ai_pkg.__path__ = [str(_ROOT / "ai")]
sys.modules["ai"] = _ai_pkg

def _load_leaf(name: str, file: Path):
    spec = _ilu.spec_from_file_location(f"ai.{name}", str(file))
    mod = _ilu.module_from_spec(spec)
    sys.modules[f"ai.{name}"] = mod
    spec.loader.exec_module(mod)
    setattr(_ai_pkg, name, mod)
    return mod

_load_leaf("board_symmetry", _ROOT / "ai" / "board_symmetry.py")
_fgdb = _load_leaf("fullgame_db", _ROOT / "ai" / "fullgame_db.py")
FullGameDB = _fgdb.FullGameDB
FullGameResult = _fgdb.FullGameResult
_pack_move = _fgdb._pack_move
_unpack_move = _fgdb._unpack_move
_EMPTY_MOVE = _fgdb._EMPTY_MOVE

from game.board import BoardState

_spec = _ilu.spec_from_file_location(
    "build_fullgame_db", str(_ROOT / "tools" / "build_fullgame_db.py"),
)
build_mod = _ilu.module_from_spec(_spec)
sys.modules["build_fullgame_db"] = build_mod
_spec.loader.exec_module(build_mod)


# ── Shared test helper ────────────────────────────────────────────────────────

def _build_tiny_bin(bin_path: Path, min_seed_frequency: int = 1, expand_depth: int = 2) -> None:
    """Build a tiny binary DB from synthetic game data for testing."""
    games_dir = bin_path.parent / "test_games"
    games_dir.mkdir(exist_ok=True)
    games = [
        {"moves": [{"to": p} for p in ["a7","d6","a4","d3","a1","d1","b6","b4","g7","g4"]], "human_color": "W"},
        {"moves": [{"to": p} for p in ["a7","g4","a4","g1","g7","d6","b6","d3","b4","f4"]], "human_color": "W"},
        {"moves": [{"to": p} for p in ["a7","d6","g7","d3","b6","f6","a4","g4","a1","g1"]], "human_color": "W"},
    ]
    with open(games_dir / "test.jsonl", "w") as f:
        for g in games:
            f.write(_json.dumps(g) + "\n")
    builder = build_mod.ExpandFromGamesBuilder(
        min_seed_frequency=min_seed_frequency,
        expand_depth=expand_depth,
    )
    builder.build(games_dir)
    builder.write_binary(bin_path)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCanonicalEncoding(unittest.TestCase):
    def test_encode_roundtrip_distinguishes_positions(self):
        b = BoardState.new_game()
        key1 = build_mod.position_key(b)
        b2 = BoardState(
            positions=dict(b.positions),
            turn="B",
            pieces_on_board=dict(b.pieces_on_board),
            pieces_placed=dict(b.pieces_placed),
            pieces_captured=dict(b.pieces_captured),
        )
        key2 = build_mod.position_key(b2)
        self.assertNotEqual(key1, key2)

    def test_d4_equivalence_keys_match(self):
        b = BoardState.new_game()
        m1 = {"from": None, "to": "a7", "capture": None}
        m2 = {"from": None, "to": "g7", "capture": None}
        b1 = b.apply_move(m1)
        b2 = b.apply_move(m2)
        self.assertEqual(build_mod.position_key(b1), build_mod.position_key(b2))


class TestBuilderFromGames(unittest.TestCase):
    def test_build_produces_queryable_binary(self):
        with tempfile.TemporaryDirectory() as td:
            bin_path = Path(td) / "tiny.bin"
            _build_tiny_bin(bin_path)
            self.assertTrue(bin_path.exists())
            db = FullGameDB(bin_path)
            self.assertTrue(db.is_available())
            stats = db.stats()
            self.assertGreater(stats["positions"], 0)
            db.close()

    def test_opening_position_present(self):
        with tempfile.TemporaryDirectory() as td:
            bin_path = Path(td) / "tiny.bin"
            _build_tiny_bin(bin_path)
            db = FullGameDB(bin_path)
            result = db.query(BoardState.new_game())
            self.assertIsNotNone(result)
            db.close()

    def test_score_delta_shape(self):
        with tempfile.TemporaryDirectory() as td:
            bin_path = Path(td) / "tiny.bin"
            _build_tiny_bin(bin_path)
            db = FullGameDB(bin_path)
            hints = db.score_delta(BoardState.new_game(), "W")
            self.assertIsInstance(hints, dict)
            for k, v in hints.items():
                self.assertIsInstance(k, str)
                self.assertGreaterEqual(v, -0.5)
                self.assertLessEqual(v, 0.5)
            db.close()

    def test_frequency_tracked(self):
        with tempfile.TemporaryDirectory() as td:
            bin_path = Path(td) / "tiny.bin"
            _build_tiny_bin(bin_path, min_seed_frequency=1)
            db = FullGameDB(bin_path)
            result = db.query(BoardState.new_game())
            self.assertIsNotNone(result)
            # Opening position appears in all 3 synthetic games
            self.assertGreaterEqual(result.frequency, 3)
            db.close()

    def test_binary_file_size_correct(self):
        with tempfile.TemporaryDirectory() as td:
            bin_path = Path(td) / "tiny.bin"
            _build_tiny_bin(bin_path)
            from ai.fullgame_db import HEADER_SIZE, RECORD_SIZE
            db = FullGameDB(bin_path)
            n = db.stats()["positions"]
            db.close()
            self.assertEqual(bin_path.stat().st_size, HEADER_SIZE + n * RECORD_SIZE)


class TestMissingDBFallback(unittest.TestCase):
    def test_missing_file_is_unavailable(self):
        db = FullGameDB("/nonexistent/path/fullgame.bin")
        self.assertFalse(db.is_available())
        self.assertIsNone(db.query(BoardState.new_game()))
        self.assertEqual(db.score_delta(BoardState.new_game(), "W"), {})


class TestBinaryPackingHelpers(unittest.TestCase):
    def test_struct_record_size(self):
        import struct
        self.assertEqual(struct.calcsize("<9sBHIIIIII"), 36)

    def test_empty_move_sentinel(self):
        self.assertEqual(_pack_move(None), _EMPTY_MOVE)
        notation, flag = _unpack_move(_EMPTY_MOVE)
        self.assertIsNone(notation)
        self.assertEqual(flag, "N")

    def test_placement_roundtrip(self):
        packed = _pack_move("a4", "W")
        notation, flag = _unpack_move(packed)
        self.assertEqual(notation, "a4")
        self.assertEqual(flag, "W")

    def test_movement_roundtrip(self):
        packed = _pack_move("a7-a4", "L")
        notation, flag = _unpack_move(packed)
        self.assertEqual(notation, "a7-a4")
        self.assertEqual(flag, "L")

    def test_capture_roundtrip(self):
        packed = _pack_move("a7-a4xb4", "N")
        notation, flag = _unpack_move(packed)
        self.assertEqual(notation, "a7-a4xb4")
        self.assertEqual(flag, "N")

    def test_placement_capture_roundtrip(self):
        packed = _pack_move("d2xa4", "W")
        notation, flag = _unpack_move(packed)
        self.assertEqual(notation, "d2xa4")
        self.assertEqual(flag, "W")

    def test_all_positions_roundtrip(self):
        from game.board import POSITIONS
        for pos in POSITIONS:
            packed = _pack_move(pos)
            notation, _ = _unpack_move(packed)
            self.assertEqual(notation, pos, f"Roundtrip failed for placement {pos!r}")


if __name__ == "__main__":
    unittest.main()
