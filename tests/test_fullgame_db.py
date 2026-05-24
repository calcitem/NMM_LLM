"""tests/test_fullgame_db.py — Sanity tests for the full-game database.

These tests stay tiny and fast (well under a second).  A full DB build is
expensive (potentially many GB) and is never attempted here.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Build a minimal `ai` namespace package containing ONLY the two leaf modules
# we need (board_symmetry, fullgame_db).  This avoids triggering the real
# ai/__init__.py which depends on chromadb / ollama / fastapi.
_ROOT = Path(__file__).resolve().parent.parent
import importlib.util as _ilu

_ai_pkg = types.ModuleType("ai")
_ai_pkg.__path__ = [str(_ROOT / "ai")]
# Mark as a regular package: relative imports inside ai/fullgame_db.py will
# now resolve against this empty package, not the real ai/__init__.py.
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

from game.board import BoardState

# Load the builder as a script-style module.
_spec = _ilu.spec_from_file_location(
    "build_fullgame_db", str(_ROOT / "tools" / "build_fullgame_db.py"),
)
build_mod = _ilu.module_from_spec(_spec)
sys.modules["build_fullgame_db"] = build_mod
_spec.loader.exec_module(build_mod)


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
        # Two boards that are D4-equivalent must yield identical keys.
        b = BoardState.new_game()
        m1 = {"from": None, "to": "a7", "capture": None}
        m2 = {"from": None, "to": "g7", "capture": None}  # 90° rotation of a7
        b1 = b.apply_move(m1)
        b2 = b.apply_move(m2)
        self.assertEqual(build_mod.position_key(b1), build_mod.position_key(b2))


class TestBuilderTinyRun(unittest.TestCase):
    def _build_tiny(self, path: Path, cap: int = 200) -> None:
        builder = build_mod.FullGameDBBuilder(
            db_path=path, max_positions=cap, commit_every=50,
        )
        builder.enumerate_forward(BoardState.new_game())
        builder.backpropagate(passes=2)
        builder.close()

    def test_tiny_build_and_query(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "tiny.sqlite"
            self._build_tiny(db_path, cap=300)
            self.assertTrue(db_path.exists())

            db = FullGameDB(db_path)
            self.assertTrue(db.is_available())
            stats = db.stats()
            self.assertGreater(stats["positions"], 0)

            # The opening board must be present.
            result = db.query(BoardState.new_game())
            self.assertIsNotNone(result)
            # And it must have trajectories (24 legal placements, but
            # canonicalisation may merge symmetric edges — at least 3 unique
            # by D4 equivalence classes of single placements on empty board).
            self.assertGreaterEqual(len(result.trajectories), 1)
            db.close()

    def test_score_delta_shape(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "tiny.sqlite"
            self._build_tiny(db_path, cap=200)
            db = FullGameDB(db_path)
            hints = db.score_delta(BoardState.new_game(), "W")
            self.assertIsInstance(hints, dict)
            for k, v in hints.items():
                self.assertIsInstance(k, str)
                self.assertGreaterEqual(v, -0.5)
                self.assertLessEqual(v, 0.5)
            db.close()


class TestMissingDBFallback(unittest.TestCase):
    def test_missing_file_is_unavailable(self):
        db = FullGameDB("/nonexistent/path/fullgame.sqlite")
        self.assertFalse(db.is_available())
        self.assertIsNone(db.query(BoardState.new_game()))
        self.assertEqual(db.score_delta(BoardState.new_game(), "W"), {})


if __name__ == "__main__":
    unittest.main()
