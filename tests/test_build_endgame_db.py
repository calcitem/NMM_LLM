"""tests/test_build_endgame_db.py — Regression tests for the generalized endgame solver.

These tests build actual WDL tables and are SLOW (~3-5 minutes for the 3v3 case).
Run them separately:
    python -m pytest tests/test_build_endgame_db.py -v

The fast unit tests for EndgameSolvedDB are in test_endgame_solved_db.py.
"""

from __future__ import annotations

import importlib.util as _ilu
import os
import sys
import types
import unittest
from math import comb
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Load builder module
_builder_spec = _ilu.spec_from_file_location(
    "build_endgame_db", str(_ROOT / "tools" / "build_endgame_db.py")
)
_builder = _ilu.module_from_spec(_builder_spec)
_builder_spec.loader.exec_module(_builder)

solve_table = _builder.solve_table
_table_size = _builder._table_size
_encode = _builder._encode
_decode = _builder._decode

# Load endgame_solved_db for helpers
_ai_pkg = types.ModuleType("ai")
_ai_pkg.__path__ = [str(_ROOT / "ai")]
sys.modules.setdefault("ai", _ai_pkg)
_esdb_spec = _ilu.spec_from_file_location(
    "ai.endgame_solved_db", str(_ROOT / "ai" / "endgame_solved_db.py")
)
_esdb = _ilu.module_from_spec(_esdb_spec)
sys.modules["ai.endgame_solved_db"] = _esdb
_esdb_spec.loader.exec_module(_esdb)

get_wdl = _esdb.get_wdl
WDL_WIN = _esdb.WDL_WIN
WDL_LOSS = _esdb.WDL_LOSS
WDL_DRAW = _esdb.WDL_DRAW
WDL_UNKNOWN = _esdb.WDL_UNKNOWN
TABLE_SIZE_3_3 = _esdb.TABLE_SIZE_3_3


class TestSolverEncodeDecodeRoundtrip(unittest.TestCase):
    """Fast: verify general encode/decode roundtrip for several (nW,nB) sizes."""

    def _check(self, nW, nB, n_samples=200):
        import random
        rng = random.Random(nW * 100 + nB)
        all_sq = list(range(24))
        nC_b = comb(24 - nW, nB)
        for _ in range(n_samples):
            w = sorted(rng.sample(all_sq, nW))
            remaining = [i for i in all_sq if i not in w]
            b = sorted(rng.sample(remaining, nB))
            for turn_bit in (0, 1):
                pos_id = _encode(w, b, turn_bit, nC_b)
                self.assertGreaterEqual(pos_id, 0)
                self.assertLess(pos_id, _table_size(nW, nB))
                w2, b2, tb2 = _decode(pos_id, nW, nB, nC_b)
                self.assertEqual(w2, w, f"W mismatch for ({nW},{nB}): {w} → {w2}")
                self.assertEqual(b2, b, f"B mismatch for ({nW},{nB}): {b} → {b2}")
                self.assertEqual(tb2, turn_bit)

    def test_3v3(self): self._check(3, 3)
    def test_4v3(self): self._check(4, 3)
    def test_3v4(self): self._check(3, 4)
    def test_4v4(self): self._check(4, 4)
    def test_5v3(self): self._check(5, 3)
    def test_5v4(self): self._check(5, 4)
    def test_6v3(self): self._check(6, 3)


class TestSolver3v3Properties(unittest.TestCase):
    """Slow: build 3v3 table and check structural properties (~3-5 min)."""

    @classmethod
    def setUpClass(cls):
        cls.table = solve_table(3, 3, {}, verbose=False)

    def test_table_length(self):
        expected_bytes = (TABLE_SIZE_3_3 + 3) >> 2
        self.assertEqual(len(self.table), expected_bytes)

    def test_no_unknown_positions(self):
        for pos_id in range(TABLE_SIZE_3_3):
            self.assertNotEqual(
                get_wdl(self.table, pos_id), WDL_UNKNOWN,
                f"Position {pos_id} is still UNKNOWN after solve",
            )

    def test_win_count_nonzero(self):
        n_win = sum(1 for i in range(TABLE_SIZE_3_3) if get_wdl(self.table, i) == WDL_WIN)
        self.assertGreater(n_win, 0)

    def test_loss_count_nonzero(self):
        n_loss = sum(1 for i in range(TABLE_SIZE_3_3) if get_wdl(self.table, i) == WDL_LOSS)
        self.assertGreater(n_loss, 0)

    def test_win_equals_loss_by_symmetry(self):
        # In a symmetric game, by color-swap symmetry the number of positions
        # where W-to-move wins should equal B-to-move wins.  Since the table
        # stores WDL from STM's perspective, WIN count == LOSS count overall.
        n_win = sum(1 for i in range(TABLE_SIZE_3_3) if get_wdl(self.table, i) == WDL_WIN)
        n_loss = sum(1 for i in range(TABLE_SIZE_3_3) if get_wdl(self.table, i) == WDL_LOSS)
        self.assertEqual(n_win, n_loss, "WIN and LOSS counts should be equal by NMM symmetry")

    def test_totals_sum_to_table_size(self):
        counts = {WDL_WIN: 0, WDL_LOSS: 0, WDL_DRAW: 0, WDL_UNKNOWN: 0}
        for i in range(TABLE_SIZE_3_3):
            counts[get_wdl(self.table, i)] += 1
        self.assertEqual(counts[WDL_UNKNOWN], 0)
        self.assertEqual(
            counts[WDL_WIN] + counts[WDL_LOSS] + counts[WDL_DRAW], TABLE_SIZE_3_3
        )


if __name__ == "__main__":
    unittest.main()
