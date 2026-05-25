"""tests/test_endgame_solved_db.py — Tests for combinatorial helpers and
EndgameSolvedDB in ai/endgame_solved_db.py.

Kept fast: no solver invocation, no WDL file build.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from itertools import combinations
from math import comb
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_ROOT = Path(__file__).resolve().parent.parent
import importlib.util as _ilu

_ai_pkg = types.ModuleType("ai")
_ai_pkg.__path__ = [str(_ROOT / "ai")]
sys.modules.setdefault("ai", _ai_pkg)

def _load_leaf(name: str, file: Path):
    spec = _ilu.spec_from_file_location(f"ai.{name}", str(file))
    mod = _ilu.module_from_spec(spec)
    sys.modules[f"ai.{name}"] = mod
    spec.loader.exec_module(mod)
    setattr(_ai_pkg, name, mod)
    return mod

_esdb = _load_leaf("endgame_solved_db", _ROOT / "ai" / "endgame_solved_db.py")

combo_rank = _esdb.combo_rank
combo_unrank = _esdb.combo_unrank
encode_position_id = _esdb.encode_position_id
decode_position_id = _esdb.decode_position_id
TABLE_SIZE_3_3 = _esdb.TABLE_SIZE_3_3
get_wdl = _esdb.get_wdl
set_wdl = _esdb.set_wdl
WDL_WIN = _esdb.WDL_WIN
WDL_LOSS = _esdb.WDL_LOSS
WDL_DRAW = _esdb.WDL_DRAW
WDL_UNKNOWN = _esdb.WDL_UNKNOWN
EndgameSolvedDB = _esdb.EndgameSolvedDB

from game.board import BoardState, POSITIONS


class TestComboRankConstants(unittest.TestCase):
    def test_table_size(self):
        self.assertEqual(TABLE_SIZE_3_3, 5_383_840)

    def test_c24_3(self):
        self.assertEqual(comb(24, 3), 2024)

    def test_c21_3(self):
        self.assertEqual(comb(21, 3), 1330)

    def test_rank_min(self):
        self.assertEqual(combo_rank([0, 1, 2]), 0)

    def test_rank_max_24_3(self):
        # C(21,1)+C(22,2)+C(23,3) = 21+231+1771 = 2023 = C(24,3)-1
        self.assertEqual(combo_rank([21, 22, 23]), 2023)

    def test_rank_max_21_3(self):
        # C(18,1)+C(19,2)+C(20,3) = 18+171+1140 = 1329 = C(21,3)-1
        self.assertEqual(combo_rank([18, 19, 20]), 1329)

    def test_rank_empty(self):
        self.assertEqual(combo_rank([]), 0)

    def test_rank_singleton(self):
        for i in range(24):
            self.assertEqual(combo_rank([i]), i)


class TestComboRankUnrankRoundtrip(unittest.TestCase):
    def test_all_white_arrangements_24_3(self):
        """All C(24,3)=2024 white arrangements: unique ranks in [0,2024), roundtrip ok."""
        seen: set[int] = set()
        for indices in combinations(range(24), 3):
            indices = list(indices)
            rank = combo_rank(indices)
            self.assertNotIn(rank, seen, f"Duplicate rank {rank} for {indices}")
            self.assertGreaterEqual(rank, 0)
            self.assertLess(rank, comb(24, 3))
            seen.add(rank)
            self.assertEqual(combo_unrank(rank, 3, 24), indices)
        self.assertEqual(len(seen), comb(24, 3))

    def test_all_black_arrangements_21_3(self):
        """All C(21,3)=1330 black arrangements: unique ranks in [0,1330), roundtrip ok."""
        seen: set[int] = set()
        for indices in combinations(range(21), 3):
            indices = list(indices)
            rank = combo_rank(indices)
            self.assertNotIn(rank, seen, f"Duplicate rank {rank} for {indices}")
            self.assertGreaterEqual(rank, 0)
            self.assertLess(rank, comb(21, 3))
            seen.add(rank)
            self.assertEqual(combo_unrank(rank, 3, 21), indices)
        self.assertEqual(len(seen), comb(21, 3))

    def test_unrank_edge_cases(self):
        self.assertEqual(combo_unrank(0, 3, 24), [0, 1, 2])
        self.assertEqual(combo_unrank(2023, 3, 24), [21, 22, 23])
        self.assertEqual(combo_unrank(0, 3, 21), [0, 1, 2])
        self.assertEqual(combo_unrank(1329, 3, 21), [18, 19, 20])
        self.assertEqual(combo_unrank(0, 1, 24), [0])
        self.assertEqual(combo_unrank(23, 1, 24), [23])


class TestPositionIDRoundtrip(unittest.TestCase):
    def _roundtrip(self, white_pieces, black_pieces, turn):
        pos_id = encode_position_id(white_pieces, black_pieces, turn)
        w2, b2, t2 = decode_position_id(pos_id, len(white_pieces), len(black_pieces))
        self.assertEqual(sorted(w2), sorted(white_pieces), f"White mismatch for {white_pieces}")
        self.assertEqual(sorted(b2), sorted(black_pieces), f"Black mismatch for {black_pieces}")
        self.assertEqual(t2, turn)
        return pos_id

    def test_first_position(self):
        # First three white squares, next three for black
        pos_id = self._roundtrip(["a7", "d7", "g7"], ["g4", "g1", "d1"], "W")
        self.assertEqual(pos_id, 0)

    def test_last_position_turn_b(self):
        # Last valid position (highest indices): white=[21,22,23], black picks first 3 remaining
        # Indices 21,22,23 → d3,c3,c4
        # Remaining [0..20], black=[18,19,20] → d5,e5,e4 (indices 17,18,19)
        # Remapped black=[18,19,20] → rank=1329
        # white_rank=2023, black_rank=1329, turn=B
        # pos_id = 2023*1330*2 + 1329*2 + 1
        expected = 2023 * 1330 * 2 + 1329 * 2 + 1
        self.assertEqual(expected, TABLE_SIZE_3_3 - 1)
        w = [POSITIONS[21], POSITIONS[22], POSITIONS[23]]  # d3,c3,c4
        # remaining after [21,22,23] = indices [0..20] → positions 0..20
        # combo_unrank(1329,3,21) = [18,19,20] → POSITIONS[18],POSITIONS[19],POSITIONS[20]
        # That is e5, e4, e3 (indices 18,19,20)
        b = [POSITIONS[18], POSITIONS[19], POSITIONS[20]]
        pos_id = self._roundtrip(w, b, "B")
        self.assertEqual(pos_id, expected)

    def test_pos_id_range(self):
        # All pos_ids must be in [0, TABLE_SIZE_3_3)
        import random
        rng = random.Random(42)
        all_pos = list(range(24))
        for _ in range(500):
            w_idx = sorted(rng.sample(all_pos, 3))
            remaining_idx = [i for i in all_pos if i not in w_idx]
            b_idx = sorted(rng.sample(remaining_idx, 3))
            turn = rng.choice(["W", "B"])
            w = [POSITIONS[i] for i in w_idx]
            b = [POSITIONS[i] for i in b_idx]
            pos_id = encode_position_id(w, b, turn)
            self.assertGreaterEqual(pos_id, 0)
            self.assertLess(pos_id, TABLE_SIZE_3_3)

    def test_roundtrip_random_sample(self):
        import random
        rng = random.Random(99)
        all_pos = list(range(24))
        for _ in range(500):
            w_idx = sorted(rng.sample(all_pos, 3))
            remaining_idx = [i for i in all_pos if i not in w_idx]
            b_idx = sorted(rng.sample(remaining_idx, 3))
            turn = rng.choice(["W", "B"])
            w = [POSITIONS[i] for i in w_idx]
            b = [POSITIONS[i] for i in b_idx]
            self._roundtrip(w, b, turn)

    def test_turn_bit_differs(self):
        w = ["a7", "d7", "g7"]
        b = ["g4", "g1", "d1"]
        pid_w = encode_position_id(w, b, "W")
        pid_b = encode_position_id(w, b, "B")
        self.assertEqual(pid_b - pid_w, 1)

    def test_distinct_positions_have_distinct_ids(self):
        w1 = ["a7", "d7", "g7"]
        b1 = ["g4", "g1", "d1"]
        w2 = ["a7", "d7", "g7"]
        b2 = ["g4", "g1", "a1"]
        pid1 = encode_position_id(w1, b1, "W")
        pid2 = encode_position_id(w2, b2, "W")
        self.assertNotEqual(pid1, pid2)


class TestWdlPacking(unittest.TestCase):
    def test_set_and_get_all_values(self):
        table = bytearray(4)  # enough for 16 positions
        for pos_id in range(16):
            for val in (WDL_UNKNOWN, WDL_WIN, WDL_LOSS, WDL_DRAW):
                set_wdl(table, pos_id, val)
                self.assertEqual(get_wdl(table, pos_id), val)

    def test_adjacent_slots_independent(self):
        table = bytearray(4)
        set_wdl(table, 0, WDL_WIN)
        set_wdl(table, 1, WDL_LOSS)
        set_wdl(table, 2, WDL_DRAW)
        set_wdl(table, 3, WDL_WIN)
        self.assertEqual(get_wdl(table, 0), WDL_WIN)
        self.assertEqual(get_wdl(table, 1), WDL_LOSS)
        self.assertEqual(get_wdl(table, 2), WDL_DRAW)
        self.assertEqual(get_wdl(table, 3), WDL_WIN)

    def test_overwrite(self):
        table = bytearray(4)
        set_wdl(table, 5, WDL_WIN)
        set_wdl(table, 5, WDL_LOSS)
        self.assertEqual(get_wdl(table, 5), WDL_LOSS)

    def test_packed_byte_layout(self):
        # pos_id 0-3 are packed into byte 0, two bits each
        table = bytearray(1)
        set_wdl(table, 0, 1)  # bits 1:0 = 01
        set_wdl(table, 1, 2)  # bits 3:2 = 10
        set_wdl(table, 2, 3)  # bits 5:4 = 11
        set_wdl(table, 3, 0)  # bits 7:6 = 00
        # byte 0 = 0b00_11_10_01 = 0x39 = 57
        self.assertEqual(table[0], 0b00111001)


class TestEndgameSolvedDBMissing(unittest.TestCase):
    def test_none_dir_unavailable(self):
        db = EndgameSolvedDB(None)
        self.assertFalse(db.is_available())

    def test_missing_dir_unavailable(self):
        db = EndgameSolvedDB("/nonexistent/endgame_dir")
        self.assertFalse(db.is_available())

    def test_query_returns_none_when_unavailable(self):
        db = EndgameSolvedDB(None)
        board = BoardState.new_game()
        self.assertIsNone(db.query(board))

    def test_wrong_size_file_unavailable(self):
        with tempfile.TemporaryDirectory() as td:
            wdl = Path(td) / "endgame_3_3.wdl"
            wdl.write_bytes(b"\x00" * 100)  # wrong size
            db = EndgameSolvedDB(td)
            self.assertFalse(db.is_available())


_WDL_BYTES_3_3 = _esdb._WDL_BYTES_3_3


def _make_wdl_table(**kwargs) -> bytes:
    """Build a zeroed WDL table with specific entries set.

    kwargs: pos_id=wdl_val pairs (or pass a dict as the only kwarg).
    Returns bytes of length _WDL_BYTES_3_3.
    """
    table = bytearray(_WDL_BYTES_3_3)
    for pos_id, val in kwargs.items():
        set_wdl(table, int(pos_id), val)
    return bytes(table)


class TestEndgameSolvedDBWithFakeTable(unittest.TestCase):
    def _make_board(self, w, b, turn, w_placed=9, b_placed=9):
        return BoardState(
            positions={p: "W" for p in w} | {p: "B" for p in b},
            turn=turn,
            pieces_on_board={"W": len(w), "B": len(b)},
            pieces_placed={"W": w_placed, "B": b_placed},
            pieces_captured={"W": 0, "B": 9 - len(w)},
        )

    def test_win_lookup(self):
        w = ["a7", "d7", "g7"]
        b = ["g4", "g1", "d1"]
        pos_id = encode_position_id(w, b, "W")
        table = bytearray(_WDL_BYTES_3_3)
        set_wdl(table, pos_id, WDL_WIN)
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "endgame_3_3.wdl").write_bytes(bytes(table))
            db = EndgameSolvedDB(td)
            self.assertTrue(db.is_available())
            result = db.query(self._make_board(w, b, "W"))
            self.assertEqual(result, "W")
            db.close()

    def test_loss_and_draw_lookup(self):
        w = ["a7", "d7", "g7"]
        b = ["g4", "g1", "d1"]
        for wdl_val, expected in [(WDL_LOSS, "L"), (WDL_DRAW, "D")]:
            pos_id = encode_position_id(w, b, "W")
            table = bytearray(_WDL_BYTES_3_3)
            set_wdl(table, pos_id, wdl_val)
            with tempfile.TemporaryDirectory() as td:
                (Path(td) / "endgame_3_3.wdl").write_bytes(bytes(table))
                db = EndgameSolvedDB(td)
                self.assertEqual(db.query(self._make_board(w, b, "W")), expected)
                db.close()

    def test_unknown_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "endgame_3_3.wdl").write_bytes(bytes(_WDL_BYTES_3_3))
            db = EndgameSolvedDB(td)
            self.assertTrue(db.is_available())
            board = self._make_board(["a7", "d7", "g7"], ["g4", "g1", "d1"], "W")
            self.assertIsNone(db.query(board))
            db.close()

    def test_query_non_fly_phase_returns_none(self):
        w = ["a7", "d7", "g7"]
        b = ["g4", "g1", "d1"]
        pos_id = encode_position_id(w, b, "W")
        table = bytearray(_WDL_BYTES_3_3)
        set_wdl(table, pos_id, WDL_WIN)
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "endgame_3_3.wdl").write_bytes(bytes(table))
            db = EndgameSolvedDB(td)
            board = self._make_board(w, b, "W", w_placed=8, b_placed=9)
            self.assertIsNone(db.query(board))
            db.close()

    def test_query_wrong_piece_count_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "endgame_3_3.wdl").write_bytes(bytes(_WDL_BYTES_3_3))
            db = EndgameSolvedDB(td)
            board = BoardState(
                positions={"a7": "W", "d7": "W", "g7": "W", "g4": "W",
                           "g1": "B", "d1": "B", "a1": "B"},
                turn="W",
                pieces_on_board={"W": 4, "B": 3},
                pieces_placed={"W": 9, "B": 9},
                pieces_captured={"W": 0, "B": 6},
            )
            self.assertIsNone(db.query(board))
            db.close()


if __name__ == "__main__":
    unittest.main()
