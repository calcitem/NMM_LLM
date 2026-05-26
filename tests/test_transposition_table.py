"""
tests/test_transposition_table.py — Tests for Zobrist hashing and the TT.

Covers:
  - hash_board() consistency with incremental apply_move() updates
  - transposition equality: different move sequences reaching the same position
    must produce the same hash
  - capturing moves update all three relevant squares
  - placement-phase vs movement-phase distinction (pieces_placed >= 9 bit)
  - TT lookup / store / depth-preferred replacement
  - TT miss on hash collision
"""
from __future__ import annotations

import unittest

from game.board import BoardState, POSITIONS
from game.zobrist import hash_board, SQ_INDEX
from game.rules import get_all_legal_moves
from ai.transposition_table import TranspositionTable, EXACT, LOWER_BOUND, UPPER_BOUND


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _place(board: BoardState, color: str, sq: str) -> BoardState:
    move = {"from": None, "to": sq, "capture": None}
    return board.apply_move(move)


def _move(board: BoardState, from_sq: str, to_sq: str, capture: str | None = None) -> BoardState:
    move = {"from": from_sq, "to": to_sq, "capture": capture}
    return board.apply_move(move)


def _setup_move_phase() -> BoardState:
    """Build a simple 3v3 mid-game position in movement phase."""
    b = BoardState.new_game()
    # Place all 9 White pieces
    for sq in ["a7", "d7", "g7", "g4", "g1", "d1", "a1", "a4", "b6"]:
        b = _place(b, "W", sq)
    # Place all 9 Black pieces
    for sq in ["d6", "f6", "f4", "f2", "d2", "b2", "b4", "c5", "d5"]:
        b = _place(b, "B", sq)
    return b


# ---------------------------------------------------------------------------
# Zobrist hash consistency
# ---------------------------------------------------------------------------

class TestHashConsistency(unittest.TestCase):
    """hash_board() must agree with the incremental hash after every apply_move()."""

    def _verify_incremental(self, board: BoardState) -> None:
        fresh = hash_board(board)
        self.assertEqual(
            board.hash_key, fresh,
            f"Incremental hash {board.hash_key:#x} != fresh hash {fresh:#x}",
        )

    def test_new_game_hash_is_zero(self):
        b = BoardState.new_game()
        # All empty, W to move, neither side done: no keys XOR'd → 0
        self.assertEqual(b.hash_key, 0)
        self.assertEqual(hash_board(b), 0)

    def test_single_placement_consistent(self):
        b = BoardState.new_game()
        b2 = _place(b, "W", "d7")
        self._verify_incremental(b2)

    def test_placement_sequence_consistent(self):
        b = BoardState.new_game()
        for sq in ["a7", "d6", "g7", "f4", "d7", "b2"]:
            b = b.apply_move({"from": None, "to": sq, "capture": None})
            self._verify_incremental(b)

    def test_movement_consistent(self):
        b = _setup_move_phase()
        self._verify_incremental(b)
        b2 = _move(b, "a7", "a4")   # illegal here but hash is position-only
        # Use a legal move from actual move set
        moves = get_all_legal_moves(b)
        b2 = b.apply_move(moves[0])
        self._verify_incremental(b2)

    def test_capturing_move_consistent(self):
        """A capturing placement must XOR three squares: placed piece + removed piece."""
        b = BoardState.new_game()
        # Place W on a7, d7, g7 — closing outer-top mill → capture
        b = _place(b, "W", "a7")
        b = _place(b, "B", "d6")
        b = _place(b, "W", "d7")
        b = _place(b, "B", "f6")
        # Now W places on g7, closes a7-d7-g7, must capture a B piece
        cap_move = {"from": None, "to": "g7", "capture": "d6"}
        b2 = b.apply_move(cap_move)
        self._verify_incremental(b2)

    def test_side_to_move_changes_hash(self):
        b = BoardState.new_game()
        b_after = _place(b, "W", "d7")   # now B's turn
        self.assertNotEqual(b.hash_key, b_after.hash_key)
        # After B places, back to W — hash should differ from both
        b2 = _place(b_after, "B", "d6")
        self.assertNotEqual(b_after.hash_key, b2.hash_key)


# ---------------------------------------------------------------------------
# Transposition equality
# ---------------------------------------------------------------------------

class TestTranspositionEquality(unittest.TestCase):
    """Two different move sequences that reach the same board must hash identically."""

    def _reach_via_two_paths(self):
        """
        Path A: W→a7, B→d6, W→d7, B→f6
        Path B: W→d7, B→f6, W→a7, B→d6
        Both end at the same position with B to move.
        """
        b = BoardState.new_game()

        # Path A
        a = b
        a = _place(a, "W", "a7")
        a = _place(a, "B", "d6")
        a = _place(a, "W", "d7")
        a = _place(a, "B", "f6")

        # Path B
        c = b
        c = _place(c, "W", "d7")
        c = _place(c, "B", "f6")
        c = _place(c, "W", "a7")
        c = _place(c, "B", "d6")

        return a, c

    def test_transposition_same_hash(self):
        a, c = self._reach_via_two_paths()
        self.assertEqual(a.hash_key, c.hash_key)

    def test_transposition_same_positions(self):
        a, c = self._reach_via_two_paths()
        self.assertEqual(a.positions, c.positions)
        self.assertEqual(a.turn, c.turn)

    def test_move_phase_legal_moves_consistent(self):
        """After legal moves in movement phase, incremental hash == fresh hash."""
        b = _setup_move_phase()
        # Use only legal moves from the position.
        for move in get_all_legal_moves(b)[:3]:
            b2 = b.apply_move(move)
            self.assertEqual(
                b2.hash_key, hash_board(b2),
                f"Hash mismatch after legal move {move}",
            )
            # Apply a legal reply and verify again.
            for reply in get_all_legal_moves(b2)[:2]:
                b3 = b2.apply_move(reply)
                self.assertEqual(b3.hash_key, hash_board(b3))


# ---------------------------------------------------------------------------
# Placement-phase vs movement-phase distinction
# ---------------------------------------------------------------------------

class TestPhaseBit(unittest.TestCase):
    """The PLACED_DONE bit must distinguish placement phase from movement phase
    even when the piece layout is identical."""

    def test_placement_vs_movement_differ(self):
        # Build a board where W has placed 8 pieces and B has placed 8 pieces.
        # Then manually compare with the same layout but pieces_placed=9 (movement phase).
        b = BoardState.new_game()
        w_squares = ["a7", "d7", "g7", "g4", "g1", "d1", "a1", "a4"]  # 8
        b_squares = ["b6", "d6", "f6", "f4", "f2", "d2", "b2", "b4"]  # 8
        for sq in w_squares:
            b = _place(b, "W", sq)
        for sq in b_squares:
            b = _place(b, "B", sq)
        # b has pieces_placed=8 each — still placement phase

        # Create a board with same layout but pieces_placed=9 each (movement phase)
        b_move = BoardState.from_setup(b.positions, b.turn, phase="move")

        self.assertEqual(b.positions, b_move.positions)
        self.assertEqual(b.turn, b_move.turn)
        self.assertNotEqual(
            b.hash_key, b_move.hash_key,
            "Placement-phase and movement-phase boards with identical layout must hash differently",
        )

    def test_done_placing_bit_toggles_once(self):
        """Placing the 9th piece must toggle the PLACED_DONE bit exactly once."""
        b = BoardState.new_game()
        w_sqs = ["a7", "d7", "g7", "g4", "g1", "d1", "a1", "a4"]
        b_sqs = ["b6", "d6", "f6", "f4", "f2", "d2", "b2", "b4"]
        for sq in w_sqs:
            b = _place(b, "W", sq)
        for sq in b_sqs:
            b = _place(b, "B", sq)
        # W has placed 8, B has placed 8.  Next W places 9th.
        b9 = _place(b, "W", "c5")   # W's 9th placement
        self.assertEqual(b9.pieces_placed["W"], 9)
        self.assertEqual(b9.hash_key, hash_board(b9))

        # Placing again (hypothetically) should not toggle again; verify hash consistency.
        b10 = _place(b9, "B", "d5")  # B's 9th placement
        self.assertEqual(b10.pieces_placed["B"], 9)
        self.assertEqual(b10.hash_key, hash_board(b10))


# ---------------------------------------------------------------------------
# TranspositionTable
# ---------------------------------------------------------------------------

class TestTranspositionTable(unittest.TestCase):

    def setUp(self):
        self.tt = TranspositionTable()

    def test_empty_lookup_returns_none(self):
        self.assertIsNone(self.tt.lookup(12345))

    def test_store_and_lookup_exact(self):
        self.tt.store(0xDEADBEEF, depth=4, score=500, flag=EXACT, from_sq="a7", to_sq="d7")
        entry = self.tt.lookup(0xDEADBEEF)
        self.assertIsNotNone(entry)
        depth, score, flag, from_sq, to_sq = entry
        self.assertEqual(depth, 4)
        self.assertEqual(score, 500)
        self.assertEqual(flag, EXACT)
        self.assertEqual(from_sq, "a7")
        self.assertEqual(to_sq, "d7")

    def test_store_placement_move(self):
        """Placement moves have from_sq=None."""
        self.tt.store(0xABCD, depth=3, score=-200, flag=LOWER_BOUND, from_sq=None, to_sq="d6")
        entry = self.tt.lookup(0xABCD)
        self.assertIsNotNone(entry)
        _, _, _, from_sq, to_sq = entry
        self.assertIsNone(from_sq)
        self.assertEqual(to_sq, "d6")

    def test_depth_preferred_replacement_deeper_wins(self):
        """A deeper entry should replace a shallower one."""
        self.tt.store(0x1234, depth=3, score=100, flag=EXACT, from_sq=None, to_sq="a7")
        self.tt.store(0x1234, depth=5, score=200, flag=EXACT, from_sq=None, to_sq="d7")
        entry = self.tt.lookup(0x1234)
        self.assertEqual(entry[0], 5)
        self.assertEqual(entry[1], 200)

    def test_depth_preferred_shallower_does_not_overwrite(self):
        """A shallower entry must NOT replace a deeper one."""
        self.tt.store(0x5678, depth=6, score=300, flag=EXACT, from_sq=None, to_sq="g7")
        self.tt.store(0x5678, depth=2, score=999, flag=EXACT, from_sq=None, to_sq="g1")
        entry = self.tt.lookup(0x5678)
        self.assertEqual(entry[0], 6)   # depth unchanged
        self.assertEqual(entry[1], 300) # score unchanged

    def test_same_depth_overwrites(self):
        """Equal depth counts as 'at least as deep' — new entry replaces old."""
        self.tt.store(0xAAAA, depth=4, score=10, flag=EXACT, from_sq=None, to_sq="a7")
        self.tt.store(0xAAAA, depth=4, score=20, flag=UPPER_BOUND, from_sq="b6", to_sq="d6")
        entry = self.tt.lookup(0xAAAA)
        self.assertEqual(entry[1], 20)
        self.assertEqual(entry[2], UPPER_BOUND)

    def test_collision_returns_none(self):
        """Different hash stored in same slot must not be returned."""
        from ai.transposition_table import _TABLE_SIZE
        key_a = 42
        key_b = 42 + _TABLE_SIZE  # same slot index, different key
        self.tt.store(key_a, depth=3, score=50, flag=EXACT, from_sq=None, to_sq="d1")
        # key_b maps to the same slot but is a different hash
        self.assertIsNone(self.tt.lookup(key_b))

    def test_clear_resets_all_slots(self):
        self.tt.store(0xFFFF, depth=5, score=1000, flag=EXACT, from_sq=None, to_sq="a4")
        self.tt.clear()
        self.assertIsNone(self.tt.lookup(0xFFFF))

    def test_flags_lower_bound(self):
        self.tt.store(0xBEEF, depth=2, score=-100, flag=LOWER_BOUND, from_sq="c5", to_sq="d5")
        _, _, flag, _, _ = self.tt.lookup(0xBEEF)
        self.assertEqual(flag, LOWER_BOUND)

    def test_flags_upper_bound(self):
        self.tt.store(0xCAFE, depth=2, score=-100, flag=UPPER_BOUND, from_sq="c5", to_sq="d5")
        _, _, flag, _, _ = self.tt.lookup(0xCAFE)
        self.assertEqual(flag, UPPER_BOUND)


# ---------------------------------------------------------------------------
# Integration: hash through a full game sequence
# ---------------------------------------------------------------------------

class TestHashThroughGame(unittest.TestCase):
    """Play a short sequence of legal moves and verify hash consistency at each step."""

    def test_first_eight_plies(self):
        b = BoardState.new_game()
        legal_placements = ["a7", "d6", "g7", "f4", "d7", "b2", "g4", "d2"]
        for sq in legal_placements:
            b = b.apply_move({"from": None, "to": sq, "capture": None})
            fresh = hash_board(b)
            self.assertEqual(
                b.hash_key, fresh,
                f"Hash mismatch after placing on {sq}: "
                f"incremental={b.hash_key:#x} fresh={fresh:#x}",
            )

    def test_hash_unique_across_placements(self):
        """Each successive placement must produce a distinct hash."""
        b = BoardState.new_game()
        seen = {b.hash_key}
        placements = ["a7", "d6", "g7", "f4", "d7", "b2", "g4", "d2"]
        for sq in placements:
            b = b.apply_move({"from": None, "to": sq, "capture": None})
            self.assertNotIn(b.hash_key, seen, f"Hash collision after placing on {sq}")
            seen.add(b.hash_key)


if __name__ == "__main__":
    unittest.main()
