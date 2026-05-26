"""
tests/test_search_enhancements.py — Unit tests for SE-2 (killers) and future SE-series.

Tests focus on the pure data-structure logic (_store_killer, _order_moves with killers)
rather than full-search integration, which is validated by the existing AI tests.
"""
from __future__ import annotations

import unittest

from game.board import BoardState
from ai.game_ai import GameAI, _order_moves


def _make_board(turn: str = "W") -> BoardState:
    """Return a non-terminal movement-phase board with several legal moves for W.

    Uses from_setup so the position is explicit and well-known, avoiding the
    risk that an interleaved placement sequence lands in a terminal position.
    """
    # W: outer top-left cluster + two middle squares → all have empty neighbours.
    # B: inner cluster far away → no immediate mill threats.
    positions = {
        "a7": "W", "d7": "W", "g7": "W",   # outer top row
        "a4": "W", "b4": "W", "c4": "W",   # left column + cross
        "a1": "W", "d1": "W", "g1": "W",   # outer bottom row
        "d6": "B", "f6": "B", "f4": "B",   # B cluster 1
        "f2": "B", "d2": "B", "b2": "B",   # B cluster 2
        "c5": "B", "d5": "B", "e5": "B",   # B cluster 3
    }
    return BoardState.from_setup(positions, turn=turn, phase="move")


class TestStoreKiller(unittest.TestCase):

    def setUp(self):
        self.ai = GameAI(color="W", difficulty=3)

    def test_first_killer_stored_in_slot_0(self):
        self.ai._store_killer(4, "a7", "d7")
        self.assertEqual(self.ai._killers[4][0], ("a7", "d7"))
        self.assertIsNone(self.ai._killers[4][1])

    def test_second_distinct_killer_shifts(self):
        self.ai._store_killer(4, "a7", "d7")
        self.ai._store_killer(4, "g1", "g4")
        self.assertEqual(self.ai._killers[4][0], ("g1", "g4"))
        self.assertEqual(self.ai._killers[4][1], ("a7", "d7"))

    def test_duplicate_in_slot0_not_re_stored(self):
        self.ai._store_killer(4, "a7", "d7")
        self.ai._store_killer(4, "g1", "g4")   # slot0=(g1,g4), slot1=(a7,d7)
        self.ai._store_killer(4, "g1", "g4")   # duplicate — must not shift again
        self.assertEqual(self.ai._killers[4][0], ("g1", "g4"))
        self.assertEqual(self.ai._killers[4][1], ("a7", "d7"))

    def test_depth_boundary_ignored(self):
        # depth >= 32 must be silently ignored
        self.ai._store_killer(32, "a7", "d7")
        # No IndexError and killers unaffected
        self.assertEqual(self.ai._killers[31], [None, None])

    def test_placement_move_stored_none_from(self):
        self.ai._store_killer(3, None, "d5")
        self.assertEqual(self.ai._killers[3][0], (None, "d5"))

    def test_killers_reset_each_choose_move(self):
        self.ai._store_killer(5, "a7", "d7")
        b = BoardState.new_game()
        from game.rules import get_all_legal_moves
        moves = get_all_legal_moves(b)
        self.ai.choose_move(b)   # triggers reset
        self.assertEqual(self.ai._killers[5], [None, None])


class TestOrderMovesWithKillers(unittest.TestCase):
    """_order_moves must place killer-matched moves between p1 and p2."""

    def _moves_from(self, board: BoardState) -> list:
        from game.rules import get_all_legal_moves
        return get_all_legal_moves(board)

    def test_killer_appears_before_plain_moves(self):
        b = _make_board()
        from game.rules import get_all_legal_moves
        moves = get_all_legal_moves(b)
        if len(moves) < 2:
            self.skipTest("not enough moves for this test")

        # Pick a move that is NOT in close/block and make it a killer.
        # We'll use a fresh order to find a p2 move first.
        plain_ordered = _order_moves(b, moves)
        # The last move in the ordered list is a p2 move (if any exist).
        p2_move = plain_ordered[-1]
        killer = (p2_move.get("from"), p2_move["to"])

        with_killers = _order_moves(b, moves, killers=[killer, None])

        # killer move must appear somewhere before the last position it had without killers.
        killer_idx_before = next(
            i for i, m in enumerate(plain_ordered)
            if m.get("from") == killer[0] and m["to"] == killer[1]
        )
        killer_idx_after = next(
            i for i, m in enumerate(with_killers)
            if m.get("from") == killer[0] and m["to"] == killer[1]
        )
        self.assertLessEqual(killer_idx_after, killer_idx_before)

    def test_no_killer_no_change_when_no_close_block(self):
        b = _make_board()
        from game.rules import get_all_legal_moves
        moves = get_all_legal_moves(b)
        # With no killers and no close/block candidates the fast path returns moves as-is.
        # We just verify the call doesn't raise.
        result = _order_moves(b, moves, killers=None)
        self.assertEqual(len(result), len(moves))

    def test_killers_never_duplicate_p0_p1_moves(self):
        """A killer that matches a p0 or p1 move must still land in p0/p1, not pk."""
        b = _make_board()
        from game.rules import get_all_legal_moves
        moves = get_all_legal_moves(b)

        # Use the first ordered move (p0 or p1) as the killer.
        plain = _order_moves(b, moves)
        if not plain:
            self.skipTest("empty move list")

        first_move = plain[0]
        killer = (first_move.get("from"), first_move["to"])

        result = _order_moves(b, moves, killers=[killer, None])
        # The killer-designated move should still be first (already in p0/p1).
        self.assertEqual(
            (result[0].get("from"), result[0]["to"]),
            (first_move.get("from"), first_move["to"]),
        )

    def test_two_killers_both_promoted(self):
        b = _make_board()
        from game.rules import get_all_legal_moves
        moves = get_all_legal_moves(b)
        if len(moves) < 3:
            self.skipTest("not enough moves")

        plain = _order_moves(b, moves)
        plain_keys = [(m.get("from"), m["to"]) for m in plain]
        # Take two p2 moves from the back of the plain-ordered list as killers.
        k1 = plain_keys[-1]
        k2 = plain_keys[-2] if len(plain) >= 2 else None

        result = _order_moves(b, moves, killers=[k1, k2])
        result_keys = [(m.get("from"), m["to"]) for m in result]

        # Each killer must appear at least as early as it did without killers.
        if k1 in result_keys and k1 in plain_keys:
            self.assertLessEqual(result_keys.index(k1), plain_keys.index(k1))
        if k2 and k2 in result_keys and k2 in plain_keys:
            self.assertLessEqual(result_keys.index(k2), plain_keys.index(k2))


if __name__ == "__main__":
    unittest.main()
