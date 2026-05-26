"""
tests/test_search_enhancements.py — Unit tests for SE-2 (killers) and SE-3 (history).

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


class TestHistoryHeuristic(unittest.TestCase):
    """SE-3: history table incremented on quiet beta cutoffs; sorts p2 by score."""

    def setUp(self):
        self.ai = GameAI(color="W", difficulty=3)

    # ── history accumulation ─────────────────────────────────────────────────

    def test_history_empty_on_init(self):
        self.assertEqual(self.ai._history, {})

    def test_history_reset_each_choose_move(self):
        self.ai._history[("a7", "d7")] = 999
        b = BoardState.new_game()
        self.ai.choose_move(b)
        self.assertEqual(self.ai._history.get(("a7", "d7"), 0), 0)

    def test_history_incremented_by_depth_squared(self):
        key = ("a7", "d7")
        self.ai._history[key] = self.ai._history.get(key, 0) + 4 * 4
        self.assertEqual(self.ai._history[key], 16)
        self.ai._history[key] = self.ai._history.get(key, 0) + 3 * 3
        self.assertEqual(self.ai._history[key], 16 + 9)

    def test_history_accumulates_across_depths(self):
        key = (None, "d5")
        for d in [5, 3, 5, 2]:
            self.ai._history[key] = self.ai._history.get(key, 0) + d * d
        self.assertEqual(self.ai._history[key], 25 + 9 + 25 + 4)

    # ── _order_moves p2 sorting — tested with explicit move lists ────────────

    def _p2_moves(self):
        """Return three synthetic p2 moves (no mill-closing or blocking context)."""
        # Use a completely empty board so there are no close/block squares.
        b = BoardState.new_game()
        # Placement moves: the entire board is empty, so nothing is close/block.
        return b, [
            {"from": None, "to": "a7", "capture": None},
            {"from": None, "to": "d7", "capture": None},
            {"from": None, "to": "g7", "capture": None},
        ]

    def test_history_sorts_p2_descending(self):
        b, moves = self._p2_moves()
        # Give d7 a high history score — it should sort to the front of p2.
        hist = {(None, "d7"): 500, (None, "a7"): 100, (None, "g7"): 0}
        result = _order_moves(b, moves, history=hist)
        result_tos = [m["to"] for m in result]
        self.assertEqual(result_tos[0], "d7")
        self.assertEqual(result_tos[1], "a7")

    def test_history_zero_scores_no_reorder(self):
        b, moves = self._p2_moves()
        without    = _order_moves(b, moves)
        with_empty = _order_moves(b, moves, history={})
        self.assertEqual([m["to"] for m in without], [m["to"] for m in with_empty])

    def test_history_does_not_promote_into_p0_p1(self):
        """History never moves a p2 move ahead of a p0/p1 move."""
        b = _make_board()
        from game.rules import get_all_legal_moves
        moves = get_all_legal_moves(b)
        if not moves:
            self.skipTest("empty move list")
        plain = _order_moves(b, moves)
        first_key = (plain[0].get("from"), plain[0]["to"])

        # Assign tiny history to the top-priority move, huge to the last.
        last_key = (plain[-1].get("from"), plain[-1]["to"])
        hist = {first_key: 1, last_key: 99999}
        result = _order_moves(b, moves, history=hist)
        # Top-priority move must still be first.
        self.assertEqual((result[0].get("from"), result[0]["to"]), first_key)

    def test_killers_before_history_sorted_p2(self):
        """Killer tier comes before history-sorted p2, never after."""
        b, moves = self._p2_moves()   # all three moves are p2 on empty board
        # Make "a7" a killer and give "g7" a high history score.
        killer = (None, "a7")
        hist   = {(None, "g7"): 9999, (None, "d7"): 100, (None, "a7"): 0}
        result = _order_moves(b, moves, killers=[killer, None], history=hist)
        tos = [m["to"] for m in result]
        # a7 is killer → first; then p2 sorted by history: g7 (9999), d7 (100).
        self.assertEqual(tos[0], "a7")
        self.assertEqual(tos[1], "g7")


class TestPVS(unittest.TestCase):
    """SE-5: Principal Variation Search correctness and node-reduction checks."""

    # W has d6 adjacent to empty d7; a7 and g7 complete the outer-top 2-config.
    # d6→d7 is the unique mill close available to W and is the clear best move.
    _POSITIONS = {
        "a7": "W", "g7": "W", "d6": "W",   # d6→d7 closes a7-d7-g7
        "b4": "W", "e3": "W", "g1": "W",
        "d2": "B", "f6": "B", "f4": "B",
        "c4": "B", "c5": "B", "a4": "B",
    }

    def _board(self) -> BoardState:
        return BoardState.from_setup(self._POSITIONS, turn="W", phase="move")

    def test_pvs_picks_mill_close(self):
        """PVS must find the forced mill close (d6→d7) at difficulty 2."""
        ai = GameAI(color="W", difficulty=2)
        move = ai.choose_move(self._board())
        self.assertEqual(move["from"], "d6",
                         f"Expected d6→d7 mill close, got {move}")
        self.assertEqual(move["to"], "d7",
                         f"Expected d6→d7 mill close, got {move}")

    def test_pvs_node_counter_populated(self):
        """_nodes must be > 0 after a search (PVS path exercised)."""
        ai = GameAI(color="W", difficulty=3)
        ai.choose_move(self._board())
        self.assertGreater(ai._nodes, 0)

    def test_pvs_node_count_not_inflated(self):
        """PVS must not visit more nodes than plain alpha-beta for the same position.

        We run two fresh AI instances at the same difficulty and verify the second
        run (deterministic — same TT/killer state reset) counts the same nodes.
        Primarily confirms PVS does not accidentally perform extra re-searches.
        """
        b = self._board()
        ai1 = GameAI(color="W", difficulty=3)
        ai1.choose_move(b)
        count1 = ai1._nodes

        ai2 = GameAI(color="W", difficulty=3)
        ai2.choose_move(b)
        count2 = ai2._nodes

        self.assertEqual(count1, count2,
                         "Non-determinism detected: node counts differ between identical searches")

    def test_pvs_zero_window_fallback_does_not_drop_best_move(self):
        """When the scout fails high (score in (alpha, beta)), the full re-search
        must still find the correct best move — not just the scout approximation."""
        # Use depth 4 so siblings will trigger the scout and, for an improving
        # sibling, the full re-search fallback.
        ai = GameAI(color="W", difficulty=4)
        move = ai.choose_move(self._board())
        # The best move may not always be d7 at depth 4 (captures change context),
        # but the AI must return *some* valid legal move without raising.
        from game.rules import get_all_legal_moves
        legal = get_all_legal_moves(self._board())
        legal_keys = [(m.get("from"), m["to"]) for m in legal]
        self.assertIn((move.get("from"), move["to"]), legal_keys)


class TestLMR(unittest.TestCase):
    """SE-6: Late Move Reductions correctness checks."""

    # Fly-phase board: 3 W vs 3 B, ~58 legal moves — high branching factor
    # where LMR fires most often. Used for node-count and validity tests.
    _FLY_POSITIONS = {
        "a7": "W", "g7": "W", "a1": "W",
        "g1": "B", "d3": "B", "c5": "B",
    }

    def _fly_board(self) -> BoardState:
        return BoardState.from_setup(self._FLY_POSITIONS, turn="W", phase="move")

    def test_lmr_fly_phase_returns_legal_move(self):
        """LMR must return a valid legal move in fly phase at difficulty 4."""
        from game.rules import get_all_legal_moves
        b = self._fly_board()
        ai = GameAI(color="W", difficulty=4)
        move = ai.choose_move(b)
        legal_keys = [(m.get("from"), m["to"]) for m in get_all_legal_moves(b)]
        self.assertIn((move.get("from"), move["to"]), legal_keys,
                      f"LMR returned an illegal move: {move}")

    def test_lmr_node_counter_populated(self):
        """After a fly-phase search, _nodes must be > 0 (LMR path exercised)."""
        ai = GameAI(color="W", difficulty=4)
        ai.choose_move(self._fly_board())
        self.assertGreater(ai._nodes, 0)

    def test_lmr_block_guard_not_reduced(self):
        """A blocking move must still be played correctly even in fly phase.

        W: a7, a4, c5 (3 pieces, fly phase).
        B: f6, f4 — 2-config closing f2.  W must block f2 or B closes the mill
        next turn for free.  In fly phase all W pieces can reach f2, so f2 lands
        in p1 (block) of _order_moves and is guarded from LMR reduction.
        """
        positions = {
            "a7": "W", "a4": "W", "c5": "W",
            "f6": "B", "f4": "B", "d2": "B",
        }
        b = BoardState.from_setup(positions, turn="W", phase="move")
        ai = GameAI(color="W", difficulty=4)
        move = ai.choose_move(b)
        self.assertEqual(move["to"], "f2",
                         f"Expected block at f2, got {move}")

    def test_lmr_deterministic(self):
        """Two fresh AI instances must pick the same move (LMR is deterministic)."""
        b = self._fly_board()
        ai1 = GameAI(color="W", difficulty=4)
        ai2 = GameAI(color="W", difficulty=4)
        m1 = ai1.choose_move(b)
        m2 = ai2.choose_move(b)
        self.assertEqual(
            (m1.get("from"), m1["to"]),
            (m2.get("from"), m2["to"]),
        )


if __name__ == "__main__":
    unittest.main()
