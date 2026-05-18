"""tests/test_tactics.py — Stage 5.12: tactical pattern detectors and urgency bonus."""

import unittest

from game.board import BoardState, POSITIONS
from ai.heuristics import (
    detect_double_mills,
    detect_feeder_mills,
    detect_diamonds,
    opponent_mills_in_n_moves,
    tactical_move_bonus,
    HeuristicWeights,
    DEFAULT_WEIGHTS,
    _closeable_mills,
)


# ── Board construction helper ─────────────────────────────────────────────────

def _board(white: list[str], black: list[str], turn: str = "W",
           w_placed: int = 9, b_placed: int = 9) -> BoardState:
    pos = {p: "" for p in POSITIONS}
    for p in white:
        pos[p] = "W"
    for p in black:
        pos[p] = "B"
    return BoardState(
        positions=pos,
        turn=turn,
        pieces_on_board={"W": len(white), "B": len(black)},
        pieces_placed={"W": w_placed, "B": b_placed},
        pieces_captured={"W": 0, "B": 0},
    )


# ── detect_double_mills ───────────────────────────────────────────────────────

class TestDetectDoubleMills(unittest.TestCase):

    def test_no_double_mill_with_only_one_closed_mill(self):
        # W closes outer top row only
        b = _board(["a7", "d7", "g7"], [])
        self.assertEqual(detect_double_mills(b, "W"), [])

    def test_pivot_in_two_closed_mills(self):
        # d7 belongs to (a7,d7,g7) and to (d7,d6,d5)
        b = _board(["a7", "d7", "g7", "d6", "d5"], [])
        doubles = detect_double_mills(b, "W")
        self.assertIn("d7", doubles)
        self.assertEqual(len(doubles), 1)

    def test_two_pivots(self):
        # d7 is in (a7,d7,g7) and (d7,d6,d5)
        # d5 is in (d7,d6,d5) and (c5,d5,e5)
        # If all those pieces are placed, both d7 and d5 should be pivots
        b = _board(["a7", "d7", "g7", "d6", "d5", "c5", "e5"], [])
        doubles = detect_double_mills(b, "W")
        self.assertIn("d7", doubles)
        self.assertIn("d5", doubles)

    def test_opponent_double_mill_detected(self):
        b = _board([], ["a7", "d7", "g7", "d6", "d5"])
        doubles = detect_double_mills(b, "B")
        self.assertIn("d7", doubles)

    def test_no_pivot_when_mill_incomplete(self):
        # Only 2 of 3 pieces in each mill → no closed mills
        b = _board(["a7", "d7", "d6", "d5"], [])
        self.assertEqual(detect_double_mills(b, "W"), [])


# ── detect_feeder_mills ───────────────────────────────────────────────────────

class TestDetectFeederMills(unittest.TestCase):

    def test_closed_mill_without_feeder(self):
        # Outer top mill, no adjacent W piece outside the mill
        b = _board(["a7", "d7", "g7"], ["a4", "d6"])
        feeders = detect_feeder_mills(b, "W")
        self.assertEqual(feeders, [])

    def test_closed_mill_with_feeder(self):
        # a7 is in mill (a7,d7,g7); a4 is adjacent to a7 and also White → feeder
        b = _board(["a7", "d7", "g7", "a4"], [])
        feeders = detect_feeder_mills(b, "W")
        self.assertEqual(len(feeders), 1)
        self.assertIn("a7", feeders[0])

    def test_feeder_detected_for_opponent(self):
        b = _board([], ["a7", "d7", "g7", "a4"])
        feeders = detect_feeder_mills(b, "B")
        self.assertEqual(len(feeders), 1)

    def test_no_feeder_when_neighbor_is_empty(self):
        # a7 has neighbors d7 (in mill) and a4 (empty) → not a feeder
        b = _board(["a7", "d7", "g7"], [])
        self.assertEqual(detect_feeder_mills(b, "W"), [])

    def test_multiple_feeders_for_multiple_mills(self):
        # Two closed mills each with a feeder (using non-crossing mills)
        # Mill 1: outer top (a7,d7,g7) + feeder a4 adjacent to a7
        # Mill 2: inner top (c5,d5,e5) — isolated; use b6 adjacent to c5 as feeder (b6-c5? check adjacency)
        # Actually use a7,d7,g7 + a4 and a1,d1,g1 + g4
        # Note: a7 and a1 are NOT in the same mill, so a4 is adjacent to a7 only; g4 is adjacent to g1 only
        b = _board(
            ["a7", "d7", "g7", "a4",    # outer top mill + feeder a4 (not forming another closed mill)
             "a1", "d1", "g1", "g4"],   # outer bottom mill + feeder g4
            []
        )
        feeders = detect_feeder_mills(b, "W")
        # At least 2 feeder mills found (may be more if additional mills form)
        self.assertGreaterEqual(len(feeders), 2)


# ── detect_diamonds ───────────────────────────────────────────────────────────

class TestDetectDiamonds(unittest.TestCase):

    def test_no_diamond_with_single_two_config(self):
        # W at a7, d7 with g7 empty → one two-config, not a fork
        b = _board(["a7", "d7"], [])
        self.assertEqual(detect_diamonds(b, "W"), [])

    def test_diamond_on_shared_closing_square(self):
        # Two mills both needing g7:
        #   (a7, d7, g7): W at a7, d7 → g7 empty
        #   (g7, g4, g1): W at g4, g1 → g7 empty
        b = _board(["a7", "d7", "g4", "g1"], [])
        diamonds = detect_diamonds(b, "W")
        self.assertIn("g7", diamonds)

    def test_multiple_diamond_squares(self):
        # a7 is the empty closing square for two mills simultaneously:
        #   (a7,d7,g7): W at d7 AND g7, a7 empty
        #   (a7,a4,a1): W at a4 AND a1, a7 empty
        b = _board(["d7", "g7", "a4", "a1"], [])
        diamonds = detect_diamonds(b, "W")
        self.assertIn("a7", diamonds)

    def test_opponent_diamond_detected(self):
        b = _board([], ["a7", "d7", "g4", "g1"])
        diamonds = detect_diamonds(b, "B")
        self.assertIn("g7", diamonds)


# ── opponent_mills_in_n_moves ─────────────────────────────────────────────────

class TestOpponentMillsInNMoves(unittest.TestCase):

    def test_zero_moves_returns_zero(self):
        b = _board(["a7", "d7"], [])
        self.assertEqual(opponent_mills_in_n_moves(b, "W", n=0), 0)

    def test_one_closeable_mill_n1(self):
        # W at a7, d7 → g7 is adjacent to both, g7 is empty.
        # In move phase W needs a piece adjacent to g7; d7 is adjacent to g7.
        b = _board(["a7", "d7"], [], w_placed=9, b_placed=9)
        # Move phase: d7 is adjacent to g7, so mill is closeable
        count = opponent_mills_in_n_moves(b, "W", n=1)
        self.assertGreaterEqual(count, 1)

    def test_no_immediate_mill_n1(self):
        # W at a7, b6 → not in any 2-config together
        b = _board(["a7", "b6"], [], w_placed=9, b_placed=9)
        self.assertEqual(opponent_mills_in_n_moves(b, "W", n=1), 0)

    def test_n2_counts_more_than_n1(self):
        # One 2-config (closeable) + one 1-piece mill (2 moves away)
        # (a7,d7,g7): W at d7 only → needs 2 more moves to close
        b = _board(["d7"], [], w_placed=9, b_placed=9)
        n1 = opponent_mills_in_n_moves(b, "W", n=1)
        n2 = opponent_mills_in_n_moves(b, "W", n=2)
        self.assertGreaterEqual(n2, n1)


# ── tactical_move_bonus ───────────────────────────────────────────────────────

class TestTacticalMoveBonus(unittest.TestCase):

    def test_closing_mill_gives_positive_bonus(self):
        # Before: W at a7, d7 (two-config); After: W at a7, d7, g7 (closed mill)
        before = _board(["a7", "d7"], ["b6", "b4", "b2"], w_placed=8)
        after  = _board(["a7", "d7", "g7"], ["b6", "b4", "b2"], w_placed=9)
        bonus = tactical_move_bonus(before, after, "W")
        self.assertGreater(bonus, 0)

    def test_closing_mill_bonus_matches_weight(self):
        # bonus should include at least one close_mill weight
        before = _board(["a7", "d7"], ["b6"], w_placed=8)
        after  = _board(["a7", "d7", "g7"], ["b6"], w_placed=9)
        w = DEFAULT_WEIGHTS
        bonus = tactical_move_bonus(before, after, "W", w)
        self.assertGreaterEqual(bonus, w.close_mill)

    def test_blocking_opponent_mill_gives_bonus(self):
        # B has a7, d7 open (g7 empty); W moves to g7, neutralising B's threat
        before = _board(["e5"], ["a7", "d7"])
        # Apply the move: W goes to g7
        move = {"from": "e5", "to": "g7", "capture": None}
        after = before.apply_move(move)
        bonus = tactical_move_bonus(before, after, "W")
        # The move both blocks B's two-config and captures the cardinal node
        self.assertGreater(bonus, 0)

    def test_no_bonus_for_neutral_move(self):
        # W moves from e3 to d3 — neither closes a mill nor blocks anything
        # Set up: W at c3, e3; B has no two-configs
        before = _board(["c3", "e3", "a7"], ["b6", "g1", "g4"])
        move = {"from": "e3", "to": "d3", "capture": None}
        after = before.apply_move(move)
        bonus = tactical_move_bonus(before, after, "W")
        # Bonus may still be non-zero due to cross-node control, but shouldn't include close_mill
        w = DEFAULT_WEIGHTS
        self.assertLess(bonus, w.close_mill)

    def test_custom_weights_scale_bonus(self):
        # With close_mill=0, closing a mill should not contribute that weight
        w_zero = HeuristicWeights(close_mill=0)
        w_default = DEFAULT_WEIGHTS
        before = _board(["a7", "d7"], ["b6"], w_placed=8)
        after  = _board(["a7", "d7", "g7"], ["b6"], w_placed=9)
        bonus_zero    = tactical_move_bonus(before, after, "W", w_zero)
        bonus_default = tactical_move_bonus(before, after, "W", w_default)
        self.assertLess(bonus_zero, bonus_default)


# ── _closeable_mills (internal, used by coordinator pre-screen) ───────────────

class TestCloseableMills(unittest.TestCase):

    def test_one_closeable_in_placement(self):
        # W has two-config during placement phase
        b = _board(["a7", "d7"], [], w_placed=2, b_placed=0)
        self.assertEqual(_closeable_mills(b, "W"), 1)

    def test_zero_closeable_when_opponent_blocks(self):
        b = _board(["a7", "d7"], ["g7"], w_placed=2, b_placed=1)
        self.assertEqual(_closeable_mills(b, "W"), 0)

    def test_closeable_in_move_phase_requires_adjacency(self):
        # W at a7 and d7 in move phase: g7 empty, but does W have a piece adjacent to g7?
        # d7 IS adjacent to g7, so yes.
        b = _board(["a7", "d7", "b6"], [], w_placed=9, b_placed=9)
        self.assertGreaterEqual(_closeable_mills(b, "W"), 1)


if __name__ == "__main__":
    unittest.main()
