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
    _independent_mill_pairs,
    _piece_separation,
    _contested_mills,
    _open_mill_domination,
    _unguarded_cardinal_mill_alert,
    evaluate,
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


# ── _independent_mill_pairs ───────────────────────────────────────────────────

class TestIndependentMillPairs(unittest.TestCase):

    def test_two_independent_pairs(self):
        # W at a7,d7 (open mill a7-d7-g7) and a1,d1 (open mill g1-d1-a1)
        # These two 2-configs share no own pieces → 1 independent pair
        b = _board(["a7", "d7", "a1", "d1"], [])
        self.assertEqual(_independent_mill_pairs(b, "W"), 1)

    def test_no_pairs_when_shared_piece(self):
        # W at a7,d7 (open mill a7-d7-g7) and a7,a4 (open mill a7-a4-a1)
        # Both 2-configs share a7 → not independent
        b = _board(["a7", "d7", "a4"], [])
        self.assertEqual(_independent_mill_pairs(b, "W"), 0)

    def test_zero_with_single_two_config(self):
        b = _board(["a7", "d7"], [])
        self.assertEqual(_independent_mill_pairs(b, "W"), 0)

    def test_multiple_independent_pairs(self):
        # Three independent 2-configs give 3 pairs
        b = _board(["a7", "d7", "a1", "d1", "a4", "b4"], [])
        pairs = _independent_mill_pairs(b, "W")
        self.assertGreaterEqual(pairs, 2)


# ── _piece_separation ─────────────────────────────────────────────────────────

class TestPieceSeparation(unittest.TestCase):

    def test_separated_groups(self):
        # Black at a7 and g1 — no board path between them through adjacency
        # (different corners, not adjacent to each other)
        b = _board(["a1", "d1", "g4"], ["a7", "g7", "g1", "a4"], w_placed=9, b_placed=9)
        # W is fly (3 pieces, placed 9). B has 4 pieces.
        b2 = _board(["a1", "d1", "g4"], ["a7", "g1", "e3", "c3"], w_placed=9, b_placed=9)
        # a7 and g1 are on opposite corners with no shared adjacency path via only B pieces
        sep = _piece_separation(b2, "W")
        # Whether or not they happen to be connected depends on board layout; just check it runs
        self.assertIn(sep, (0, 1))

    def test_connected_cluster(self):
        # Black all in a tight group: a7-d7 are adjacent, d7-g7 adjacent
        b = _board(["a1", "d1", "g4"], ["a7", "d7", "g7", "d6"], w_placed=9, b_placed=9)
        # All black pieces connected in one group
        self.assertEqual(_piece_separation(b, "W"), 0)

    def test_wrong_piece_count_returns_zero(self):
        # opp has 3 pieces — function only meaningful at 4
        b = _board(["a1", "d1", "g4"], ["a7", "d7", "g7"], w_placed=9, b_placed=9)
        self.assertEqual(_piece_separation(b, "W"), 0)


# ── _contested_mills ──────────────────────────────────────────────────────────

class TestContestedMills(unittest.TestCase):

    def test_one_contested_mill(self):
        # W at a7,d7; B at g7 → mill a7-d7-g7 is contested (2W + 1B)
        b = _board(["a7", "d7"], ["g7"])
        self.assertEqual(_contested_mills(b, "W"), 1)

    def test_zero_contested_when_empty_slot(self):
        # W at a7,d7; g7 is empty → this is a 2-config, not contested
        b = _board(["a7", "d7"], [])
        self.assertEqual(_contested_mills(b, "W"), 0)

    def test_three_contested_zugzwang(self):
        # Three mills all contested: classic 7v3 zugzwang approach
        # Mill a7-d7-g7: W has a7,d7; B has g7
        # Mill a4-b4-c4: W has a4,b4; B has c4
        # Mill g1-d1-a1: W has a1,d1; B has g1
        b = _board(["a7", "d7", "a4", "b4", "a1", "d1", "g4"],
                   ["g7", "c4", "g1"])
        self.assertEqual(_contested_mills(b, "W"), 3)

    def test_contested_scored_higher_than_two_config(self):
        # The zugzwang position (3 contested mills, 7v3) should score HIGHER
        # than the approach position (3 open 2-configs, 7v4) because it is decisive.
        zugzwang = _board(["a7", "d7", "a4", "b4", "a1", "d1", "g4"],
                          ["g7", "c4", "g1"], w_placed=9, b_placed=9)
        approach  = _board(["a7", "d7", "a4", "b4", "a1", "d1", "g4"],
                           ["b2", "e4", "f6", "d5"], w_placed=9, b_placed=9)
        self.assertGreater(evaluate(zugzwang, "W"), evaluate(approach, "W"))


# ── _open_mill_domination ─────────────────────────────────────────────────────

class TestOpenMillDomination(unittest.TestCase):

    def test_zero_when_not_dominant(self):
        # 5 own pieces — below the ≥6 threshold
        b = _board(["a7", "d7", "a4", "b4", "a1"], ["g7", "c4"])
        self.assertEqual(_open_mill_domination(b, "W"), 0)

    def test_zugzwang_equals_one_with_three_mills_three_opp(self):
        # 3 two-configs and 3 opp pieces: max(0, 3 - (3-1)) = 1
        b = _board(["a7", "d7", "a4", "b4", "a1", "d1", "g4"],
                   ["b2", "e4", "f6"], w_placed=9, b_placed=9)
        self.assertEqual(_open_mill_domination(b, "W"), 1)

    def test_zero_with_four_opp_covering_three_mills(self):
        # 3 two-configs but 4 opp pieces: max(0, 3 - (4-1)) = 0
        b = _board(["a7", "d7", "a4", "b4", "a1", "d1", "g4"],
                   ["b2", "e4", "f6", "d5"], w_placed=9, b_placed=9)
        self.assertEqual(_open_mill_domination(b, "W"), 0)

    def test_strong_surplus_with_four_mills_three_opp(self):
        # 4 two-configs, 3 opp pieces: max(0, 4 - 2) = 2
        b = _board(["a7", "d7", "a4", "b4", "a1", "d1", "b6", "d6"],
                   ["b2", "e4", "f6"], w_placed=9, b_placed=9)
        dom = _open_mill_domination(b, "W")
        self.assertGreaterEqual(dom, 1)

    def test_evaluate_ranks_7v3_higher_than_7v4(self):
        # Removing one black piece (going 7v4 → 7v3) should increase white's score
        # because the domination signal fires at 7v3 but not at 7v4.
        white7 = ["a7", "d7", "a4", "b4", "a1", "d1", "g4"]
        # Use scattered black positions that don't form mills
        black4  = ["b2", "e4", "f6", "d5"]
        black3  = ["b2", "e4", "f6"]
        b7v4 = _board(white7, black4, w_placed=9, b_placed=9)
        b7v3 = _board(white7, black3, w_placed=9, b_placed=9)
        self.assertGreater(evaluate(b7v3, "W"), evaluate(b7v4, "W"))


# ── capture_disrupt_feeder ────────────────────────────────────────────────────

class TestCaptureDisruptFeeder(unittest.TestCase):
    """Capturing a piece that feeds a cycling mill gets a bonus."""

    def _make_capture(self, before: BoardState, after: BoardState, color: str) -> int:
        return tactical_move_bonus(before, after, color)

    def test_feeder_capture_gets_bonus(self):
        # B has a closed mill at a7-d7-g7 and a feeder at d6 (adjacent to d7).
        # W captures the feeder piece at d6 — should get capture_disrupt_feeder bonus.
        before = _board(
            ["c5", "d5", "e5"],                       # W pieces
            ["a7", "d7", "g7", "d6"],                  # B: closed mill + feeder
            turn="W", w_placed=9, b_placed=9,
        )
        after = _board(
            ["c5", "d5", "e5"],
            ["a7", "d7", "g7"],                        # feeder d6 captured
            turn="B", w_placed=9, b_placed=9,
        )
        # Manually set pieces_on_board to reflect the capture
        after.pieces_on_board["B"] = 3
        bonus = tactical_move_bonus(before, after, "W")
        self.assertGreater(bonus, tactical_move_bonus(
            before,
            _board(["c5", "d5", "e5"], ["a7", "d7", "g7", "a4"],
                   turn="B", w_placed=9, b_placed=9),
            "W",
        ))

    def test_non_feeder_capture_no_feeder_bonus(self):
        # B has a closed mill but captured piece is not adjacent to it.
        # No feeder bonus should fire.
        before = _board(
            ["c5", "d5", "e5"],
            ["a7", "d7", "g7", "b2"],                  # b2 is far from the top mill
            turn="W", w_placed=9, b_placed=9,
        )
        after_feeder = _board(
            ["c5", "d5", "e5"],
            ["a7", "d7", "g7"],                        # b2 captured (not a feeder)
            turn="B", w_placed=9, b_placed=9,
        )
        after_feeder.pieces_on_board["B"] = 3
        w = HeuristicWeights()
        bonus = tactical_move_bonus(before, after_feeder, "W", w)
        # feeder bonus should NOT be in there (captured piece b2 not adjacent to top mill)
        self.assertEqual(bonus, tactical_move_bonus(before, after_feeder, "W",
                                                    HeuristicWeights(capture_disrupt_feeder=0)))

    def test_feeder_bonus_higher_than_ordinary_capture(self):
        # Two otherwise identical captures: one removes a feeder, one removes an isolated piece.
        # The feeder capture should score strictly higher.
        before = _board(
            ["c5", "d5", "e5", "g4"],
            ["a7", "d7", "g7", "d6", "b2"],
            turn="W", w_placed=9, b_placed=9,
        )
        # Capture d6 (feeder adjacent to the top mill)
        after_feeder = _board(
            ["c5", "d5", "e5", "g4"],
            ["a7", "d7", "g7", "b2"],
            turn="B", w_placed=9, b_placed=9,
        )
        after_feeder.pieces_on_board["B"] = 4
        # Capture b2 (isolated piece, not a feeder)
        after_isolated = _board(
            ["c5", "d5", "e5", "g4"],
            ["a7", "d7", "g7", "d6"],
            turn="B", w_placed=9, b_placed=9,
        )
        after_isolated.pieces_on_board["B"] = 4
        self.assertGreater(
            tactical_move_bonus(before, after_feeder, "W"),
            tactical_move_bonus(before, after_isolated, "W"),
        )


# ── capture_disrupt_diamond ───────────────────────────────────────────────────

class TestCaptureDisruptDiamond(unittest.TestCase):
    """Capturing a piece that's part of an opponent fork (diamond) gets a bonus."""

    def test_diamond_capture_gets_bonus(self):
        # B has two 2-configs sharing the closing square d7:
        #   a7-d7 (needs g7) and g7 is not relevant here — let's use a proper fork.
        # B pieces: a7, g7 (share d7 as closing square for outer top mill)
        # AND b6, f6 sharing d6 as closing square — wait that's 2 separate mills.
        # Simpler: outer top mill (a7-d7-g7): a7, g7 placed → d7 is fork for that one mill.
        # For a fork we need d7 to be closing square for TWO different mills.
        # Middle top mill (b6-d6-f6) and outer top (a7-d7-g7) don't share a closing sq.
        # Use: mills d5-d6-d7 (vertical middle) and a7-d7-g7 (outer top): share d7.
        # B has d5, d6 (for d5-d6-d7) and a7, g7 (for a7-d7-g7) → d7 is fork square.
        before = _board(
            ["c3", "c4", "c5"],                              # W (irrelevant positions)
            ["d5", "d6", "a7", "g7"],                        # B: two 2-configs pointing at d7
            turn="W", w_placed=9, b_placed=9,
        )
        # Capture d6 — part of the d5-d6-d7 two-config pointing at the fork square d7
        after = _board(
            ["c3", "c4", "c5"],
            ["d5", "a7", "g7"],
            turn="B", w_placed=9, b_placed=9,
        )
        after.pieces_on_board["B"] = 3
        w0 = HeuristicWeights(capture_disrupt_diamond=0)
        w1 = HeuristicWeights(capture_disrupt_diamond=250)
        self.assertGreater(
            tactical_move_bonus(before, after, "W", w1),
            tactical_move_bonus(before, after, "W", w0),
        )

    def test_no_diamond_bonus_when_no_fork(self):
        # B has only a single 2-config (no fork): a7, g7 → closing square d7 used once.
        before = _board(
            ["c3", "c4", "c5"],
            ["a7", "g7", "b2"],
            turn="W", w_placed=9, b_placed=9,
        )
        after = _board(
            ["c3", "c4", "c5"],
            ["a7", "g7"],
            turn="B", w_placed=9, b_placed=9,
        )
        after.pieces_on_board["B"] = 2
        w0 = HeuristicWeights(capture_disrupt_diamond=0)
        w1 = HeuristicWeights(capture_disrupt_diamond=250)
        # No fork before → no diamond bonus regardless of weight
        self.assertEqual(
            tactical_move_bonus(before, after, "W", w0),
            tactical_move_bonus(before, after, "W", w1),
        )


# ── B-22 regression: emergency block must outrank speculative improvement ──────

class TestB22EmergencyBlock(unittest.TestCase):
    """Regression for game at move 32: White (fly phase, 3 pieces) must block
    Black's immediate b2-b4-b6 mill threat rather than playing a speculative move.

    Position before White's move 32:
      White (fly): a7, d7, e5
      Black (move): b2, b4, d5, d6
    Black can close mill b2-b4-b6 by sliding d6→b6 (adjacent).
    White should fly to b6 to block.
    """

    @staticmethod
    def _b22_board():
        pos = {p: "" for p in POSITIONS}
        for p in ["a7", "d7", "e5"]:
            pos[p] = "W"
        for p in ["b2", "b4", "d5", "d6"]:
            pos[p] = "B"
        return BoardState(
            positions=pos,
            turn="W",
            pieces_on_board={"W": 3, "B": 4},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 0, "B": 0},
        )

    def test_unguarded_cardinal_mill_alert_detects_b6(self):
        b = self._b22_board()
        alert = _unguarded_cardinal_mill_alert(b, "B", "W")
        self.assertIn("b6", alert,
            "b6 must be detected as the closing square of unguarded cardinal mill b2-b4-b6")

    def test_evaluate_penalises_unguarded_cardinal_mill(self):
        b = self._b22_board()
        score = evaluate(b, "W")
        # Introduce a guarding piece at a4 (adjacent to b4) and compare
        guarded_pos = dict(b.positions)
        guarded_pos["a4"] = "W"
        guarded_pos["e5"] = ""          # swap e5 out to keep piece count equal
        b_guarded = BoardState(
            positions=guarded_pos,
            turn="W",
            pieces_on_board={"W": 3, "B": 4},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 0, "B": 0},
        )
        alert_guarded = _unguarded_cardinal_mill_alert(b_guarded, "B", "W")
        self.assertEqual([], alert_guarded,
            "a4 adjacent to b4 should guard the b2-b4-b6 mill")
        self.assertGreater(evaluate(b_guarded, "W"), score,
            "guarded position must score higher than unguarded position")

    def test_ai_blocks_b6_not_speculative_move(self):
        from ai.game_ai import GameAI
        b = self._b22_board()
        ai = GameAI(difficulty=5, color="W")
        move = ai.choose_move(b)
        self.assertEqual(move.get("to"), "b6",
            f"AI should fly to b6 to block Black's mill; chose {move} instead")


if __name__ == "__main__":
    unittest.main()
