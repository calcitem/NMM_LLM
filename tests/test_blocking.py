"""
tests/test_blocking.py — Unit tests for placement-phase threat blocking.

_immediate_mill_threats() restricts the current player to the opponent's
closing squares when:
  - Fork (≥2 simultaneous opponent 2-configs): always restrict.
  - Single threat: restrict unless STM can close their own mill this turn.
"""
from __future__ import annotations

import unittest

from game.board import BoardState
from ai.game_ai import GameAI, _immediate_mill_threats


def _place(positions: dict, turn: str = "W") -> BoardState:
    return BoardState.from_setup(positions, turn=turn, phase="place")


class TestImmediateMillThreatsPlacement(unittest.TestCase):

    # ── Single 2-config: mandatory block (unless STM has own 2-config) ──────────

    def test_single_two_config_is_threat_when_stm_has_no_own_config(self):
        # Black has one 2-config (a7+d7 → closing g7); White has g4 only (no own 2-config).
        # Single threat with no carveout → g7 must be blocked.
        b = _place({"a7": "B", "d7": "B", "g4": "W"})
        self.assertEqual(_immediate_mill_threats(b), {"g7"})

    def test_zero_two_configs_no_threat(self):
        # No opponent 2-config at all.
        b = _place({"a7": "B", "b2": "B", "g4": "W"})
        self.assertEqual(_immediate_mill_threats(b), set())

    # ── Two 2-configs: fork triggers block ─────────────────────────────────────

    def test_two_configs_both_closing_squares_returned(self):
        # Black: a7+d7 (→ g7) and a1+d1 (→ g1).
        b = _place({"a7": "B", "d7": "B", "a1": "B", "d1": "B", "g4": "W", "e4": "W"})
        threats = _immediate_mill_threats(b)
        self.assertIn("g7", threats)
        self.assertIn("g1", threats)

    def test_three_configs_all_closing_squares_returned(self):
        # Black: a7+d7 (→ g7), a1+d1 (→ g1), b6+d6 (→ f6).
        b = _place({
            "a7": "B", "d7": "B",
            "a1": "B", "d1": "B",
            "b6": "B", "d6": "B",
            "g4": "W",
        })
        threats = _immediate_mill_threats(b)
        self.assertIn("g7", threats)
        self.assertIn("g1", threats)
        self.assertIn("f6", threats)

    def test_shared_pivot_fork_both_closing_squares_returned(self):
        # a7 is pivot: sits in mill a7-d7-g7 AND mill a1-a4-a7.
        # Black at a7+d7 → g7; Black at a7+a4 → a1.
        b = _place({"a7": "B", "d7": "B", "a4": "B", "g4": "W", "e4": "W"})
        threats = _immediate_mill_threats(b)
        self.assertIn("g7", threats)
        self.assertIn("a1", threats)

    # ── One of two closing squares already occupied ────────────────────────────

    def test_single_two_config_is_now_a_threat(self):
        # a7+g7 → closing d7 is the one real 2-config.
        # f2+b2 would close d2, but d2 is White → not a 2-config.
        # W (e4 only) has no own 2-config so carveout doesn't fire.
        # Single threat → d7 is mandatory block.
        b = _place({
            "a7": "B", "g7": "B",              # → d7
            "f2": "B", "b2": "B", "d2": "W",   # blocked by White → not counted
            "e4": "W",
        })
        threats = _immediate_mill_threats(b)
        self.assertEqual(threats, {"d7"})

    # ── Phase guard: move-phase board must not use placement-fork logic ─────────

    def test_move_phase_single_two_config_fires_normally(self):
        # In move phase, single 2-config with adjacent opp piece IS a threat.
        b = BoardState.from_setup(
            {"a7": "B", "d7": "B", "a4": "B",   # a4 adjacent to a7 → move-phase threat at g7
             "g1": "W", "d1": "W", "a1": "W",
             "b6": "W", "b4": "W", "b2": "W",
             "f6": "W", "f4": "W", "f2": "W"},
            turn="W", phase="move",
        )
        threats = _immediate_mill_threats(b)
        self.assertIn("g7", threats)

    def test_move_phase_single_threat_carveout_when_stm_closes_mill(self):
        # White threatens b6 (b2-b4-b6).  Black can close c3-c4-c5 — no block-only filter.
        b = BoardState.from_setup(
            {
                "a7": "B", "g4": "B", "g1": "B", "a1": "B", "d6": "B",
                "f2": "B", "c5": "B", "d3": "B", "c4": "B",
                "d7": "W", "g7": "W", "d1": "W", "a4": "W", "f6": "W",
                "f4": "W", "b2": "W", "b4": "W", "d5": "W",
            },
            turn="B", phase="move",
        )
        self.assertEqual(_immediate_mill_threats(b), set())

    # ── choose_move integration ────────────────────────────────────────────────

    def test_choose_move_restricted_to_closing_squares_on_fork(self):
        # Black: b6+d6 (→ f6) and f2+d2 (→ b2) — exactly two 2-configs, no overlap.
        # White must land on f6 or b2.
        b = _place({
            "b6": "B", "d6": "B",   # → f6
            "f2": "B", "d2": "B",   # → b2
            "g4": "W", "e4": "W",
        })
        threats = _immediate_mill_threats(b)
        self.assertEqual(threats, {"f6", "b2"})
        ai = GameAI(color="W", difficulty=3)
        move = ai.choose_move(b)
        self.assertIn(move["to"], threats)

    def test_choose_move_unrestricted_when_stm_has_own_two_config(self):
        # Black has one 2-config (a7+d7 → g7); White has e4+g4 (own 2-config e4-f4-g4).
        # Carveout fires: stm_can_close → threats empty → AI free to choose any square.
        b = _place({
            "a7": "B", "d7": "B",
            "g4": "W", "e4": "W",
        })
        self.assertEqual(_immediate_mill_threats(b), set())
        ai = GameAI(color="W", difficulty=3)
        move = ai.choose_move(b)
        self.assertIsNotNone(move)
        self.assertIn("to", move)


if __name__ == "__main__":
    unittest.main()
