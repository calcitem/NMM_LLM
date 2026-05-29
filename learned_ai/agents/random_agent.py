"""Uniform-random legal-move agent — mirrors the heuristic AI's interface."""

from __future__ import annotations

import random
from typing import Optional

from game.board import BoardState
from game.rules import get_all_legal_moves


class RandomAgent:
    def __init__(self, color: str = "B", seed: Optional[int] = None) -> None:
        self.color = color
        self._rng = random.Random(seed)
        self.last_was_blunder = False
        self.last_thinking = "random"

    def choose_move(self, board: BoardState, **_: object) -> dict:
        moves = get_all_legal_moves(board)
        if not moves:
            return {}
        return self._rng.choice(moves)
