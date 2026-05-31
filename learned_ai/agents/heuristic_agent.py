"""Thin wrapper around the existing heuristic minimax AI.

Provides the unified `choose_move(board, **kwargs)` surface so training and
evaluation code can swap agents without conditionals. All keyword arguments
beyond ``board`` are forwarded to the underlying ``GameAI.choose_move``.
"""

from __future__ import annotations

import random
from typing import Optional

from game.board import BoardState
from game.rules import get_all_legal_moves


def _load_game_ai():
    """Import GameAI without triggering ai/__init__'s heavy imports.

    ai/__init__.py pulls in chromadb / ollama which are not required for
    minimax inference. We register the ``ai`` package as a namespace package
    that contains *only* the submodules game_ai needs (heuristics,
    transposition_table, board_symmetry), then load game_ai itself.
    """
    import importlib
    import importlib.util
    import os
    import sys
    import types

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    ai_dir = os.path.join(repo_root, "ai")

    if "ai" not in sys.modules:
        ai_pkg = types.ModuleType("ai")
        ai_pkg.__path__ = [ai_dir]
        sys.modules["ai"] = ai_pkg

    needed = ["heuristics", "transposition_table", "board_symmetry", "game_ai"]
    for name in needed:
        full = f"ai.{name}"
        if full in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(
            full, os.path.join(ai_dir, f"{name}.py")
        )
        if spec is None or spec.loader is None:
            return importlib.import_module("ai.game_ai").GameAI
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules["ai.game_ai"].GameAI


GameAI = _load_game_ai()


def get_heuristic_evaluate():
    """Return ai.heuristics.evaluate without triggering heavy ai/__init__ imports.

    Safe to call after _load_game_ai() has already registered ai.heuristics in
    sys.modules (which happens at import time of this module).
    """
    import sys
    return sys.modules["ai.heuristics"].evaluate


class HeuristicAgent:
    def __init__(
        self,
        color: str = "B",
        difficulty: int = 3,
        blunder_probability: float = 0.0,
        game_ai: Optional[GameAI] = None,
    ) -> None:
        self.color = color
        self._blunder_probability = float(blunder_probability)
        # Always construct the inner GameAI with blunder_probability=0 so that
        # choose_move() never calls the expensive _pick_blunder() minimax scorer.
        # We intercept blunders here and return a random legal move instead,
        # which is effectively instantaneous and sufficient for training.
        self._inner: GameAI = game_ai or GameAI(
            color=color,
            difficulty=difficulty,
            blunder_probability=0.0,
        )
        self._last_was_blunder: bool = False

    def choose_move(self, board: BoardState, **kwargs: object) -> dict:
        # Intercept blunder moves here: pick a random legal move rather than
        # running the inner AI's expensive depth-3 minimax blunder scorer.
        if self._blunder_probability > 0.0 and random.random() < self._blunder_probability:
            moves = get_all_legal_moves(board)
            self._last_was_blunder = True
            return random.choice(moves) if moves else {}

        self._last_was_blunder = False
        accepted = {
            k: v
            for k, v in kwargs.items()
            if k
            in {
                "recognition",
                "endgame_state",
                "trajectory_hints",
                "top_n",
                "fast_early_game",
                "force_book_early",
                "fullgame_db",
            }
        }
        return self._inner.choose_move(board, **accepted)

    @property
    def last_was_blunder(self) -> bool:
        return self._last_was_blunder

    @property
    def last_thinking(self) -> str:
        return self._inner.last_thinking
