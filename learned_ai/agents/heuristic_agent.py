"""Thin wrapper around the existing heuristic minimax AI.

Provides the unified `choose_move(board, **kwargs)` surface so training and
evaluation code can swap agents without conditionals. All keyword arguments
beyond ``board`` are forwarded to the underlying ``GameAI.choose_move``.
"""

from __future__ import annotations

from typing import Optional

from game.board import BoardState


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


class HeuristicAgent:
    def __init__(
        self,
        color: str = "B",
        difficulty: int = 3,
        blunder_probability: float = 0.0,
        game_ai: Optional[GameAI] = None,
    ) -> None:
        self.color = color
        self._inner: GameAI = game_ai or GameAI(
            color=color,
            difficulty=difficulty,
            blunder_probability=blunder_probability,
        )

    def choose_move(self, board: BoardState, **kwargs: object) -> dict:
        # The wrapped GameAI silently ignores unknown kwargs by accepting
        # named params; we forward only the ones it understands.
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
        return self._inner.last_was_blunder

    @property
    def last_thinking(self) -> str:
        return self._inner.last_thinking
