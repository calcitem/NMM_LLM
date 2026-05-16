"""
ai/game_ai.py — Minimax AI using negamax + alpha-beta pruning.

GameAI plays the computer's side.  It also exposes score_move() for rating
human moves, used by the LLM commentary system to decide whether to comment.

Blunder mode: set blunder_probability > 0 to make the AI occasionally play a
deliberately poor move so the human can practise exploiting mistakes.
"""

from __future__ import annotations

import random
import time
from typing import Optional, Tuple

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
from .heuristics import INF, evaluate

# Maps difficulty (1–10) to minimax search depth.
# Difficulties 9–10 use iterative deepening; 9 gets 20 s, 10 gets 45 s.
_DEPTH_TABLE = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}
_TIME_LIMIT = {5: 10.0, 9: 20.0, 10: 45.0}  # difficulty → iterative-deepening budget

# First 2 placements per AI colour use a short time budget regardless of
# difficulty — the tree is tiny with so few pieces on the board.
_EARLY_GAME_MOVES = 2
_EARLY_GAME_TIME  = 1.5  # seconds


class GameAI:
    """
    Minimax AI for Nine Men's Morris.

    Parameters
    ----------
    color : "W" or "B"
        The colour this AI controls.
    difficulty : int [1-5]
        Search depth. Difficulty 5 uses iterative deepening up to _TIME_LIMIT.
    blunder_probability : float [0.0-1.0]
        Probability of playing a deliberately bad move each turn.
        0.0 = always plays best; 1.0 = always blunders.
        Bad moves are drawn from the bottom quartile of legal-move scores.
    """

    def __init__(
        self,
        color: str = "B",
        difficulty: int = 3,
        blunder_probability: float = 0.0,
    ) -> None:
        self.color = color
        self.difficulty = max(1, min(10, difficulty))
        self.blunder_probability = max(0.0, min(1.0, blunder_probability))
        self._nodes = 0
        self.last_was_blunder: bool = False   # flag readable by Coordinator / MillsLLM
        self.force_aggressive: bool = False   # when True, disables fly-sacrifice heuristic

    # ── Public API ────────────────────────────────────────────────────────────

    def choose_move(
        self,
        board: BoardState,
        recognition=None,   # RecognitionResult  — Stage 4
        endgame_state=None, # EndgameState        — Stage 5
    ) -> dict:
        """Return the best (or deliberately bad) legal move dict for self.color."""
        moves = get_all_legal_moves(board)
        if not moves:
            return {}
        if len(moves) == 1:
            self.last_was_blunder = False
            return moves[0]

        # Blunder mode: occasionally play a bad move on purpose
        if self.blunder_probability > 0.0 and random.random() < self.blunder_probability:
            blunder = self._pick_blunder(board, moves)
            self.last_was_blunder = True
            return blunder

        self.last_was_blunder = False

        # Early-game fast path: use a short time budget for the first two
        # placements this AI makes — the position tree is tiny with so few
        # pieces on the board and deep search wastes time.
        if (board.pieces_placed.get(self.color, 0) < _EARLY_GAME_MOVES
                and sum(board.pieces_placed.values()) < _EARLY_GAME_MOVES * 2):
            return self._iterative_deepen(board, _EARLY_GAME_TIME)

        if self.difficulty in _TIME_LIMIT:
            return self._iterative_deepen(board, _TIME_LIMIT[self.difficulty])

        depth = _DEPTH_TABLE[self.difficulty]

        # Deeper search in endgame for better tactical accuracy.
        if endgame_state is not None and endgame_state.active:
            depth += 2 if endgame_state.deep else 1

        scored = self._score_all(board, moves, depth, endgame_state=endgame_state)

        if recognition is not None:
            scored = self._apply_opening_adjustments(scored, recognition)

        return max(scored, key=lambda x: x[1])[0]

    def score_move(self, board: BoardState, move: dict) -> float:
        """
        Rate `move` relative to all legal moves from 0.0 (worst) to 1.0 (best).

        Used by the LLM commentary system: a score below the configured threshold
        triggers a MillsLLM comment on the human's move.
        """
        moves = get_all_legal_moves(board)
        if not moves:
            return 0.5

        depth = max(2, _DEPTH_TABLE.get(self.difficulty, 9) - 1)
        scored = self._score_all(board, moves, depth)

        move_key = (move.get("from"), move["to"], move.get("capture"))
        my_score = next(
            (s for m, s in scored
             if (m.get("from"), m["to"], m.get("capture")) == move_key),
            None,
        )
        if my_score is None:
            return 0.0

        all_s = [s for _, s in scored]
        lo, hi = min(all_s), max(all_s)
        if hi == lo:
            return 1.0
        return (my_score - lo) / (hi - lo)

    # ── Opening book integration ──────────────────────────────────────────────

    def _apply_opening_adjustments(
        self,
        scored: list[tuple[dict, int]],
        recognition,
        book_bonus: float = 0.2,
        blunder_penalty: float = 0.3,
    ) -> list[tuple[dict, int]]:
        """Apply opening-book bonus/penalty to a scored move list."""
        if recognition.status in ("novel", "inactive"):
            return scored

        all_scores = [s for _, s in scored]
        lo, hi = min(all_scores), max(all_scores)
        scale = max(1, hi - lo)

        book_dest = None
        if recognition.book_move:
            # book_move may be "d2" (placement) or "a4-a7" (movement)
            book_dest = recognition.book_move.split("-")[-1].split("x")[0]

        blunder_dests = set()
        for b in (recognition.common_blunders or []):
            blunder_dests.add(b.split("-")[-1].split("x")[0])

        adjusted = []
        for move, raw in scored:
            dest = move.get("to", "")
            delta = 0
            if book_dest and dest == book_dest:
                delta += book_bonus * scale
            if dest in blunder_dests:
                delta -= blunder_penalty * scale
            adjusted.append((move, raw + delta))
        return adjusted

    # ── Internals ─────────────────────────────────────────────────────────────

    def _root_search(self, board: BoardState, depth: int) -> Tuple[dict, int]:
        """Search all root moves and return (best_move, best_score)."""
        moves = get_all_legal_moves(board)
        self._nodes = 0
        best_move = moves[0]
        best_score = -INF
        alpha = -INF

        for move in moves:
            nb = board.apply_move(move)
            score = -self._negamax(nb, depth - 1, -INF, -alpha)
            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score

        return best_move, best_score

    def _negamax(
        self,
        board: BoardState,
        depth: int,
        alpha: int,
        beta: int,
        endgame_state=None,
    ) -> int:
        """
        Negamax with alpha-beta pruning.
        Returns score from board.turn's perspective (higher = better for board.turn).
        """
        self._nodes += 1

        terminal, _ = is_terminal(board)
        if terminal:
            return -(INF - depth)

        if depth == 0:
            return evaluate(board, board.turn, endgame_state, self.force_aggressive)

        moves = get_all_legal_moves(board)
        if not moves:
            return -(INF - depth)

        value = -INF
        for move in moves:
            nb = board.apply_move(move)
            score = -self._negamax(nb, depth - 1, -beta, -alpha, endgame_state)
            if score > value:
                value = score
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value

    def _score_all(
        self, board: BoardState, moves: list, depth: int, endgame_state=None
    ) -> list[tuple[dict, int]]:
        """Score every move in `moves` and return [(move, score), ...]."""
        self._nodes = 0
        results = []
        for move in moves:
            nb = board.apply_move(move)
            score = -self._negamax(nb, depth - 1, -INF, INF, endgame_state)
            results.append((move, score))
        return results

    def _pick_blunder(self, board: BoardState, moves: list) -> dict:
        """
        Select a deliberately poor move from the bottom quartile of scored moves.
        Used by blunder mode to create teachable mistakes.
        """
        depth = max(2, _DEPTH_TABLE.get(self.difficulty, 9) - 1)
        scored = self._score_all(board, moves, depth)
        scored.sort(key=lambda x: x[1])  # ascending: worst first
        cutoff = max(1, len(scored) // 4)
        worst = scored[:cutoff]
        return random.choice(worst)[0]

    def _iterative_deepen(self, board: BoardState, time_limit: float = 10.0) -> dict:
        """Iterative deepening up to `time_limit` seconds."""
        start = time.time()
        moves = get_all_legal_moves(board)
        best_move = moves[0]
        for depth in range(2, 20):
            if time.time() - start >= time_limit * 0.8:
                break
            move, _ = self._root_search(board, depth)
            best_move = move
        return best_move

    def position_eval(self, board: BoardState) -> float:
        """
        Return tanh-normalised score in (-1, +1): positive = White winning.
        Uses phase-specific scale so each game stage reads meaningfully.
        """
        import math
        from .heuristics import evaluate as _eval, TANH_SCALE
        from game.rules import is_terminal, get_game_phase
        terminal, winner = is_terminal(board)
        if terminal:
            return 1.0 if winner == "W" else (-1.0 if winner == "B" else 0.0)
        w_score = _eval(board, "W")
        b_score = _eval(board, "B")
        raw   = w_score - b_score
        phase = get_game_phase(board, board.turn)
        scale = TANH_SCALE.get(phase, 180)
        return math.tanh(raw / scale)
