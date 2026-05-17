"""
ai/game_ai.py — Minimax AI using negamax + alpha-beta pruning.

GameAI plays the computer's side.  It also exposes score_move() for rating
human moves, used by the LLM commentary system to decide whether to comment.

Blunder mode: set blunder_probability > 0 to make the AI occasionally play a
deliberately poor move so the human can practise exploiting mistakes.
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional, Tuple


class _SearchAbort(Exception):
    """Raised inside _negamax when the search deadline has passed."""

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
from .heuristics import INF, evaluate, HeuristicWeights, DEFAULT_WEIGHTS, tactical_move_bonus

# Fixed-depth table for quick levels (1–4): search completes fast so no time cap needed.
_DEPTH_TABLE = {1: 2, 2: 3, 3: 4, 4: 5}

# Iterative-deepening time budgets for levels 5–10.
# Levels 6–8 are promoted from fixed-depth so force_move never fires mid-search.
_TIME_LIMIT = {
    5: 15.0,   # was 10 s
    6: 24.0,   # was fixed depth-7 (no time cap)
    7: 36.0,   # was fixed depth-8
    8: 60.0,   # was fixed depth-9
    9: 60.0,   # was 20 s
    10: 90.0,  # was 45 s
}

# While fewer than this many pieces are on the board in total, use a short
# time budget regardless of difficulty — the tree is tiny and deep search wastes time.
_EARLY_GAME_PIECE_THRESHOLD = 10  # covers roughly the first 5 placements per side
_EARLY_GAME_TIME            = 4.0  # seconds


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
        weights: HeuristicWeights | None = None,
    ) -> None:
        self.color = color
        self.difficulty = max(1, min(10, difficulty))
        self.blunder_probability = max(0.0, min(1.0, blunder_probability))
        self._weights: HeuristicWeights = weights if weights is not None else DEFAULT_WEIGHTS
        self._nodes = 0
        self._deadline: float = math.inf   # set by _iterative_deepen; checked in _negamax
        self._force_stop: bool = False     # set by force_stop(); cleared by choose_move()
        self.last_was_blunder: bool = False   # flag readable by Coordinator / MillsLLM
        self.force_aggressive: bool = False   # when True, disables fly-sacrifice heuristic

    # ── Public API ────────────────────────────────────────────────────────────

    def force_stop(self) -> None:
        """Interrupt any running search immediately; _negamax raises _SearchAbort.
        Also sets _force_stop so the subsequent score_move() returns immediately.
        """
        self._force_stop = True
        self._deadline   = 0.0

    def choose_move(
        self,
        board: BoardState,
        recognition=None,           # RecognitionResult  — Stage 4
        endgame_state=None,         # EndgameState        — Stage 5
        trajectory_hints=None,      # dict[str, float] from TrajectoryDB.query()
    ) -> dict:
        """Return the best (or deliberately bad) legal move dict for self.color."""
        self._force_stop = False
        self._deadline   = math.inf  # reset any prior force_stop() effect
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

        # Early-game fast path: while few pieces are on the board the tree is
        # tiny — cap the search to a short budget regardless of difficulty.
        total_on_board = sum(board.pieces_on_board.values())
        if total_on_board < _EARLY_GAME_PIECE_THRESHOLD:
            return self._iterative_deepen(
                board, _EARLY_GAME_TIME,
                recognition=recognition, trajectory_hints=trajectory_hints,
            )

        if self.difficulty in _TIME_LIMIT:
            return self._iterative_deepen(
                board, _TIME_LIMIT[self.difficulty],
                recognition=recognition, trajectory_hints=trajectory_hints,
            )

        depth = _DEPTH_TABLE[self.difficulty]

        # Deeper search in endgame for better tactical accuracy.
        if endgame_state is not None and endgame_state.active:
            depth += 2 if endgame_state.deep else 1

        scored = self._score_all(board, moves, depth, endgame_state=endgame_state)

        if recognition is not None:
            scored = self._apply_opening_adjustments(scored, recognition)

        if trajectory_hints:
            scored = self._apply_trajectory_hints(scored, trajectory_hints)

        return max(scored, key=lambda x: x[1])[0]

    # Time budget for score_move() — kept short because it is only used for relative
    # ranking (is this move good or bad?), not for the actual played move.
    _SCORE_TIME = 3.0

    def score_move(self, board: BoardState, move: dict) -> float:
        """
        Rate `move` relative to all legal moves from 0.0 (worst) to 1.0 (best).

        Used by the LLM commentary system: a score below the configured threshold
        triggers a MillsLLM comment on the human's move.
        """
        if self._force_stop:
            return 0.5   # force-stopped; skip scoring rather than adding more delay

        moves = get_all_legal_moves(board)
        if not moves:
            return 0.5

        total_on_board = sum(board.pieces_on_board.values())
        if total_on_board < _EARLY_GAME_PIECE_THRESHOLD:
            depth = 3
        else:
            depth = max(2, _DEPTH_TABLE.get(self.difficulty, 9) - 1)

        # Hard cap: score_move must finish within _SCORE_TIME seconds.
        self._deadline = time.time() + self._SCORE_TIME
        scored = self._score_all(board, moves, depth)
        self._deadline = math.inf

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

    # ── Opening book + trajectory integration ────────────────────────────────

    @staticmethod
    def _move_notation(move: dict) -> str:
        """Convert a move dict to its notation string (matches coordinator _move_str)."""
        s = f"{move['from']}-{move['to']}" if move.get("from") else move.get("to", "")
        if move.get("capture"):
            s += f"x{move['capture']}"
        return s

    def _apply_trajectory_hints(
        self,
        scored: list[tuple[dict, int]],
        hints: dict[str, float],
    ) -> list[tuple[dict, int]]:
        """Apply trajectory-database score deltas to a scored move list.

        `hints` maps move notation → float in [-0.5, +0.5] where +0.5 means
        that move won 100 % of sampled games for the current colour.

        The absolute bonus at 50 % adherence is ±750, growing to ±1500 at
        100 % — smaller than the opening-book bonus so book lines still
        dominate in the opening phase, while trajectory hints fill the gap
        in the mid/late game where the opening book has no opinion.
        """
        if not hints:
            return scored
        adherence = self._weights.opening_adherence
        if adherence == 0:
            return scored
        scale = int(3000 * adherence / 100)   # max ±1500 at 50 % adherence
        adjusted = []
        for move, raw in scored:
            notation = self._move_notation(move)
            delta    = hints.get(notation, 0.0)
            adjusted.append((move, raw + int(delta * scale)))
        return adjusted

    def _apply_opening_adjustments(
        self,
        scored: list[tuple[dict, int]],
        recognition,
    ) -> list[tuple[dict, int]]:
        """Apply opening-book bonus/penalty to a scored move list.

        Uses absolute bonuses proportional to the opening_adherence slider so
        the book preference always outweighs tactical noise at high adherence.
        """
        if recognition.status in ("novel", "inactive"):
            return scored

        adherence = self._weights.opening_adherence
        if adherence == 0:
            return scored

        # Absolute bonus scales linearly with adherence: 50 % -> 1500, 100 % -> 3000
        book_bonus_abs    = int(3000 * adherence / 100)
        blunder_penalty_abs = int(1500 * adherence / 100)

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
                delta += book_bonus_abs
            if dest in blunder_dests:
                delta -= blunder_penalty_abs
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
            nb    = board.apply_move(move)
            score = -self._negamax(nb, depth - 1, -INF, -alpha)
            score += tactical_move_bonus(board, nb, self.color, self._weights)
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
        Raises _SearchAbort when the search deadline is exceeded.
        """
        self._nodes += 1
        # Check deadline every 4096 nodes to avoid time.time() call overhead.
        if self._nodes & 0xFFF == 0 and time.time() >= self._deadline:
            raise _SearchAbort()

        terminal, _ = is_terminal(board)
        if terminal:
            return -(INF - depth)

        if depth == 0:
            return evaluate(board, board.turn, endgame_state, self.force_aggressive, self._weights)

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
        """Score every move in `moves` and return [(move, score), ...].

        If _SearchAbort is raised mid-loop (force_stop() called), unscored moves
        receive the worst score seen so far so max() still picks the best partial result.
        """
        self._nodes = 0
        results = []
        for i, move in enumerate(moves):
            nb = board.apply_move(move)
            try:
                score = -self._negamax(nb, depth - 1, -INF, INF, endgame_state)
                score += tactical_move_bonus(board, nb, self.color, self._weights)
                results.append((move, score))
            except _SearchAbort:
                worst = min(s for _, s in results) if results else -INF
                for remaining in moves[i:]:
                    results.append((remaining, worst))
                break
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

    def _iterative_deepen(
        self,
        board: BoardState,
        time_limit: float = 10.0,
        recognition=None,
        trajectory_hints=None,
    ) -> dict:
        """
        Iterative deepening up to `time_limit` seconds.

        When opening recognition or trajectory hints are active, scores every
        root move at each depth so the adjustments can be applied before
        picking the best.  Otherwise uses the faster _root_search path.
        """
        self._deadline = time.time() + time_limit
        moves         = get_all_legal_moves(board)
        best_move     = moves[0]
        use_adjustments = (
            (
                recognition is not None
                and recognition.status not in ("novel", "inactive")
            ) or bool(trajectory_hints)
        ) and self._weights.opening_adherence > 0

        for depth in range(2, 20):
            if time.time() >= self._deadline:
                break
            try:
                if use_adjustments:
                    scored = self._score_all(board, moves, depth)
                    if recognition is not None:
                        scored = self._apply_opening_adjustments(scored, recognition)
                    if trajectory_hints:
                        scored = self._apply_trajectory_hints(scored, trajectory_hints)
                    best_move = max(scored, key=lambda x: x[1])[0]
                else:
                    move, _ = self._root_search(board, depth)
                    best_move = move      # only update if depth completed cleanly
            except _SearchAbort:
                break                     # deadline hit mid-depth; keep previous best
        self._deadline = math.inf
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
