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

from game.board import ADJACENCY, MILLS, BoardState
from game.rules import get_all_legal_moves, is_terminal
from .heuristics import INF, evaluate, HeuristicWeights, DEFAULT_WEIGHTS, tactical_move_bonus


def _immediate_mill_threats(board: BoardState) -> set[str]:
    """Return empty squares where the opponent can close a mill in exactly 1 move.

    In fly phase the opponent can reach any empty square, so every 2-config is
    an immediate threat.  In move phase only 2-configs where an opponent piece is
    adjacent to the empty closing square count.
    """
    opp = "B" if board.turn == "W" else "W"
    opp_placed = board.pieces_placed.get(opp, 0)
    opp_in_fly = opp_placed >= 9 and board.pieces_on_board[opp] <= 3

    threats: set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(opp) == 2 and vals.count("") == 1:
            empty = next(p for p in mill if board.positions[p] == "")
            if opp_in_fly:
                threats.add(empty)
            elif opp_placed >= 9:  # move phase: need adjacent opp piece
                if any(board.positions[nb] == opp for nb in ADJACENCY[empty]):
                    threats.add(empty)
    return threats


def _order_moves(board: BoardState, moves: list) -> list:
    """Sort moves so the most urgent are tried first (better alpha-beta pruning).

    Priority 0 — close own mill (immediate win/capture) OR create a fork
                 (land on a diamond square — closing 2+ own 2-configs simultaneously).
    Priority 1 — block opponent mill (prevent their immediate threat).
    Priority 2 — all other moves.

    In fly phase with ~54 legal moves per side, this ensures blocking/closing
    moves are evaluated before the search deadline, so force_move returns a
    tactically sound choice even if the full tree isn't searched.
    """
    from game.rules import get_game_phase
    color = board.turn
    opp = "B" if color == "W" else "W"

    close: set[str] = set()
    block: set[str] = set()
    closing_count: dict[str, int] = {}
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        c = vals.count(color)
        o = vals.count(opp)
        e = vals.count("")
        if c == 2 and e == 1:
            empty = next(p for p in mill if board.positions[p] == "")
            close.add(empty)
            closing_count[empty] = closing_count.get(empty, 0) + 1
        if o == 2 and e == 1:
            block.add(next(p for p in mill if board.positions[p] == ""))

    # In fly phase, also prioritize moves to diamond squares (fork creation:
    # landing on a square that simultaneously closes 2+ own 2-configs).
    if get_game_phase(board, color) == "fly":
        for sq, cnt in closing_count.items():
            if cnt >= 2:
                close.add(sq)  # diamond squares join priority-0 (already in close)

    if not close and not block:
        return moves  # nothing to prioritize — skip the pass

    p0, p1, p2 = [], [], []
    for m in moves:
        t = m["to"]
        if t in close:
            p0.append(m)
        elif t in block:
            p1.append(m)
        else:
            p2.append(m)
    return p0 + p1 + p2

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
    difficulty : int [1-10]
        Search depth / time budget.  Difficulty 5+ uses iterative deepening.
    blunder_probability : float [0.0-1.0]
        Probability of playing a deliberately bad move each turn.
        0.0 = always plays best; 1.0 = always blunders.
        Bad moves are drawn from the bottom quartile of legal-move scores.
    use_mcts : bool
        When True, MCTS replaces negamax for the main move decision.
        Time budget is taken from _TIME_LIMIT[difficulty] (default 10 s).
    value_net : ValueNet | None
        Optional trained value network passed to MCTS as the leaf evaluator.
        Loaded automatically from data/value_net.npz by the web app when present.
    """

    def __init__(
        self,
        color: str = "B",
        difficulty: int = 3,
        blunder_probability: float = 0.0,
        weights: HeuristicWeights | None = None,
        use_mcts: bool = False,
        value_net=None,
    ) -> None:
        self.color = color
        self.difficulty = max(1, min(10, difficulty))
        self.blunder_probability = max(0.0, min(1.0, blunder_probability))
        self._weights: HeuristicWeights = weights if weights is not None else DEFAULT_WEIGHTS
        self._nodes = 0
        self._deadline: float = math.inf   # set by _iterative_deepen; checked in _negamax
        self.use_mcts = use_mcts
        self._value_net = value_net
        self._mcts = None
        if use_mcts:
            from .mcts import MCTS
            time_budget = _TIME_LIMIT.get(self.difficulty, 10.0)
            self._mcts = MCTS(
                color=color,
                time_limit=time_budget,
                weights=self._weights,
                value_net=value_net,
            )
        self._force_stop: bool = False     # set by force_stop(); cleared by choose_move()
        self.last_was_blunder: bool = False   # flag readable by Coordinator / MillsLLM
        self.force_aggressive: bool = False   # when True, disables fly-sacrifice heuristic
        # Position-specific move bans: board_fen → set of banned notations.
        # A ban only applies when the board is in the exact state it was in when
        # the move was marked bad; if any piece moves or is captured the position
        # key changes and the move becomes legal again.
        self._pos_bans: dict[str, set[str]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def ban_move(self, notation: str, board_fen: str) -> None:
        """Ban `notation` from this exact board position only.

        If any piece moves or is captured the FEN changes and the ban
        no longer applies — the move is valid again from the new position.
        """
        self._pos_bans.setdefault(board_fen, set()).add(notation)

    def reset_game_bans(self) -> None:
        """Clear all per-game move bans (call when a new game starts)."""
        self._pos_bans.clear()

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
        top_n: int = 1,             # if >1, pick randomly from top-N moves (self-play noise)
        fast_early_game: bool = False,  # skip the 4s early-game budget (self-play mode)
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

        # Mandatory block: if the opponent has an immediate mill threat (closeable
        # in exactly one move), restrict candidates to blocking moves only.
        # In fly phase every 2-config is an immediate threat regardless of
        # adjacency; in move phase only 2-configs with an adjacent opponent piece count.
        threats = _immediate_mill_threats(board)
        if threats:
            blocking = [m for m in moves if m["to"] in threats]
            if blocking:
                moves = blocking

        # Position-specific move bans (set via bad-move button): filter AFTER mandatory
        # block so a banned blocking move can still be played if it's the only way to block.
        _banned_here = self._pos_bans.get(board.to_fen_string())
        if _banned_here:
            non_banned = [m for m in moves if self._move_notation(m) not in _banned_here]
            if non_banned:  # safety: never reduce to zero legal moves
                moves = non_banned

        # Blunder mode: occasionally play a bad move on purpose
        if self.blunder_probability > 0.0 and random.random() < self.blunder_probability:
            blunder = self._pick_blunder(board, moves)
            self.last_was_blunder = True
            return blunder

        self.last_was_blunder = False

        # MCTS path: delegate to Monte Carlo Tree Search when enabled.
        if self._mcts is not None:
            time_budget = _TIME_LIMIT.get(self.difficulty, 10.0)
            if fast_early_game:
                time_budget = 2.0
            deadline = time.time() + time_budget
            return self._mcts.choose_move(board, deadline=deadline)

        # Early-game fast path: while few pieces are on the board the tree is
        # tiny — cap the search to a short budget regardless of difficulty.
        # Early-game cap: for time-limited difficulties only (5+), use a shorter
        # budget before enough pieces are placed for the full time budget to be useful.
        # Fixed-depth difficulties (1–4) don't need this; their tree is already small.
        total_on_board = sum(board.pieces_on_board.values())
        if (total_on_board < _EARLY_GAME_PIECE_THRESHOLD
                and not fast_early_game
                and self.difficulty in _TIME_LIMIT):
            return self._iterative_deepen(
                board, _EARLY_GAME_TIME,
                recognition=recognition, trajectory_hints=trajectory_hints,
                top_n=top_n, moves=moves,
            )

        if self.difficulty in _TIME_LIMIT:
            time_budget = 2.0 if fast_early_game else _TIME_LIMIT[self.difficulty]
            return self._iterative_deepen(
                board, time_budget,
                recognition=recognition, trajectory_hints=trajectory_hints,
                top_n=top_n, moves=moves,
            )

        depth = _DEPTH_TABLE[self.difficulty]

        # Deeper search in endgame for better tactical accuracy.
        # Skip in fast self-play mode to keep per-move time bounded.
        if endgame_state is not None and endgame_state.active and not fast_early_game:
            depth += 2 if endgame_state.deep else 1

        _has_hard_bans = bool(trajectory_hints and any(
            d <= -1.0 for d in trajectory_hints.values()
        ))
        use_adjustments = (
            (recognition is not None and recognition.status not in ("novel", "inactive"))
            or (bool(trajectory_hints) and self._weights.opening_adherence > 0)
        ) or _has_hard_bans
        if use_adjustments:
            scored = self._score_all(board, moves, depth, endgame_state=endgame_state)
            if recognition is not None:
                scored = self._apply_opening_adjustments(scored, recognition)
            if trajectory_hints:
                scored = self._apply_trajectory_hints(scored, trajectory_hints)
            if top_n > 1:
                scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
                return random.choice(scored_sorted[:top_n])[0]
            return max(scored, key=lambda x: x[1])[0]

        move, _ = self._root_search(board, depth, top_n=top_n, moves=moves)
        return move

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

    _HARD_BAN_THRESHOLD = -1.0  # sentinel from TrajectoryDB for user-marked bad moves

    def _apply_trajectory_hints(
        self,
        scored: list[tuple[dict, int]],
        hints: dict[str, float],
    ) -> list[tuple[dict, int]]:
        """Apply trajectory-database score deltas to a scored move list.

        Deltas in [-0.5, +0.5] are statistical hints scaled by opening_adherence.
        Delta == -1.0 is a hard-ban sentinel from the user's bad-move button:
        the move receives -INF+1 so it is never chosen regardless of adherence.
        """
        if not hints:
            return scored
        adherence = self._weights.opening_adherence
        scale = int(3000 * adherence / 100) if adherence > 0 else 0
        adjusted = []
        for move, raw in scored:
            notation = self._move_notation(move)
            delta    = hints.get(notation, 0.0)
            if delta <= self._HARD_BAN_THRESHOLD:
                adjusted.append((move, -INF + 1))  # always last; still legal
                continue
            bonus = int(delta * scale) if scale else 0
            adjusted.append((move, raw + bonus))
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

    def _root_search(self, board: BoardState, depth: int,
                     top_n: int = 1, moves: list | None = None) -> Tuple[dict, int]:
        """Search all root moves and return (best_move, best_score).
        When top_n > 1, pick randomly from the top-N scoring moves.
        If _SearchAbort fires (force_stop called mid-search), returns the
        best move found so far rather than propagating the exception.
        `moves` may be pre-filtered (e.g. mandatory-block constraint); if None
        all legal moves are used."""
        if moves is None:
            moves = get_all_legal_moves(board)
        moves = _order_moves(board, moves)
        self._nodes = 0
        best_move = moves[0]
        best_score = -INF
        alpha = -INF
        all_scored: list[Tuple[dict, int]] = []

        for move in moves:
            nb = board.apply_move(move)
            try:
                score = -self._negamax(nb, depth - 1, -INF, -alpha)
            except _SearchAbort:
                break
            score += tactical_move_bonus(board, nb, self.color, self._weights)
            if top_n > 1:
                all_scored.append((move, score))
            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score

        if top_n > 1 and all_scored:
            top = sorted(all_scored, key=lambda x: x[1], reverse=True)[:top_n]
            best_move = random.choice(top)[0]
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

        # Sort at upper levels only — biggest benefit to alpha-beta, negligible overhead
        if depth >= 2:
            moves = _order_moves(board, moves)

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

    _BLUNDER_TIME = 2.0   # hard cap for blunder scoring — ranking doesn't need deep search
    _BLUNDER_DEPTH = 3    # shallow depth is enough to distinguish good/bad moves

    def _pick_blunder(self, board: BoardState, moves: list) -> dict:
        """
        Select a deliberately poor move from the bottom quartile of scored moves.
        Uses a shallow fixed depth (3) with a hard time cap — blunders just need
        to avoid picking the obviously best move, not evaluate perfectly.
        """
        self._deadline = time.time() + self._BLUNDER_TIME
        scored = self._score_all(board, moves, self._BLUNDER_DEPTH)
        self._deadline = math.inf
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
        top_n: int = 1,
        moves: list | None = None,
    ) -> dict:
        """
        Iterative deepening up to `time_limit` seconds.

        When opening recognition or trajectory hints are active, scores every
        root move at each depth so the adjustments can be applied before
        picking the best.  Otherwise uses the faster _root_search path.
        `moves` may be pre-filtered (e.g. mandatory-block constraint); if None
        all legal moves are used.
        """
        self._deadline = time.time() + time_limit
        if moves is None:
            moves = get_all_legal_moves(board)
        best_move     = moves[0]
        _has_hard_bans = bool(trajectory_hints and any(
            d <= -1.0 for d in trajectory_hints.values()
        ))
        use_adjustments = (
            (
                recognition is not None
                and recognition.status not in ("novel", "inactive")
            ) or (bool(trajectory_hints) and self._weights.opening_adherence > 0)
        ) or _has_hard_bans

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
                    if top_n > 1:
                        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
                        best_move = random.choice(scored_sorted[:top_n])[0]
                    else:
                        best_move = max(scored, key=lambda x: x[1])[0]
                else:
                    move, _ = self._root_search(board, depth, top_n=top_n, moves=moves)
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
