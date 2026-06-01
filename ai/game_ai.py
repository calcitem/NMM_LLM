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

from game.board import ADJACENCY, MILLS, POSITIONS, BoardState
from game.rules import get_all_legal_moves, is_terminal
from .heuristics import INF, evaluate, HeuristicWeights, DEFAULT_WEIGHTS, tactical_move_bonus, _sealed_two_configs, _dual_connected_mill_alert
from .transposition_table import TranspositionTable, EXACT, LOWER_BOUND, UPPER_BOUND
from .board_symmetry import SYM_INVERSE, transform_notation as _transform_notation

# B-73: value network scale — maps VN output (-1, 1) to heuristic score range.
# A VN score of +1.0 (certain win) maps to ±3000, comparable to a large positional advantage.
_VN_SCALE = 3000
# SE-11b/11c: how many opponent plies from root receive trajectory extension and VN ordering.
_MAX_OPP_PLIES = 2


def _stm_can_close_mill(board: BoardState, color: str) -> bool:
    """True if color can close a mill on the next placement or slide."""
    return any(
        [board.positions[p] for p in mill].count(color) == 2
        and [board.positions[p] for p in mill].count("") == 1
        for mill in MILLS
    )


def _immediate_mill_threats(board: BoardState) -> set[str]:
    """Return empty squares where the opponent can close a mill in exactly 1 move.

    In fly phase the opponent can reach any empty square, so every 2-config is
    an immediate threat.  In move phase only 2-configs where an opponent piece is
    adjacent to the empty closing square count.  In placement phase, a fork
    (≥2 simultaneous opponent 2-configs) makes all closing squares mandatory
    blocking targets — a single response cannot stop both.

    Single opponent threat (placement or move): if the side to move can close their
    own mill this turn, no mandatory block restriction is applied — closing with
    capture is at least as urgent as occupying the opponent's closing square (B-66).
    """
    opp = "B" if board.turn == "W" else "W"
    stm = board.turn
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

    # Move phase: single threat + own mill available → do not restrict (B-66).
    if opp_placed >= 9 and not opp_in_fly and len(threats) == 1:
        if _stm_can_close_mill(board, stm):
            threats.clear()

    # Placement phase: any opponent 2-config is an immediate threat — restrict STM
    # to blocking squares.  Fork (≥2 simultaneous threats) always restricts.
    # Single threat: carveout allows STM to close their own mill instead.
    if opp_placed < 9:
        closing = [
            next(p for p in mill if board.positions[p] == "")
            for mill in MILLS
            if ([board.positions[p] for p in mill].count(opp) == 2
                and [board.positions[p] for p in mill].count("") == 1)
        ]
        if len(closing) >= 2:
            threats.update(closing)
        elif closing:
            if not _stm_can_close_mill(board, stm):
                threats.update(closing)

    return threats


def _pinned_fly_squares(board: BoardState, color: str) -> frozenset:
    """Return own squares that are the sole blocker of an opponent 2-config.

    When in fly phase, each own piece can jump anywhere — but if an own piece
    sits in the closing square of an opponent 2-config (opp has the other two
    slots), vacating it hands the opponent an immediate mill closure.  Those
    squares are "pinned": the piece must not move unless no other move exists.
    """
    opp = "B" if color == "W" else "W"
    pinned: set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(opp) == 2 and vals.count(color) == 1:
            our_sq = next(p for p in mill if board.positions[p] == color)
            pinned.add(our_sq)
    return frozenset(pinned)


def _pinned_move_squares(board: BoardState, color: str) -> frozenset:
    """Own squares that are the sole blocker of an opponent 2-config AND have an
    adjacent opponent piece ready to slide in immediately (move-phase pin rule).

    Unlike _pinned_fly_squares (which fires whenever opp has a 2-config), this
    requires an adjacent opp piece so the threat is *immediately* cashable in
    move phase (opp must be able to slide one step into the vacated square).
    """
    opp = "B" if color == "W" else "W"
    pinned: set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(opp) == 2 and vals.count(color) == 1:
            our_sq = next(p for p in mill if board.positions[p] == color)
            if any(board.positions.get(nb, "") == opp for nb in ADJACENCY.get(our_sq, [])):
                pinned.add(our_sq)
    return frozenset(pinned)


def _squeeze_targets(board: BoardState) -> set[str]:
    """Return empty squares that are the last escape route of an opponent piece.

    When an opponent piece has exactly one empty neighbour, occupying that square
    would fully block it.  Herding moves to these squares are searched at the
    same priority as blocking opponent mills — they represent an equally urgent
    path to a win (forcing zero-mobility blockade rather than a mill capture).
    Only applies in move phase; fly-phase pieces can jump to any empty square so
    adjacency blocking is irrelevant.
    """
    from game.rules import get_game_phase
    opp = "B" if board.turn == "W" else "W"
    if get_game_phase(board, board.turn) != "move":
        return set()
    targets: set[str] = set()
    for pos in POSITIONS:
        if board.positions[pos] == opp:
            empties = [n for n in ADJACENCY[pos] if board.positions[n] == ""]
            if len(empties) == 1:
                targets.add(empties[0])
    return targets


def _order_moves(board: BoardState, moves: list, killers=None, history=None) -> list:
    """Sort moves so the most urgent are tried first (better alpha-beta pruning).

    Priority 0 — close own mill (immediate win/capture) OR create a fork
                 (land on a diamond square — closing 2+ own 2-configs simultaneously).
    Priority 1 — block opponent mill (prevent their immediate threat)
                 OR occupy the last escape square of an opponent piece (herding).
    Priority K — killer moves: quiet moves that caused beta cutoffs at this depth
                 in sibling branches (SE-2).
    Priority 2 — all other moves, sorted descending by history score (SE-3).

    In fly phase with ~54 legal moves per side, this ensures blocking/closing
    moves are evaluated before the search deadline, so force_move returns a
    tactically sound choice even if the full tree isn't searched.
    """
    from game.rules import get_game_phase
    color = board.turn
    opp = "B" if color == "W" else "W"

    # Build killer set for O(1) lookup.
    killer_set: set = set()
    if killers:
        for k in killers:
            if k is not None:
                killer_set.add(k)

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

    # Squeeze moves: the last empty neighbour of a nearly-blocked opponent piece.
    # Searched with the same urgency as blocking a mill threat.
    squeeze = _squeeze_targets(board)
    block |= squeeze

    # B-55: in placement phase, add the closing squares of opponent 2-configs that would
    # complete a second mill sharing a square with an already-closed opponent mill.
    # Two interconnected cycling mills are nearly unbeatable; block the second formation
    # with the same urgency as blocking any direct mill threat.
    if get_game_phase(board, color) == "place":
        block |= set(_dual_connected_mill_alert(board, opp))

    if not close and not block and not killer_set and not history:
        return moves  # nothing to prioritize — skip the pass

    # B-59: P0.5 — moves that create a new sealed 2-config (uncontestable forced mill).
    # Only computed in move phase when there are no direct mill closes (P0) — those
    # dominate regardless, and the extra apply_move calls would be wasted.
    # Covers all 16 MILLS (not just inner ring): any sealed pattern is elevated.
    sealed_creates: set = set()  # stores (from, to) tuples
    if get_game_phase(board, color) == "move" and not close:
        sealed_before = _sealed_two_configs(board, color)
        if sealed_before < len(MILLS):  # skip if already at max — nothing to gain
            for m in moves:
                nb = board.apply_move(m)
                if _sealed_two_configs(nb, color) > sealed_before:
                    sealed_creates.add((m.get("from"), m["to"]))

    p0, p05, p1, pk, p2 = [], [], [], [], []
    for m in moves:
        t = m["to"]
        if t in close:
            p0.append(m)
        elif (m.get("from"), t) in sealed_creates:
            p05.append(m)
        elif t in block:
            p1.append(m)
        elif (m.get("from"), t) in killer_set:
            pk.append(m)
        else:
            p2.append(m)
    if history and p2:
        p2.sort(key=lambda m: history.get((m.get("from"), m["to"]), 0), reverse=True)
    return p0 + p05 + p1 + pk + p2

# Fixed-depth table for quick levels (1–4): search completes fast so no time cap needed.
_DEPTH_TABLE = {1: 2, 2: 3, 3: 4, 4: 5}

# Iterative-deepening time budgets.
# Difficulties 1–4 previously used fixed depth.  SE-8 search extensions and SE-9
# quiescence can push effective depth 2–4 plies deeper than the nominal depth,
# turning fixed-depth levels into unbounded searches.  Adding them to _TIME_LIMIT
# caps the wall-clock cost while letting later iterations go deeper when time allows.
_TIME_LIMIT = {
    1: 0.3,    # ~depth 2–3; very fast
    2: 0.8,    # ~depth 3–4; fast
    3: 2.5,    # ~depth 4–5; moderate
    4: 6.0,    # ~depth 5–6; standard
    5: 15.0,
    6: 24.0,
    7: 36.0,
    8: 60.0,
    9: 60.0,
    10: 90.0,
}

# While fewer than this many pieces are on the board in total, use a short
# time budget regardless of difficulty — the tree is tiny and deep search wastes time.
_EARLY_GAME_PIECE_THRESHOLD = 10  # covers roughly the first 5 placements per side
_EARLY_GAME_TIME            = 2.0  # seconds — SE-8 extensions can push effective depth 2–4 plies
                                   # deeper than nominal, so the budget must be tighter


def _is_dead_placement(board: BoardState, move: dict) -> bool:
    """True when *move* places a piece on a square with no free (empty) neighbours.

    Such a square is permanently immobile: the piece can never slide away or
    form a mill by approach.  Mill-closing placements are exempted because
    they deliver immediate tactical value regardless of mobility.
    Movement moves (have a 'from' key) are never dead placements.
    """
    if move.get("from"):
        return False
    to = move["to"]
    free_nb = sum(1 for nb in ADJACENCY.get(to, []) if board.positions.get(nb) == "")
    if free_nb > 0:
        return False
    # Mill-closing exemption: placing here closes a mill right now.
    stm = board.turn
    for mill in MILLS:
        if to in mill and all(board.positions.get(sq) == stm or sq == to for sq in mill):
            return False
    return True


def _parse_book_move(book_move_str: str, legal_moves: list) -> dict | None:
    """Return the legal move dict that matches the book move notation, or None."""
    if not book_move_str:
        return None
    if "-" in book_move_str:
        parts = book_move_str.split("-", 1)
        from_pos = parts[0]
        to_pos   = parts[1].split("x")[0]
        return next(
            (m for m in legal_moves if m.get("from") == from_pos and m["to"] == to_pos),
            None,
        )
    to_pos = book_move_str.split("x")[0]
    return next(
        (m for m in legal_moves if not m.get("from") and m["to"] == to_pos),
        None,
    )


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
        fullgame_db=None,           # ai.fullgame_db.FullGameDB | None
        endgame_solved_db=None,     # ai.endgame_solved_db.EndgameSolvedDB | None
        neural_evaluator=None,      # ai.neural_evaluator.NeuralEvaluator | None
        override_time_budget: float | None = None,  # seconds; overrides _TIME_LIMIT for training
    ) -> None:
        self.color = color
        self.difficulty = max(1, min(10, difficulty))
        self.blunder_probability = max(0.0, min(1.0, blunder_probability))
        self._weights: HeuristicWeights = weights if weights is not None else DEFAULT_WEIGHTS
        self._fullgame_db = fullgame_db
        self._endgame_solved_db = endgame_solved_db
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
        self._tt = TranspositionTable()
        # SE-2: 2 killer moves per remaining-depth level (up to depth 32).
        # Each slot is (from_sq, to_sq) or None.
        self._killers: list[list] = [[None, None] for _ in range(32)]
        # SE-3: global history table keyed by (from_sq, to_sq); value = Σ depth².
        self._history: dict = {}
        self._force_stop: bool = False     # set by force_stop(); cleared by choose_move()
        self.last_was_blunder: bool = False   # flag readable by Coordinator / MillsLLM
        self.last_thinking: str = ""          # short plain-English label for the chosen move
        self.force_aggressive: bool = False   # when True, disables fly-sacrifice heuristic
        # Set True by Coordinator when opponent's last move scored below poor_move_threshold.
        # Amplifies the placement busy-chain bonus so the AI exploits passive opponent play.
        self._opp_last_weak: bool = False
        # Position-specific move bans: board_fen → set of banned notations.
        # A ban only applies when the board is in the exact state it was in when
        # the move was marked bad; if any piece moves or is captured the position
        # key changes and the move becomes legal again.
        self._pos_bans: dict[str, set[str]] = {}
        # SE-11: trajectory DB + game notations for opponent-frequency-based extension.
        # Set by choose_move each call; used in _root_search and _score_all.
        self._trajectory_db = None
        self._game_notations: list = []
        self._move_path_buf: list = []  # SE-11b: shared push/pop path buffer for _negamax recursion
        # Neural leaf evaluator (replaces heuristic evaluate() at depth-0 leaves).
        self._neural_evaluator = neural_evaluator
        # Per-instance time-budget override (used during training to keep games fast).
        self._override_time_budget = override_time_budget

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
        force_book_early: bool = False, # force book move for first 2 AI placements
        fullgame_db=None,           # ai.fullgame_db.FullGameDB | None — overrides self._fullgame_db
        trajectory_db=None,         # TrajectoryDB | None — SE-11 opponent freq lookup
        game_notations=None,        # list[str] | None — current game move notations (SE-11)
    ) -> dict:
        """Return the best (or deliberately bad) legal move dict for self.color.

        When a fullgame_db is available (via constructor or parameter), resolved
        positions return the DB best move directly; unresolved positions blend DB
        score deltas into trajectory_hints before the normal search.  Misses fall
        back transparently.
        """
        self._force_stop = False
        self._deadline   = math.inf  # reset any prior force_stop() effect
        self.last_thinking = ""       # reset thinking trace
        self._tt.clear()
        self._killers = [[None, None] for _ in range(32)]
        self._history = {}
        self._trajectory_db = trajectory_db            # SE-11
        self._game_notations = list(game_notations) if game_notations else []  # SE-11
        moves = get_all_legal_moves(board)
        if not moves:
            return {}
        if len(moves) == 1:
            self.last_was_blunder = False
            return moves[0]

        # ── Optional retrograde endgame DB consultation ───────────────────
        # Consulted first (before fullgame_db) because WDL is exact.
        # Guard: both sides must have placed all 9 pieces AND each has ≤3 on board.
        _esdb = self._endgame_solved_db
        if _esdb is not None and _esdb.is_available():
            _w_on = board.pieces_on_board.get("W", 0)
            _b_on = board.pieces_on_board.get("B", 0)
            if (board.pieces_placed.get("W", 0) >= 9
                    and board.pieces_placed.get("B", 0) >= 9
                    and _w_on <= 3 and _b_on <= 3
                    and _w_on + _b_on <= 6):
                try:
                    _wdl = _esdb.query(board)
                except Exception:
                    _wdl = None
                if _wdl == "W":
                    self.last_was_blunder = False
                    for _move in moves:
                        _succ = board.apply_move(_move)
                        _succ_terminal, _ = is_terminal(_succ)
                        if _succ_terminal:
                            # Capture reduces opponent to 2 pieces → instant win
                            self.last_thinking = "endgame DB (win)"
                            return _move
                        try:
                            _succ_wdl = _esdb.query(_succ)
                        except Exception:
                            _succ_wdl = None
                        if _succ_wdl == "L":
                            self.last_thinking = "endgame DB (win)"
                            return _move
                    # fall through to heuristic if no winning continuation found
                elif _wdl == "L":
                    self.last_thinking = "endgame DB (loss/search)"
                    # fall through to heuristic search for most stubborn defence
                elif _wdl == "D":
                    self.last_thinking = "endgame DB (draw)"

        # ── Optional full-game DB consultation ────────────────────────────
        # Falls back to self._fullgame_db when no explicit parameter passed.
        _fgdb = fullgame_db if fullgame_db is not None else self._fullgame_db
        if _fgdb is not None and _fgdb.is_available():
            try:
                result = _fgdb.query(board)
            except Exception as exc:    # never let DB errors kill the AI
                logger_msg = f"fullgame_db query failed: {exc}"
                try:
                    import logging
                    logging.getLogger(__name__).warning(logger_msg)
                except Exception:
                    pass
                result = None
            if result is not None:
                # Resolved exact hit — return DB best move when legal.
                if result.outcome is not None and result.best_move_canonical:
                    best_notation = _fgdb.best_move(board)
                    if best_notation:
                        match = next(
                            (m for m in moves if self._move_notation(m) == best_notation),
                            None,
                        )
                        if match is not None:
                            self.last_was_blunder = False
                            self.last_thinking = "fullgame DB"
                            return match
                # Unresolved row — merge DB deltas into trajectory hints.
                db_hints = _fgdb.score_delta(board, self.color)
                if db_hints:
                    merged = dict(trajectory_hints or {})
                    for k, v in db_hints.items():
                        merged[k] = merged.get(k, 0.0) + v
                    trajectory_hints = merged

        # Mandatory block: if the opponent has an immediate mill threat (closeable
        # in exactly one move), restrict candidates to blocking moves only.
        # In fly phase every 2-config is an immediate threat regardless of
        # adjacency; in move phase only 2-configs with an adjacent opponent piece count.
        threats = _immediate_mill_threats(board)
        if threats:
            blocking = [m for m in moves if m["to"] in threats]
            if blocking:
                moves = blocking

        # Dead-placement filter: remove placements on squares with 0 free
        # neighbours (permanently immobile) unless they close a mill.
        # Applied AFTER mandatory-block so a forced dead block is kept.
        # The mill-closing exemption inside _is_dead_placement preserves
        # any move that delivers an immediate mill.
        if board.phase == "place":
            non_dead = [m for m in moves if not _is_dead_placement(board, m)]
            if non_dead:
                moves = non_dead

        # Position-specific move bans (set via bad-move button): filter AFTER mandatory
        # block so a banned blocking move can still be played if it's the only way to block.
        _banned_here = self._pos_bans.get(board.to_fen_string())
        if _banned_here:
            non_banned = [m for m in moves if self._move_notation(m) not in _banned_here]
            if non_banned:  # safety: never reduce to zero legal moves
                moves = non_banned

        # Fly-phase pin rule: don't vacate the sole blocker of an opponent 2-config.
        # Moving a pinned piece immediately gives the opponent a free mill closure.
        from game.rules import get_game_phase
        if get_game_phase(board, self.color) == "fly":
            pinned = _pinned_fly_squares(board, self.color)
            if pinned:
                unpinned = [m for m in moves if m.get("from") not in pinned]
                if unpinned:
                    moves = unpinned

        # Movement-phase pin rule: don't vacate the sole blocker of an opponent
        # 2-config when the opponent has an adjacent piece ready to slide in.
        # Harder constraint than fly-phase (adjacency required) but same spirit:
        # vacating the square hands the opponent an immediate mill closure.
        if get_game_phase(board, self.color) == "move":
            pinned = _pinned_move_squares(board, self.color)
            if pinned:
                unpinned = [m for m in moves if m.get("from") not in pinned]
                if unpinned:
                    moves = unpinned

        # Book forcing: early-game forcing (first 2 AI placements) or 100% adherence.
        # Applied after ban filtering so a banned book move is never forced.
        if recognition is not None and recognition.book_move:
            _should_force = (
                force_book_early
                or self._weights.opening_adherence >= 100
            )
            if _should_force:
                book_mv = _parse_book_move(recognition.book_move, moves)
                if book_mv is not None:
                    self.last_was_blunder = False
                    return book_mv

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
        # Guard: never apply the early-game cap in fly phase (3v3 endgame also has
        # < 10 pieces but needs the full time budget to find multi-move fork combinations).
        total_on_board = sum(board.pieces_on_board.values())
        _in_placement = get_game_phase(board, board.turn) == "place"
        if (total_on_board < _EARLY_GAME_PIECE_THRESHOLD
                and _in_placement
                and not fast_early_game
                and self.difficulty in _TIME_LIMIT
                and self._override_time_budget is None):
            # Cap search depth for the very first placements: on a near-empty board,
            # deep iterative deepening produces horizon effects where corner-based
            # mill-fork patterns score artificially high, overriding the structural
            # preference for high-mobility cardinal/cross nodes.  Depth 6 gives 3 plies
            # per side — enough for tactical awareness without the distortion.
            # Skipped when override_time_budget is set (training fast-mode).
            early_max = 6 if total_on_board < 4 else 19
            move = self._iterative_deepen(
                board, _EARLY_GAME_TIME,
                recognition=recognition, trajectory_hints=trajectory_hints,
                top_n=top_n, moves=moves,
                max_depth=early_max,
            )
            self._populate_thinking(board, move, _forced_block=bool(threats))
            return move

        if self.difficulty in _TIME_LIMIT:
            _base_budget = (
                self._override_time_budget
                if self._override_time_budget is not None
                else _TIME_LIMIT[self.difficulty]
            )
            time_budget = 2.0 if fast_early_game else _base_budget
            move = self._iterative_deepen(
                board, time_budget,
                recognition=recognition, trajectory_hints=trajectory_hints,
                top_n=top_n, moves=moves,
            )
            self._populate_thinking(board, move, _forced_block=bool(threats))
            return move

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
                scored = self._apply_opening_adjustments(scored, recognition, board)
            if trajectory_hints:
                scored = self._apply_trajectory_hints(scored, trajectory_hints)
            if top_n > 1:
                scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
                move = random.choice(scored_sorted[:top_n])[0]
            else:
                move = max(scored, key=lambda x: x[1])[0]
            self._populate_thinking(board, move, _forced_block=bool(threats))
            return move

        move, _ = self._root_search(board, depth, top_n=top_n, moves=moves)
        self._populate_thinking(board, move, _forced_block=bool(threats))
        return move

    def _populate_thinking(
        self, board: BoardState, move: dict, _forced_block: bool = False
    ) -> None:
        """Compute and store a plain-English thinking label for the chosen move.

        Calls tactical_move_bonus with return_breakdown=True to identify the top
        1-2 highest-magnitude contributions.  Stored in self.last_thinking.
        Failures are silently swallowed — thinking is decorative.
        """
        try:
            after = board.apply_move(move)
            bd = tactical_move_bonus(
                board, after, self.color, self._weights,
                self._opp_last_weak, return_breakdown=True,
            )
            if not isinstance(bd, dict):
                return
            top = bd.get("top_terms", [])
            if not top:
                return
            if len(top) >= 2 and abs(top[1][1]) >= abs(top[0][1]) * 0.5:
                # Second term is at least half the first — mention both
                self.last_thinking = f"{top[0][0]} + {top[1][0].lower()}"
            else:
                self.last_thinking = top[0][0]
            # When a mandatory block happens to land on a dead square, the B-64
            # label is misleading — relabel to show the move was unavoidable.
            if _forced_block and "Dead" in self.last_thinking and "placement" in self.last_thinking:
                self.last_thinking = "Forced block (dead square — unavoidable)"
            # Trim to ≤ 8 words
            words = self.last_thinking.split()
            if len(words) > 8:
                self.last_thinking = " ".join(words[:8])
        except Exception:
            self.last_thinking = ""

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

        from game.rules import get_game_phase
        total_on_board = sum(board.pieces_on_board.values())
        if (total_on_board < _EARLY_GAME_PIECE_THRESHOLD
                and get_game_phase(board, board.turn) == "place"):
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
        board: "BoardState | None" = None,
    ) -> list[tuple[dict, int]]:
        """Apply opening-book bonus/penalty to a scored move list.

        Uses absolute bonuses proportional to the opening_adherence slider so
        the book preference always outweighs tactical noise at high adherence.
        The book bonus is suppressed for placement-phase moves landing on dead
        or near-dead squares (0–1 free neighbours) that don't close a mill —
        mirroring B-64 so a book recommendation can't override the dead-
        placement penalty.
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

        # Pre-compute dead suppression for the book dest (placement phase only).
        _book_dest_dead = False
        if board is not None and book_dest:
            # A placement has no "from" key — detect via board context.
            # Dead = 0 or 1 free neighbour after placing; exempt if the placement
            # closes a mill (the piece has tactical value regardless of mobility).
            free_nb = sum(1 for nb in ADJACENCY.get(book_dest, []) if board.positions.get(nb) == "")
            if free_nb <= 1:
                stm = board.turn
                _closes_mill = any(
                    book_dest in mill
                    and all(board.positions.get(sq) == stm or sq == book_dest for sq in mill)
                    for mill in MILLS
                )
                _book_dest_dead = not _closes_mill

        adjusted = []
        for move, raw in scored:
            dest = move.get("to", "")
            delta = 0
            if book_dest and dest == book_dest:
                # Suppress bonus when landing on a dead square during placement.
                is_placement = not move.get("from")
                if not (is_placement and _book_dest_dead):
                    delta += book_bonus_abs
            if dest in blunder_dests:
                delta -= blunder_penalty_abs
            adjusted.append((move, raw + delta))
        return adjusted

    def _store_killer(self, depth: int, from_sq: str | None, to_sq: str) -> None:
        """Record a quiet move that caused a beta cutoff at this remaining depth.

        Killers are stored in a 2-slot FIFO per depth.  Duplicates in slot 0
        are skipped so a repeat cutoff move does not discard the other killer.
        """
        if depth >= 32:
            return
        new_killer = (from_sq, to_sq)
        slot = self._killers[depth]
        if slot[0] != new_killer:
            slot[1] = slot[0]
            slot[0] = new_killer

    # ── Internals ─────────────────────────────────────────────────────────────

    def _root_search(self, board: BoardState, depth: int,
                     top_n: int = 1, moves: list | None = None,
                     alpha: int = -INF, beta: int = INF) -> Tuple[dict, int]:
        """Search all root moves and return (best_move, best_score).
        When top_n > 1, pick randomly from the top-N scoring moves.
        If _SearchAbort fires (force_stop called mid-search), returns the
        best move found so far rather than propagating the exception.
        `moves` may be pre-filtered (e.g. mandatory-block constraint); if None
        all legal moves are used.
        alpha/beta: aspiration window bounds (default full window)."""
        if moves is None:
            moves = get_all_legal_moves(board)
        killers = self._killers[depth] if depth < 32 else None
        moves = _order_moves(board, moves, killers, self._history)
        self._nodes = 0
        best_move = moves[0]
        best_score = -INF
        all_scored: list[Tuple[dict, int]] = []
        # Track raw alpha separately from the total (raw + tactical_bonus) score.
        # The tactical_move_bonus is a root-only adjustment that must NOT inflate
        # the alpha-beta window passed down to _negamax — doing so causes fail-hard
        # clipping to return the alpha bound rather than the true score, making
        # later moves with small bonuses appear spuriously competitive.
        alpha_raw = -INF

        # SE-11b: init path buffer with game history; root moves pushed/popped below.
        self._move_path_buf = list(self._game_notations)

        scored_any = False
        for move in moves:
            nb = board.apply_move(move)
            _root_mn = self._move_notation(move)
            self._move_path_buf.append(_root_mn)
            try:
                score_raw = -self._negamax(nb, depth - 1, -beta, -alpha_raw, None, depth // 2, _MAX_OPP_PLIES, 1)
            except _SearchAbort:
                self._move_path_buf.pop()
                if not scored_any:
                    raise  # no moves fully evaluated — propagate so _iterative_deepen keeps previous depth
                break
            self._move_path_buf.pop()
            scored_any = True
            # Deadline check before expensive tactical bonus: abort between root moves if time is up.
            if time.time() >= self._deadline:
                break
            # Don't apply tactical bonus when the move is already a near-certain win/loss
            # (score near INF via endgame DB or terminal).  Bonuses would otherwise favour
            # cycling moves over mill-closing captures, preventing actual conversion.
            if abs(score_raw) < INF // 2:
                score = score_raw + tactical_move_bonus(board, nb, self.color, self._weights, self._opp_last_weak)
            else:
                score = score_raw
            if top_n > 1:
                all_scored.append((move, score))
            if score > best_score:
                best_score = score
                best_move = move
            if score_raw > alpha_raw:
                alpha_raw = score_raw
            if alpha_raw >= beta:
                break

        if top_n > 1 and all_scored:
            top = sorted(all_scored, key=lambda x: x[1], reverse=True)[:top_n]
            best_move = random.choice(top)[0]
        if best_score == -INF:
            # Deadline fired between scored_any=True and the score comparison, so
            # best_score was never updated from its sentinel.  Signal as an abort so
            # _iterative_deepen treats this depth as incomplete.
            raise _SearchAbort()
        return best_move, best_score

    def _negamax(
        self,
        board: BoardState,
        depth: int,
        alpha: int,
        beta: int,
        endgame_state=None,
        ext_budget: int = 0,
        opp_plies_left: int = 0,  # SE-11b/11c: opponent plies remaining for trajectory/VN extension
        ply: int = 0,              # plies from root; used for mate scoring (faster wins score higher)
    ) -> int:
        """
        Negamax with alpha-beta pruning and transposition table.
        Returns score from board.turn's perspective (higher = better for board.turn).
        Raises _SearchAbort when the search deadline is exceeded.
        """
        self._nodes += 1
        # Check deadline every 4096 nodes to avoid time.time() call overhead.
        if self._nodes & 0xFFF == 0 and time.time() >= self._deadline:
            raise _SearchAbort()

        terminal, _ = is_terminal(board)
        if terminal:
            return -(INF - ply)

        # SE-14: FullGameDB probe — exact outcomes short-circuit; best-move hints
        # are stored for promotion after the move list is generated.
        _db14_best_move: Optional[str] = None
        if self._fullgame_db is not None:
            try:
                _db14 = self._fullgame_db.query(board)
                if _db14 is not None:
                    if _db14.outcome is not None:
                        _stm = board.turn
                        if _db14.outcome == 1:
                            return INF - ply if _stm == "W" else -(INF - ply)
                        elif _db14.outcome == -1:
                            return INF - ply if _stm == "B" else -(INF - ply)
                        else:  # draw
                            return 0
                    elif _db14.best_move_canonical:
                        _inv = SYM_INVERSE[_db14.sym_idx]
                        _db14_best_move = _transform_notation(_db14.best_move_canonical, _inv)
            except Exception:
                pass

        # SE-8: search extension for critical positions.
        if ext_budget > 0:
            _color = board.turn
            _opp8 = "B" if _color == "W" else "W"
            _own_threat = False
            _opp_forks = 0
            for _mill in MILLS:
                _mv = [board.positions[p] for p in _mill]
                if _mv.count(_color) == 2 and _mv.count("") == 1:
                    _own_threat = True
                if _mv.count(_opp8) == 2 and _mv.count("") == 1:
                    _opp_forks += 1
            if _own_threat or _opp_forks >= 2:
                depth += 1
                ext_budget -= 1

        if depth == 0:
            # SE-4: endgame tablebase probe at leaves — ply-based scoring so faster
            # wins score higher than slower ones (INF - ply decreases as ply increases).
            if (self._endgame_solved_db is not None
                    and board.pieces_placed.get("W", 0) >= 9
                    and board.pieces_placed.get("B", 0) >= 9):
                try:
                    _wdl = self._endgame_solved_db.query(board)
                except Exception:
                    _wdl = None
                if _wdl == "W":
                    return INF - ply
                elif _wdl == "L":
                    return -(INF - ply)
                elif _wdl == "D":
                    return 0
            if self._neural_evaluator is not None:
                return self._neural_evaluator.evaluate(board)
            _q_moves = get_all_legal_moves(board)
            if any(m.get("capture") for m in _q_moves):
                heur = self._qsearch(board, self._Q_DEPTH, alpha, beta, endgame_state, _q_moves)
            else:
                heur = evaluate(board, board.turn, endgame_state, self.force_aggressive, self._weights)
            # B-73: blend in value network score when loaded and blend > 0
            if self._value_net is not None and self._weights.value_net_blend > 0:
                vn_raw = self._value_net.predict(board, board.turn)  # (-1, 1)
                vn_score = int(vn_raw * _VN_SCALE)
                blend = self._weights.value_net_blend / 100.0
                return int(blend * vn_score + (1.0 - blend) * heur)
            return heur

        # ── Transposition table probe ─────────────────────────────────────────
        alpha_orig = alpha
        tt_move_from = tt_move_to = None
        tt_entry = self._tt.lookup(board.hash_key)
        if tt_entry is not None:
            tt_depth, tt_score, tt_flag, tt_move_from, tt_move_to = tt_entry
            # Denormalize mate scores: stored as "mate in N from this position",
            # convert back to "mate in N from root" by adjusting by current ply.
            if abs(tt_score) > INF // 2:
                tt_score = tt_score - ply if tt_score > 0 else tt_score + ply
            if tt_depth >= depth:
                if tt_flag == EXACT:
                    return tt_score
                if tt_flag == LOWER_BOUND and tt_score >= beta:
                    return tt_score
                if tt_flag == UPPER_BOUND and tt_score <= alpha:
                    return tt_score

        moves = get_all_legal_moves(board)
        if not moves:
            return -(INF - ply)

        # Sort at upper levels only — biggest benefit to alpha-beta, negligible overhead
        if depth >= 2:
            killers = self._killers[depth] if depth < 32 else None
            moves = _order_moves(board, moves, killers, self._history)

        # SE-11b/11c: classify node and prepare trajectory/VN extension state.
        is_opp_node = (board.turn != self.color)
        _do_path = opp_plies_left > 0
        _next_opp_plies = max(0, opp_plies_left - 1) if is_opp_node else opp_plies_left

        # SE-11b: query trajectory frequency dict at first opponent ply only (307µs/call — too
        # expensive at ply 2 where ~27k nodes × 307µs ≈ 8 s overhead).
        _opp_freq = None
        if is_opp_node and opp_plies_left == _MAX_OPP_PLIES and self._trajectory_db is not None and self._move_path_buf:
            _opp_freq = self._trajectory_db.query_all_frequencies(
                self._move_path_buf, min_samples=3
            ) or None

        # SE-14: Promote DB hint to front (done before TT so TT gets final priority).
        if _db14_best_move is not None:
            for i, m in enumerate(moves):
                _mv_str = (f"{m['from']}-{m['to']}" if m.get("from") else m.get("to", ""))
                if m.get("capture"):
                    _mv_str += f"x{m['capture']}"
                if _mv_str == _db14_best_move:
                    if i > 0:
                        moves.insert(0, moves.pop(i))
                    break

        # Promote the TT best-move to the front of the list regardless of its
        # priority bucket — it was the best move last time we searched this position.
        if tt_move_to is not None:
            for i, m in enumerate(moves):
                if m.get("from") == tt_move_from and m["to"] == tt_move_to:
                    if i > 0:
                        moves.insert(0, moves.pop(i))
                    break

        # SE-11c: at first opponent ply, re-sort the non-priority (p2) tail by VN score
        # so LMR reduces the moves the opponent is least likely to play strongly.
        # Gated to first opponent ply only (opp_plies_left == _MAX_OPP_PLIES) to contain overhead.
        if (is_opp_node and opp_plies_left == _MAX_OPP_PLIES
                and self._value_net is not None and depth >= 3 and len(moves) > 2):
            _vn_n = len(moves)
            _vn_lmr_s = _vn_n - int(_vn_n * 0.6)
            if _vn_n - _vn_lmr_s > 1:
                _tail = moves[_vn_lmr_s:]
                _vn_sc = [self._value_net.predict(board.apply_move(m), board.turn)
                          for m in _tail]
                moves[_vn_lmr_s:] = [m for _, m in sorted(zip(_vn_sc, _tail), reverse=True)]

        # SE-6: LMR — pre-compute opponent blocking squares so they are never reduced.
        _opp_color = "B" if board.turn == "W" else "W"
        _block_squares: set = set()
        for _bm in MILLS:
            _bv = [board.positions[p] for p in _bm]
            if _bv.count(_opp_color) == 2 and _bv.count("") == 1:
                _block_squares.add(next(p for p in _bm if board.positions[p] == ""))

        # SE-5 + SE-6: PVS with Late Move Reductions.
        # First move: full window.  Non-late siblings: PVS zero-window scout.
        # Late quiet non-blocking moves (last 60% at depth ≥ 4): reduced by 1 ply
        # with a zero-window scout; re-searched at full depth on fail-high, then at
        # full window if PVS also fails high.
        n_moves = len(moves)
        lmr_start = n_moves - int(n_moves * 0.6)  # first index of the late-move tier
        value = -INF
        best_from = best_to = None
        is_first = True
        for move_idx, move in enumerate(moves):
            nb = board.apply_move(move)

            _mv_notation = self._move_notation(move)

            # SE-11b: extend by 1 for high-frequency opponent moves (first 2 opponent plies).
            _se11_ext = 0
            if is_opp_node and _opp_freq is not None:
                if _opp_freq.get(_mv_notation, 0.0) >= 0.5:
                    _se11_ext = 1

            # SE-11b: push move to shared path buffer so deeper trajectory/VN nodes see full history.
            if _do_path:
                self._move_path_buf.append(_mv_notation)

            use_lmr = (
                depth >= 4
                and not is_first
                and move_idx >= lmr_start
                and not move.get("capture")
                and move["to"] not in _block_squares
            )

            if is_first:
                score = -self._negamax(nb, depth - 1 + _se11_ext, -beta, -alpha, endgame_state, ext_budget, _next_opp_plies, ply + 1)
                is_first = False
            elif use_lmr:
                # LMR: reduced-depth zero-window scout
                score = -self._negamax(nb, depth - 2 + _se11_ext, -alpha - 1, -alpha, endgame_state, ext_budget, _next_opp_plies, ply + 1)
                if score > alpha:
                    # Failed high — re-search at full depth with PVS zero-window
                    score = -self._negamax(nb, depth - 1 + _se11_ext, -alpha - 1, -alpha, endgame_state, ext_budget, _next_opp_plies, ply + 1)
                    if alpha < score < beta:
                        # PVS also failed high — full window re-search
                        score = -self._negamax(nb, depth - 1 + _se11_ext, -beta, -alpha, endgame_state, ext_budget, _next_opp_plies, ply + 1)
            else:
                # Standard PVS zero-window scout
                score = -self._negamax(nb, depth - 1 + _se11_ext, -alpha - 1, -alpha, endgame_state, ext_budget, _next_opp_plies, ply + 1)
                if alpha < score < beta:
                    score = -self._negamax(nb, depth - 1 + _se11_ext, -beta, -alpha, endgame_state, ext_budget, _next_opp_plies, ply + 1)

            # SE-11b: restore path buffer after exploring this branch.
            if _do_path:
                self._move_path_buf.pop()

            if score > value:
                value = score
                best_from = move.get("from")
                best_to   = move["to"]
            if value > alpha:
                alpha = value
            if alpha >= beta:
                # Beta cutoff: update killer and history tables for quiet moves.
                # Captures are already in priority-0 and don't need these signals.
                if not move.get("capture"):
                    self._store_killer(depth, move.get("from"), move["to"])
                    key = (move.get("from"), move["to"])
                    self._history[key] = self._history.get(key, 0) + depth * depth
                break

        # ── Transposition table store ─────────────────────────────────────────
        if best_to is not None:
            if value <= alpha_orig:
                flag = UPPER_BOUND
            elif value >= beta:
                flag = LOWER_BOUND
            else:
                flag = EXACT
            # Normalize mate scores to "mate in N from this position" so they
            # remain correct when retrieved at a different ply in later iterations.
            store_value = value
            if abs(value) > INF // 2:
                store_value = value + ply if value > 0 else value - ply
            self._tt.store(board.hash_key, depth, store_value, flag, best_from, best_to)

        return value

    def _qsearch(self, board: BoardState, q_depth: int, alpha: int, beta: int,
                 endgame_state=None, moves: list | None = None) -> int:
        """SE-9: Quiescence search — extend only capturing moves to resolve tactical noise."""
        self._nodes += 1
        if self._nodes & 0xFFF == 0 and time.time() >= self._deadline:
            raise _SearchAbort()

        terminal, _ = is_terminal(board)
        if terminal:
            return -(INF - q_depth)

        stand_pat = evaluate(board, board.turn, endgame_state, self.force_aggressive, self._weights)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        if q_depth <= 0:
            return alpha

        if moves is None:
            moves = get_all_legal_moves(board)
        for move in moves:
            if not move.get("capture"):
                continue
            nb = board.apply_move(move)
            score = -self._qsearch(nb, q_depth - 1, -beta, -alpha, endgame_state)
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return alpha

    def _score_all(
        self, board: BoardState, moves: list, depth: int, endgame_state=None
    ) -> list[tuple[dict, int]]:
        """Score every move in `moves` and return [(move, score), ...].

        If _SearchAbort is raised mid-loop (force_stop() called), unscored moves
        receive the worst score seen so far so max() still picks the best partial result.
        """
        self._nodes = 0
        # SE-11b: init path buffer with game history; root moves pushed/popped below.
        self._move_path_buf = list(self._game_notations)
        results = []
        for i, move in enumerate(moves):
            nb = board.apply_move(move)
            _root_mn = self._move_notation(move)
            self._move_path_buf.append(_root_mn)
            try:
                score = -self._negamax(nb, depth - 1, -INF, INF, endgame_state, depth // 2, _MAX_OPP_PLIES, 1)
                if abs(score) < INF // 2:
                    score += tactical_move_bonus(board, nb, self.color, self._weights, self._opp_last_weak)
                self._move_path_buf.pop()
                results.append((move, score))
            except _SearchAbort:
                self._move_path_buf.pop()
                worst = min(s for _, s in results) if results else -INF
                for remaining in moves[i:]:
                    results.append((remaining, worst))
                break
        return results

    _BLUNDER_TIME = 2.0   # hard cap for blunder scoring — ranking doesn't need deep search
    _BLUNDER_DEPTH = 3    # shallow depth is enough to distinguish good/bad moves
    _Q_DEPTH = 2          # SE-9: extra plies in quiescence search (cap on capture chain depth)

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
        max_depth: int = 19,
    ) -> dict:
        """
        Iterative deepening up to `time_limit` seconds.

        When opening recognition or trajectory hints are active, scores every
        root move at each depth so the adjustments can be applied before
        picking the best.  Otherwise uses the faster _root_search path.
        `moves` may be pre-filtered (e.g. mandatory-block constraint); if None
        all legal moves are used.
        """
        _ASP_MARGIN = 175
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

        prev_score: int | None = None
        for depth in range(2, max_depth + 1):
            if time.time() >= self._deadline:
                break
            try:
                if use_adjustments:
                    scored = self._score_all(board, moves, depth)
                    if recognition is not None:
                        scored = self._apply_opening_adjustments(scored, recognition, board)
                    if trajectory_hints:
                        scored = self._apply_trajectory_hints(scored, trajectory_hints)
                    if top_n > 1:
                        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
                        best_move = random.choice(scored_sorted[:top_n])[0]
                    else:
                        best_move = max(scored, key=lambda x: x[1])[0]
                else:
                    # SE-7: aspiration windows — narrow the window around the previous score.
                    if prev_score is not None and depth >= 3 and top_n == 1:
                        asp_lo = prev_score - _ASP_MARGIN
                        asp_hi = prev_score + _ASP_MARGIN
                        move, score = self._root_search(
                            board, depth, top_n=1, moves=moves,
                            alpha=asp_lo, beta=asp_hi,
                        )
                        if score <= asp_lo:
                            try:
                                move, score = self._root_search(
                                    board, depth, top_n=1, moves=moves, beta=asp_hi,
                                )
                            except _SearchAbort:
                                # Fail-low re-search aborted: first pass returned an incomplete
                                # (possibly all-LOSS) result because deadline fired during move
                                # ordering (losses first).  Keep previous depth's best_move.
                                break
                        elif score >= asp_hi:
                            try:
                                move, score = self._root_search(
                                    board, depth, top_n=1, moves=moves, alpha=asp_lo,
                                )
                            except _SearchAbort:
                                pass  # first call's high-scoring move is still in `move`
                        best_move = move
                        prev_score = score
                    else:
                        move, score = self._root_search(board, depth, top_n=top_n, moves=moves)
                        best_move = move      # only update if depth completed cleanly
                        prev_score = score
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
