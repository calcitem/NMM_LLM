"""
ai/game_ai.py — Minimax AI using negamax + alpha-beta pruning.

GameAI plays the computer's side.  It also exposes score_move() for rating
human moves, used by the LLM commentary system to decide whether to comment.

Blunder mode: set blunder_probability > 0 to make the AI occasionally play a
deliberately poor move so the human can practise exploiting mistakes.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Optional, Tuple

_logger = logging.getLogger(__name__)


class _SearchAbort(Exception):
    """Raised inside _negamax when the search deadline has passed."""

from game.board import ADJACENCY, MILLS, POSITIONS, BoardState
from game.rules import get_all_legal_moves, is_terminal

# T-D2: O(1) reverse lookup used by _notation_to_triple and _choose_rust_scored filter.
_POS_TO_IDX: dict[str, int] = {pos: i for i, pos in enumerate(POSITIONS)}
from .heuristics import INF, evaluate, clear_eval_cache, HeuristicWeights, DEFAULT_WEIGHTS, tactical_move_bonus, _sealed_two_configs, _dual_connected_mill_alert, _closeable_mills, evaluate_v2
from .transposition_table import TranspositionTable, EXACT, LOWER_BOUND, UPPER_BOUND
from .board_symmetry import SYM_INVERSE, transform_notation as _transform_notation

# B-73: value network scale — maps VN output (-1, 1) to heuristic score range.
# A VN score of +1.0 (certain win) maps to ±3000, comparable to a large positional advantage.
_VN_SCALE = 3000
# SE-11b/11c: how many opponent plies from root receive trajectory extension and VN ordering.
_MAX_OPP_PLIES    = 2
_MAX_OPP_PLIES_V2 = 6   # deeper path-buffer tracking when v2 heuristics are active


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
    opp_placed = board.pieces_placed.get(opp, 0)
    opp_in_fly = opp_placed >= 9 and board.pieces_on_board[opp] <= 3

    try:
        from . import native_core as _nc
        if _nc.RUST_AVAILABLE:
            import nmm_core as _rc
            white, black, wp, bp, stm_u8 = _nc.board_to_bits(board)
            mask = _rc.py_immediate_threats(white, black, wp, bp, stm_u8)
            threats: set[str] = {POSITIONS[i] for i in range(24) if mask & (1 << i)}
            # B-66 carveout: single threat (non-fly) + STM can close own mill → do not restrict.
            if not opp_in_fly and len(threats) == 1 and _stm_can_close_mill(board, board.turn):
                threats.clear()
            return threats
    except Exception:
        pass

    # Pure-Python fallback.
    stm = board.turn
    threats_py: set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(opp) == 2 and vals.count("") == 1:
            empty = next(p for p in mill if board.positions[p] == "")
            if opp_in_fly:
                threats_py.add(empty)
            elif opp_placed >= 9:  # move phase: need adjacent EXTERNAL opp piece
                mill_set = set(mill)
                if any(
                    board.positions[nb] == opp and nb not in mill_set
                    for nb in ADJACENCY[empty]
                ):
                    threats_py.add(empty)

    # Move phase: single threat + own mill available → do not restrict (B-66).
    if opp_placed >= 9 and not opp_in_fly and len(threats_py) == 1:
        if _stm_can_close_mill(board, stm):
            threats_py.clear()

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
            threats_py.update(closing)
        elif closing:
            if not _stm_can_close_mill(board, stm):
                threats_py.update(closing)

    return threats_py


def _notation_to_triple(notation: str) -> "tuple[int | None, int, int | None] | None":
    """Parse a move notation into a (from_idx|None, to_idx, cap_idx|None) triple."""
    try:
        cap: "int | None" = None
        if "x" in notation:
            main, cap_str = notation.split("x", 1)
            cap = _POS_TO_IDX[cap_str]
        else:
            main = notation
        frm: "int | None" = None
        if "-" in main:
            fr_str, to_str = main.split("-", 1)
            frm = _POS_TO_IDX[fr_str]
        else:
            to_str = main
        to = _POS_TO_IDX[to_str]
        return (frm, to, cap)
    except KeyError:
        return None


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
            mill_set = set(mill)
            # Require the adjacent opp piece to be EXTERNAL to the mill: a piece
            # already inside the mill cannot slide to the vacated square and close it.
            if any(
                board.positions.get(nb, "") == opp and nb not in mill_set
                for nb in ADJACENCY.get(our_sq, [])
            ):
                pinned.add(our_sq)
    return frozenset(pinned)


def _pinned_move_squares_2ply(board: BoardState, color: str) -> frozenset:
    """2-ply move-phase pin: own square that, if vacated, lets the opponent build
    a 2-config in two moves (slide into vacated square + feeder slides into mill).

    Pattern: mill (S, X, Y) where own=1, opp=1, empty=1; opp_sq is adjacent to
    own_sq (can slide in immediately); a feeder opp piece is adjacent to opp_sq
    but outside the mill (would complete a 2-config on the next move).
    """
    opp = "B" if color == "W" else "W"
    pinned: set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) != 1 or vals.count(opp) != 1 or vals.count("") != 1:
            continue
        our_sq = next(p for p in mill if board.positions[p] == color)
        opp_sq = next(p for p in mill if board.positions[p] == opp)
        if opp_sq not in ADJACENCY.get(our_sq, []):
            continue
        mill_set = set(mill)
        if any(
            board.positions.get(nb) == opp and nb not in mill_set
            for nb in ADJACENCY.get(opp_sq, [])
        ):
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


def _order_moves(board: BoardState, moves: list, killers=None, history=None, _is_beginner: bool = False) -> list:
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
    # B-95: skip complex ordering heuristics at beginner difficulty.
    if not _is_beginner and get_game_phase(board, color) == "place":
        block |= set(_dual_connected_mill_alert(board, opp))

    if not close and not block and not killer_set and not history:
        return moves  # nothing to prioritize — skip the pass

    # B-59: P0.5 — moves that create a new sealed 2-config (uncontestable forced mill).
    # Only computed in move phase when there are no direct mill closes (P0) — those
    # dominate regardless, and the extra apply_move calls would be wasted.
    # Covers all 16 MILLS (not just inner ring): any sealed pattern is elevated.
    # B-95: skip at beginner difficulty.
    sealed_creates: set = set()  # stores (from, to) tuples
    if not _is_beginner and get_game_phase(board, color) == "move" and not close:
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
_EARLY_GAME_PIECE_THRESHOLD = 3   # only the first 3 placements total get the short budget
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


def _dead_has_mill_potential(board: BoardState, to: str) -> bool:
    """True when `to` belongs to at least one mill line not already opponent-blocked.

    Used as a tiebreaker when all remaining placements are dead: prefer squares
    that still have a plausible mill to aim for (opponent has not yet occupied
    another square in every containing mill).
    """
    opp = "B" if board.turn == "W" else "W"
    for mill in MILLS:
        if to not in mill:
            continue
        if not any(board.positions.get(sq) == opp for sq in mill if sq != to):
            return True
    return False


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
        malom_db=None,              # learned_ai.sentinel.db_teacher.ExternalSolvedDB | None
        override_time_budget: float | None = None,  # seconds; overrides _TIME_LIMIT for training
    ) -> None:
        self.color = color
        self.difficulty = max(1, min(10, difficulty))
        self.blunder_probability = max(0.0, min(1.0, blunder_probability))
        self._weights: HeuristicWeights = weights if weights is not None else DEFAULT_WEIGHTS
        self._beginner_weights: HeuristicWeights | None = None  # lazily built in _is_beginner
        self.use_v2_heuristics: bool = True   # evaluate_v2() at leaves (v1 kept for MCTS/position_eval)
        self._fullgame_db = fullgame_db
        self._endgame_solved_db = endgame_solved_db
        self._malom_db = malom_db
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
        self._rust_tt_handle: "object | None" = None              # T-C4: lazy RustTtHandle (persists between turns)
        self._rust_fullgame_db_handle: "object | None" = None     # T-C2: lazy FullgameDbHandle (mmap'd DB)
        self._rust_endgame_solved_handle: "object | None" = None  # T-C3: lazy EndgameSolvedDbHandle
        self.search_threads: int = 1         # T-E3: Lazy SMP thread count; 1 = single-threaded
        # SE-2: 2 killer moves per remaining-depth level (up to depth 32).
        # Each slot is (from_sq, to_sq) or None.
        self._killers: list[list] = [[None, None] for _ in range(32)]
        # SE-3: global history table keyed by (from_sq, to_sq); value = Σ depth².
        self._history: dict = {}
        self._opp_plies_budget: int = _MAX_OPP_PLIES  # updated per-search; see Stage 8
        self.max_search_depth: int = 19   # hard cap on iterative-deepening depth; settable per difficulty
        self.time_budget_override: float | None = None  # depth-derived budget set by _apply_search_depth
        self._force_stop: bool = False     # set by force_stop(); cleared by choose_move()
        self.last_was_blunder: bool = False   # flag readable by Coordinator / MillsLLM
        self.last_thinking: str = ""          # short plain-English label for the chosen move
        # Prefix used in terminal search output: "R" = main search, "P" = ponder branch.
        self._search_label: str = "R"
        self.last_depth_reached: int = 1      # deepest completed depth from last _iterative_deepen
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
        self._trajectory_line: list[tuple[str, float]] = []
        self._game_notations: list = []
        self._move_path_buf: list = []  # SE-11b: shared push/pop path buffer for _negamax recursion
        # Per-instance time-budget override (used during training to keep games fast).
        self._override_time_budget = override_time_budget
        # Sentinel overlay (advisory only by default). None => zero impact; the
        # game plays identically. Set via set_sentinel(). See ai/../learned_ai/sentinel.
        self.sentinel = None                 # SentinelAdvisor | None
        self.sentinel_mode: str = "advisory"  # "advisory" | "score_adjust" | "reconsider"
        # score_adjust scale + reconsider threshold: read from the advisor's config
        # when present (set in set_sentinel), else fall back to documented defaults.
        self._sentinel_score_scale: float = 0.05
        self._sentinel_reconsider_threshold: float = 0.15
        # Minimum opportunity gap required before sentinel overrides in any active mode.
        # 0.0 = always override when blended score prefers a different move (default).
        # Higher values (e.g. 0.20) mean sentinel only intercedes on larger mistakes.
        self._sentinel_min_gap: float = 0.0
        # Optional LLM move recommender for reconsider mode. The LLM lives in the
        # Coordinator, not in GameAI, so it is injected as a callback to avoid
        # duplicating any LLM logic here. Signature: fn(board, legal_moves) -> move|None.
        # When None, the reconsider LLM path is skipped (falls through to deepened search).
        self._llm_move_fn = None
        # Rolling trajectory of recent chosen-move scores, for sentinel context.
        self._sentinel_trajectory: list[float] = []
        self.last_sentinel_advice = None     # last SentinelAdvice (debug/logging)
        # Probability gate: fraction of moves where sentinel (or DB fallback) fires.
        # 1.0 = always (default, preserves existing use_sentinel=True behaviour).
        self._sentinel_activation_prob: float = 1.0
        # When True, the AI uses the Malom perfect DB (via _db_score_adjust) instead of
        # the sentinel model for move selection.  Probability gating still applies.
        self.use_perfect_db: bool = False
        # When True, feeder_diamond and capture_creates_diamond are zeroed in
        # _active_weights() to prevent the AI from always playing fork-creating
        # placements. Set randomly per game (~50%) by the web server for variety.
        # Within such games, suppression fires on ~50% of individual moves
        # (_suppress_fork_this_move is re-rolled each choose_move() call).
        self.suppress_fork_variety: bool = False
        self._suppress_fork_this_move: bool = False
        self._variety_weights: HeuristicWeights | None = None  # lazily built

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def _db_access_prob(self) -> float:
        """Probability (0–1) that solved-DB probes fire this move.
        Level 1 = 0 %, levels 2–7 = linearly interpolated, levels 8–10 = 100 %.
        """
        if self.difficulty <= 1:
            return 0.0
        if self.difficulty >= 8:
            return 1.0
        return (self.difficulty - 1) / 7.0

    def set_sentinel(self, sentinel, mode: str = "advisory") -> None:
        """Attach a SentinelAdvisor overlay. ``mode`` is advisory|score_adjust|reconsider.

        Advisory mode only logs warnings/flags and never changes the chosen move.
        """
        self.sentinel = sentinel
        self.sentinel_mode = mode or "advisory"
        # Pull tunables from the advisor's config when available.
        try:
            cfg = getattr(sentinel, "config", None)
            if cfg is not None:
                self._sentinel_score_scale = float(
                    getattr(cfg, "score_adjust_scale", self._sentinel_score_scale)
                )
                self._sentinel_reconsider_threshold = float(
                    getattr(cfg, "reconsider_threshold", self._sentinel_reconsider_threshold)
                )
        except Exception:
            pass
        try:
            _logger.info("[GameAI] Sentinel overlay loaded in %s mode.", self.sentinel_mode)
        except Exception:
            pass

    @property
    def _is_beginner(self) -> bool:
        return self.difficulty <= 2

    def _active_weights(self) -> HeuristicWeights:
        """Return weights adjusted for difficulty and per-game/per-move variety suppression."""
        _suppress = self.suppress_fork_variety and self._suppress_fork_this_move
        if not self._is_beginner and not _suppress:
            return self._weights
        if self._is_beginner and self._beginner_weights is None:
            import dataclasses
            self._beginner_weights = dataclasses.replace(
                self._weights,
                fork_anticipation=0,
                black_fork_anticipation_early=0,
                defer_for_chain=0,
            )
        base = self._beginner_weights if self._is_beginner else self._weights
        if not _suppress:
            return base
        if self._variety_weights is None:
            import dataclasses
            self._variety_weights = dataclasses.replace(
                base,
                feeder_diamond=0,
                capture_creates_diamond=0,
            )
        return self._variety_weights

    def set_llm_move_fn(self, fn) -> None:
        """Inject an LLM move recommender for reconsider mode.

        ``fn(board, legal_moves) -> move_dict | None``. The LLM itself lives in the
        Coordinator; this callback lets reconsider mode reuse it without GameAI
        duplicating any LLM logic. Optional — when unset the LLM path is skipped.
        """
        self._llm_move_fn = fn

    def _consult_sentinel(self, board: BoardState, moves: list) -> None:
        """Deprecated pre-selection hook — retained as a no-op.

        The move-level sentinel needs the chosen move to score it against its
        alternatives, so all sentinel work now happens in
        :meth:`_apply_sentinel_intervention` after the engine has picked a move.
        """
        return

    def _sentinel_advise(self, board: BoardState, move: dict, moves: list):
        """Score every candidate with the sentinel and return its SentinelAdvice.

        ``move`` is the engine's chosen move; its index within ``moves`` becomes
        ``played_move_idx`` so the advice reports the played move's quality and
        opportunity gap. Never raises — returns None on any failure.
        """
        if self.sentinel is None:
            return None
        try:
            candidates = list(moves or [])
            if not candidates:
                return None
            try:
                played_idx = candidates.index(move)
            except ValueError:
                played_idx = 0
            advice = self.sentinel.advise(
                board, candidates, self.color, played_move_idx=played_idx
            )
            return advice
        except Exception as exc:
            try:
                _logger.debug("[Sentinel] advise() failed: %s", exc)
            except Exception:
                pass
            return None

    def _apply_sentinel_intervention(
        self, board: BoardState, move: dict, moves: list
    ) -> dict:
        """Score the chosen move against its alternatives and intervene per mode.

        Runs one batched forward pass over all candidates, stores the resulting
        SentinelAdvice in ``self.last_sentinel_advice`` (advisory logging always
        happens), then — unless in advisory mode — may swap to a better move. On
        ANY error the original heuristic move is returned unchanged. Only moves
        from ``moves`` are ever returned.
        """
        # Perfect DB mode: use Malom DB directly, bypassing the sentinel model.
        if self.use_perfect_db:
            import random as _random
            if self._sentinel_activation_prob >= 1.0 or _random.random() <= self._sentinel_activation_prob:
                return self._db_score_adjust(board, move, moves)
            return move

        if self.sentinel is None:
            # DB fallback when sentinel unavailable but probability gate would fire
            if (self._sentinel_activation_prob > 0.0
                    and self.sentinel_mode == "score_adjust"):
                import random as _random
                if _random.random() <= self._sentinel_activation_prob:
                    return self._db_score_adjust(board, move, moves)
            return move
        # Probability gate: skip intervention some fraction of the time at lower difficulties
        import random as _random
        if _random.random() > self._sentinel_activation_prob:
            return move
        advice = self._sentinel_advise(board, move, moves)
        self.last_sentinel_advice = advice
        if advice is None:
            return move

        # Always record engine's intended move and sentinel's top recommendation.
        try:
            advice.engine_move_notation = self._move_notation(move)
            best_idx = int(getattr(advice, "best_sentinel_move_idx", 0))
            if 0 <= best_idx < len(moves):
                advice.best_sentinel_move_notation = self._move_notation(moves[best_idx])
        except Exception:
            pass

        # Advisory logging (all modes): record the chosen move's quality.
        try:
            self._sentinel_trajectory.append(advice.played_move_quality)
            if advice.advisory_message != "safe":
                _logger.warning(
                    "[Sentinel] %s for %s at phase=%s "
                    "(played_q=%.2f, best_q=%.2f, gap=%.2f)",
                    advice.advisory_message, advice.player, board.phase,
                    advice.played_move_quality, advice.best_available_quality,
                    advice.opportunity_gap,
                )
        except Exception:
            pass

        if self.sentinel_mode == "advisory":
            return move
        try:
            if self.sentinel_mode == "score_adjust":
                return self._sentinel_score_adjust(board, move, moves, advice)
            if self.sentinel_mode == "reconsider":
                return self._sentinel_reconsider(board, move, moves, advice)
        except Exception as exc:
            try:
                _logger.debug("[Sentinel] intervention failed, keeping heuristic move: %s", exc)
            except Exception:
                pass
        return move

    def _sentinel_score_adjust(
        self, board: BoardState, move: dict, moves: list, advice
    ) -> dict:
        """score_adjust mode: override engine when sentinel sees a clear improvement.

        Sentinel's top-ranked move replaces the engine's choice when both guards pass:
          1. gap >= _sentinel_min_gap  (user-configured threshold)
          2. best sentinel quality >= 0.65  (sentinel is confident)

        The old blended-rank approach could silently swallow large gaps when the
        sentinel's recommended move happened to rank low in the engine's move-ordering,
        making the user-set gap threshold unreliable.  Direct override is predictable:
        if you set gap=20% and sentinel sees 32%, it intervenes.
        """
        scores = list(getattr(advice, "move_scores", []) or [])
        if len(scores) < 2 or len(scores) != len(moves):
            return move

        gap  = float(getattr(advice, "opportunity_gap", 0.0))
        best_q = float(getattr(advice, "best_available_quality", 0.0))
        best_idx = int(getattr(advice, "best_sentinel_move_idx", 0))

        # Gap guard: user-controlled threshold — sole arbiter of intervention.
        if gap < self._sentinel_min_gap:
            return move

        if not (0 <= best_idx < len(moves)):
            return move
        new_move = moves[best_idx]
        if new_move == move:
            return move

        advice.original_move_notation = self._move_notation(move)
        advice.intervention_applied = "score_adjust"
        advice.intervention_detail = (
            f"Score adjust — gap {gap:.0%} ≥ {self._sentinel_min_gap:.0%} threshold "
            f"(engine {advice.played_move_quality:.0%} → sentinel {best_q:.0%})"
        )
        _logger.info(
            "[Sentinel] intervened: engine intended %s → redirected to %s "
            "(type: score_adjust, gap: %.2f, threshold: %.2f)",
            self._move_notation(move), self._move_notation(new_move),
            gap, self._sentinel_min_gap,
        )
        return new_move

    def _sentinel_reconsider(
        self, board: BoardState, move: dict, moves: list, advice
    ) -> dict:
        """reconsider mode: act on a meaningful opportunity gap.

        When ``opportunity_gap > reconsider_threshold`` the chosen move is
        materially worse than the sentinel's best alternative. Prefer an LLM
        recommendation when one is available; otherwise fall back to the
        sentinel's best move.
        """
        gap = float(getattr(advice, "opportunity_gap", 0.0))
        if gap <= self._sentinel_reconsider_threshold:
            return move

        # 1. Try the LLM first (reuses the Coordinator's recommender via callback).
        if self._llm_move_fn is not None:
            try:
                llm_move = self._llm_move_fn(board, list(moves))
            except Exception:
                llm_move = None
            if llm_move is not None and llm_move in moves:
                advice.original_move_notation = self._move_notation(move)
                advice.intervention_applied = "llm_override"
                advice.intervention_detail = (
                    f"LLM override — {advice.advisory_message} (gap={gap:.0%})"
                )
                _logger.info(
                    "[Sentinel] intervened: engine intended %s → redirected to %s "
                    "(type: llm_override, gap: %.2f)",
                    self._move_notation(move), self._move_notation(llm_move), gap,
                )
                return llm_move

        # 2. LLM unavailable → take the sentinel's best-scoring candidate.
        best_idx = int(getattr(advice, "best_sentinel_move_idx", 0))
        if 0 <= best_idx < len(moves):
            best_move = moves[best_idx]
            if best_move != move:
                advice.original_move_notation = self._move_notation(move)
                advice.intervention_applied = "sentinel_best"
                advice.intervention_detail = (
                    f"Sentinel best move — {advice.advisory_message} (gap={gap:.0%})"
                )
                _logger.info(
                    "[Sentinel] intervened: engine intended %s → redirected to %s "
                    "(type: sentinel_best, gap: %.2f)",
                    self._move_notation(move), self._move_notation(best_move), gap,
                )
                return best_move
        return move

    def _db_score_adjust(self, board: BoardState, move: dict, moves: list) -> dict:
        """Pick the best move using perfect/solved DB WDL.  Query order (highest authority first):
        Malom perfect DB → endgame solved DB → fullgame DB.  Falls back to the original move if
        no DB is available or no position is found."""
        malom = self._malom_db
        esdb  = self._endgame_solved_db
        fgdb  = self._fullgame_db
        _db_gate = getattr(self, "_db_active_this_move", True)
        malom_ok = malom is not None and malom.is_available() and _db_gate
        esdb_ok  = esdb is not None and _db_gate
        if not malom_ok and not esdb_ok and not (fgdb and fgdb.is_available()):
            return move
        try:
            wdl_map = {"W": 2, "D": 1, "L": 0}
            _flip   = {"W": "L", "L": "W", "D": "D"}
            best_move  = move
            best_score = -1
            for m in moves:
                try:
                    after = board.apply_move(m)
                except Exception:
                    continue
                # Query from opponent's POV (after the move it's their turn), then flip
                res = None
                if malom_ok:
                    res = malom.query(after)
                if res is None and esdb_ok:
                    res = esdb.query(after)
                if res is None and fgdb and fgdb.is_available():
                    res = fgdb.query(after)
                if res:
                    score = wdl_map.get(_flip.get(res), -1)
                    if score > best_score:
                        best_score = score
                        best_move  = m
            if best_move != move:
                _logger.info(
                    "[Sentinel] DB override: engine intended %s → redirected to %s "
                    "(type: db_score_adjust, best_wdl_score: %d)",
                    self._move_notation(move), self._move_notation(best_move), best_score,
                )
            return best_move
        except Exception:
            return move

    def ban_move(self, notation: str, board_fen: str) -> None:
        """Ban `notation` from this exact board position only.

        If any piece moves or is captured the FEN changes and the ban
        no longer applies — the move is valid again from the new position.
        """
        self._pos_bans.setdefault(board_fen, set()).add(notation)

    def reset_game_bans(self) -> None:
        """Clear all per-game move bans (call when a new game starts)."""
        self._pos_bans.clear()
        # T-C4: also clear persistent Rust TT on new game so stale positions don't pollute.
        if self._rust_tt_handle is not None:
            try:
                self._rust_tt_handle.clear()
            except Exception:
                pass

    def force_stop(self) -> None:
        """Interrupt any running search immediately; _negamax raises _SearchAbort.
        Also sets _force_stop so the subsequent score_move() returns immediately.
        """
        self._force_stop = True
        self._deadline   = 0.0

    def score_root_moves(
        self,
        board: BoardState,
        depth: int = 3,
        time_budget: float = 2.0,
    ) -> list:
        """Score all legal moves with alpha-beta search at `depth`.

        Returns list of (move_dict, score_norm) sorted best-first, where
        score_norm is normalised to [0, 1] relative to the best/worst score
        at the root.  Intended as per-move feature input for the Overseer;
        no book, trajectory, or blunder logic is applied.
        Returns [] when there are no legal moves.
        """
        moves = get_all_legal_moves(board)
        if not moves:
            return []

        # Minimal state setup — mirrors choose_move preamble
        self._force_stop        = False
        self._deadline          = time.time() + time_budget
        self._tt.clear()
        self._killers           = [[None, None] for _ in range(32)]
        self._history           = {}
        self._nodes             = 0
        self._trajectory_db     = None
        self._game_notations    = []
        self._trajectory_line   = []
        self._move_path_buf     = []

        killers = self._killers[depth] if depth < 32 else None
        ordered = _order_moves(board, moves, killers, self._history,
                               _is_beginner=self._is_beginner)

        self._opp_plies_budget = _MAX_OPP_PLIES_V2 if self.use_v2_heuristics else _MAX_OPP_PLIES

        scored: list = []
        for mv in ordered:
            try:
                nb  = board.apply_move(mv)
                self._move_path_buf = [self._move_notation(mv)]
                raw = -self._negamax(
                    nb, depth - 1, -INF, INF, None,
                    depth // 2, self._opp_plies_budget, 1,
                )
            except (_SearchAbort, Exception):
                raw = 0.0
            scored.append((mv, float(raw)))

        if not scored:
            return []

        raws = [s for _, s in scored]
        lo, hi = min(raws), max(raws)
        span   = max(hi - lo, 1e-9)
        result = [(mv, (s - lo) / span) for mv, s in scored]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

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
        if self.suppress_fork_variety:
            import random as _r
            self._suppress_fork_this_move = _r.random() < 0.5
        self._tt.clear()
        self._killers = [[None, None] for _ in range(32)]
        self._history = {}
        self._trajectory_db = trajectory_db            # SE-11
        self._game_notations = list(game_notations) if game_notations else []  # SE-11
        # Phase 4: pre-fetch top trajectory moves for root ordering (lightweight).
        self._trajectory_line: list[tuple[str, float]] = (
            trajectory_db.query_line(board) if trajectory_db is not None else []
        )
        moves = get_all_legal_moves(board)
        _original_move_count = len(moves)   # for detecting later filtering (mandatory block, bans, …)
        if not moves:
            return {}
        if len(moves) == 1:
            self.last_was_blunder = False
            return moves[0]

        # Roll once per move so all DB probe sites share the same gate (avoids
        # mixing INF-scale DB scores with heuristic scores inside alpha-beta).
        import random as _rng
        self._db_active_this_move: bool = _rng.random() < self._db_access_prob

        # ── Optional retrograde endgame DB consultation ───────────────────
        # Consulted first (before fullgame_db) because WDL is exact.
        # EndgameSolvedDB.query() returns None for piece counts with no loaded table,
        # so no piece-count cap is needed here — the DB handles its own validity.
        _esdb = self._endgame_solved_db
        if _esdb is not None and _esdb.is_available() and self._db_active_this_move:
            if (board.pieces_placed.get("W", 0) >= 9
                    and board.pieces_placed.get("B", 0) >= 9):
                try:
                    _wdl = _esdb.query(board)
                except Exception:
                    _wdl = None
                if _wdl == "W":
                    self.last_was_blunder = False
                    # First: any immediate terminal win (capture → opponent has 2 pieces).
                    for _move in moves:
                        _succ = board.apply_move(_move)
                        _succ_terminal, _ = is_terminal(_succ)
                        if _succ_terminal:
                            self.last_thinking = "endgame DB (win)"
                            return _move
                    # Collect all DB-correct winning continuations, then let the search
                    # pick the BEST among them rather than returning the first found.
                    # This prevents choosing a slow/suboptimal win when a faster fork
                    # setup is available (e.g. preferring a7→d5+d7→c5 fork over a7→d6).
                    _db_winning: list[dict] = []
                    for _move in moves:
                        _succ = board.apply_move(_move)
                        try:
                            _succ_wdl = _esdb.query(_succ)
                        except Exception:
                            _succ_wdl = None
                        if _succ_wdl == "L":
                            _db_winning.append(_move)
                    if len(_db_winning) == 1:
                        self.last_thinking = "endgame DB (win)"
                        return _db_winning[0]
                    if _db_winning:
                        # Multiple winning moves: restrict search to DB-correct moves
                        # so the search can identify the fastest/cleanest win.
                        moves = _db_winning
                        # last_thinking set by _populate_thinking after search
                    # fall through to search (with moves restricted to winners if found)
                elif _wdl == "L":
                    self.last_thinking = "endgame DB (loss/search)"
                    # fall through to heuristic search for most stubborn defence
                elif _wdl == "D":
                    self.last_thinking = "endgame DB (draw)"

        # ── Optional full-game DB consultation ────────────────────────────
        # Falls back to self._fullgame_db when no explicit parameter passed.
        _db14_forced_notation: str | None = None   # B-78: set below, applied after filters
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
                # Resolved exact hit — stash best notation to apply after filters.
                # B-78: only return DB move when no capture is available this turn.
                if result.outcome is not None and result.best_move_canonical:
                    _db14_forced_notation = _fgdb.best_move(board)
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
            _base_blocking = [m for m in moves if m["to"] in threats]
            blocking = list(_base_blocking)
            # B-66 extended: when in move phase and own player can close a mill this
            # turn, also include mill-closing moves even if there are multiple threats.
            # Closing + capturing may eliminate one of the opponent's threats; the
            # search ranks closing vs blocking correctly.  Conservative: we still
            # restrict to {block-threats ∪ close-own-mill} rather than all moves.
            if board.phase == "move" and _stm_can_close_mill(board, board.turn):
                _close_sq: set[str] = {
                    next(p for p in _ml if board.positions[p] == "")
                    for _ml in MILLS
                    if ([board.positions[p] for p in _ml].count(board.turn) == 2
                        and [board.positions[p] for p in _ml].count("") == 1)
                }
                blocking = [m for m in moves if m["to"] in threats or m["to"] in _close_sq]
            # Cycling-mill exception: when STM has a closed mill and only ONE
            # opponent threat square is actually reachable (STM cannot block the
            # rest), allow cycling moves alongside the block.  Re-closing on the
            # next turn creates a capture threat as urgent as the single block.
            # Guard: STM must have > 3 pieces so it can absorb a capture.
            if board.phase == "move" and board.pieces_on_board.get(board.turn, 0) > 3:
                _reachable_threat_sqs = {m["to"] for m in _base_blocking if m["to"] in threats}
                if len(_reachable_threat_sqs) == 1:
                    _cycle_src: set[str] = set()
                    for _ml in MILLS:
                        if all(board.positions[p] == board.turn for p in _ml):
                            _cycle_src.update(_ml)
                    if _cycle_src:
                        _already = {m["to"] for m in blocking}
                        _cycling = [
                            m for m in moves
                            if m.get("from") in _cycle_src and m["to"] not in _already
                        ]
                        if _cycling:
                            blocking = blocking + _cycling
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
                # Junction-rescue: when live alternatives exist, also allow dead
                # placements whose square is the junction of ≥2 opponent developing
                # mills (all neighbours are opponent pieces).  Placing here blocks
                # the opponent's fork even though the piece gains no own mobility.
                # Deliberately excluded from the all-dead path below so the
                # mill-potential secondary filter is not disrupted.
                _opp_junc = "B" if board.turn == "W" else "W"
                _rescued = [
                    m for m in moves
                    if m.get("from") is None and _is_dead_placement(board, m)
                    and ADJACENCY.get(m["to"])
                    and all(
                        board.positions.get(nb) == _opp_junc
                        for nb in ADJACENCY[m["to"]]
                    )
                    and sum(
                        1 for _ml in MILLS
                        if m["to"] in _ml
                        and any(board.positions.get(p) == _opp_junc for p in _ml if p != m["to"])
                        and not any(board.positions.get(p) == board.turn for p in _ml if p != m["to"])
                    ) >= 2
                ]
                # Setup-rescue: also allow dead placements that gain a new
                # closeable 2-config — the placed piece is immobile but anchors a
                # mill that a different piece can close (e.g. placing at b2 with
                # d2=own means d2 can close b2-d2-f2 without b2 ever moving).
                _own = board.turn
                _cb_rescue = _closeable_mills(board, _own)
                _setup_rescued = [
                    m for m in moves
                    if m.get("from") is None and _is_dead_placement(board, m)
                    and m not in _rescued
                    and _closeable_mills(board.apply_move(m), _own) > _cb_rescue
                ]
                # Junction-rescue goes first (blocking priority), setup-rescue
                # second; both evaluated before the search deadline fires.
                moves = _rescued + _setup_rescued + non_dead
            else:
                # All placements are dead (late placement phase, board is packed).
                # Secondary filter: prefer squares with surviving mill potential —
                # at least one containing mill line not already blocked by an
                # opponent piece.  Squares like a7 (both lines blocked by opponent)
                # are discarded in favour of b6, f6, etc. that still have
                # plausible mill formations.
                with_potential = [
                    m for m in moves
                    if m.get("from") is None
                    and _dead_has_mill_potential(board, m["to"])
                ]
                if with_potential:
                    moves = with_potential

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
        # B-95: skip pin rules at difficulty ≤ 1 so the AI can blunder into traps.
        if get_game_phase(board, self.color) == "move" and self.difficulty > 1:
            pinned = _pinned_move_squares(board, self.color)
            if pinned:
                # Mill-closing exemption: allow departure from a pinned square when
                # the destination immediately closes an own mill (tactical gain outweighs pin).
                _mill_close_dests1: set[str] = set()
                for _ml in MILLS:
                    _vals1 = [board.positions.get(p, "") for p in _ml]
                    if _vals1.count(self.color) == 2 and _vals1.count("") == 1:
                        _mill_close_dests1.add(next(p for p in _ml if board.positions.get(p) == ""))
                unpinned = [
                    m for m in moves
                    if m.get("from") not in pinned or m["to"] in _mill_close_dests1
                ]
                if unpinned:
                    moves = unpinned
            # 2-ply pin: vacating this square lets opp build a 2-config in two moves.
            pinned2 = _pinned_move_squares_2ply(board, self.color)
            if pinned2:
                # Mill-closing exemption: allow a pinned-square departure when the
                # destination closes an own mill immediately — tactical gain outweighs pin.
                _mill_close_dests: set[str] = set()
                for _ml in MILLS:
                    _vals = [board.positions.get(p, "") for p in _ml]
                    if _vals.count(self.color) == 2 and _vals.count("") == 1:
                        _mill_close_dests.add(next(p for p in _ml if board.positions.get(p) == ""))
                unpinned2 = [
                    m for m in moves
                    if m.get("from") not in pinned2 or m["to"] in _mill_close_dests
                ]
                if unpinned2:
                    moves = unpinned2

        # Sentinel advisory pass: consult the learned overlay on the finalized
        # candidate set. Advisory mode only logs; it never changes the move. Fully
        # guarded inside _consult_sentinel so it can never break the game loop.
        if self.sentinel is not None:
            self._consult_sentinel(board, moves)

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

        # B-78: Apply deferred full-game DB forced move now that all filters have run.
        # Skip when a capture is available — capturing is always correct and must not be
        # overridden by a DB move that was recorded for a non-capture position.
        if _db14_forced_notation is not None:
            _has_capture = any(m.get("capture") for m in moves)
            if not _has_capture:
                match = next(
                    (m for m in moves if self._move_notation(m) == _db14_forced_notation),
                    None,
                )
                if match is not None:
                    self.last_was_blunder = False
                    self.last_thinking = "fullgame DB"
                    return match

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
            move = self._choose_rust_scored(board, early_max, recognition, trajectory_hints, moves,
                                            time_limit_ms=int(_EARLY_GAME_TIME * 1000), top_n=top_n) \
                or self._iterative_deepen(
                    board, _EARLY_GAME_TIME,
                    recognition=recognition, trajectory_hints=trajectory_hints,
                    top_n=top_n, moves=moves,
                    max_depth=early_max,
                )
            move = self._apply_sentinel_intervention(board, move, moves)
            self._populate_thinking(board, move, _forced_block=bool(threats))
            return move

        if self.difficulty in _TIME_LIMIT:
            _base_budget = (
                self._override_time_budget
                if self._override_time_budget is not None
                else (
                    self.time_budget_override
                    if self.time_budget_override is not None
                    else _TIME_LIMIT[self.difficulty]
                )
            )
            time_budget = 2.0 if fast_early_game else _base_budget
            move = self._choose_rust_scored(board, self.max_search_depth, recognition, trajectory_hints, moves,
                                            time_limit_ms=int(time_budget * 1000), top_n=top_n) \
                or self._iterative_deepen(
                    board, time_budget,
                    recognition=recognition, trajectory_hints=trajectory_hints,
                    top_n=top_n, moves=moves,
                    max_depth=self.max_search_depth,
                )
            move = self._apply_sentinel_intervention(board, move, moves)
            self._populate_thinking(board, move, _forced_block=bool(threats))
            return move

        depth = _DEPTH_TABLE[self.difficulty]

        # Deeper search in endgame for better tactical accuracy.
        # Skip in fast self-play mode to keep per-move time bounded.
        if endgame_state is not None and endgame_state.active and not fast_early_game:
            depth += 2 if endgame_state.deep else 1

        _vn_blend_active = self._value_net is not None and self._weights.value_net_blend > 0
        use_adjustments = (
            (recognition is not None and recognition.status not in ("novel", "inactive"))
            or (bool(trajectory_hints) and self._weights.opening_adherence > 0)
            or _vn_blend_active
        )
        if use_adjustments:
            scored = self._score_all(board, moves, depth, endgame_state=endgame_state)
            if recognition is not None:
                scored = self._apply_opening_adjustments(scored, recognition, board)
            if trajectory_hints:
                scored = self._apply_trajectory_hints(scored, trajectory_hints)
            if _vn_blend_active:
                scored = self._apply_vn_blend(scored, board)
            _var_pct = (self._weights.move_variance_pct if self._weights else 0)
            if _var_pct > 0 and scored:
                _best_sc = max(sc for _, sc in scored)
                if abs(_best_sc) < INF // 2:
                    _spread = _best_sc - min(sc for _, sc in scored)
                    _threshold = _best_sc - (_var_pct / 100.0) * max(1, _spread)
                    _candidates = [mv for mv, sc in scored if sc >= _threshold]
                    move = random.choice(_candidates)
                else:
                    move = max(scored, key=lambda x: x[1])[0]
            elif top_n > 1:
                scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
                move = random.choice(scored_sorted[:top_n])[0]
            else:
                move = max(scored, key=lambda x: x[1])[0]
            move = self._apply_sentinel_intervention(board, move, moves)
            self._populate_thinking(board, move, _forced_block=bool(threats))
            return move

        move, _ = self._root_search(board, depth, top_n=top_n, moves=moves)
        move = self._apply_sentinel_intervention(board, move, moves)
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
            if self.use_v2_heuristics:
                self.last_thinking = ""
                return
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

    def _apply_trajectory_hints(
        self,
        scored: list[tuple[dict, int]],
        hints: dict[str, float],
    ) -> list[tuple[dict, int]]:
        """Apply trajectory-database score deltas to a scored move list.

        Deltas in [-0.5, +0.5] are statistical hints scaled by opening_adherence.
        """
        if not hints:
            return scored
        adherence = self._weights.opening_adherence
        scale = int(3000 * adherence / 100) if adherence > 0 else 0
        adjusted = []
        for move, raw in scored:
            notation = self._move_notation(move)
            delta    = hints.get(notation, 0.0)
            # B-78: cap bonus so a trajectory hint cannot override a mill-close move.
            _bonus_cap = self._weights.close_mill - 1   # 499 < 500 (close_mill)
            bonus = min(int(delta * scale), _bonus_cap) if scale else 0
            adjusted.append((move, raw + bonus))
        return adjusted

    def _apply_vn_blend(
        self,
        scored: list[tuple[dict, int]],
        board: "BoardState",
    ) -> list[tuple[dict, int]]:
        """Blend value-network score into root move scores for ordering.

        VN runs once per root move (not at every leaf), so cost is negligible
        (~5–25 calls × 13µs = <1ms). Terminal scores (|s| >= INF/2) are
        preserved unchanged so mate distances aren't distorted.
        """
        blend = self._weights.value_net_blend / 100.0
        adjusted = []
        for move, raw in scored:
            if abs(raw) >= 5_000_000:  # terminal/INF score — don't distort
                adjusted.append((move, raw))
                continue
            succ = board.apply_move(move)
            vn_raw = self._value_net.predict(succ, board.turn)  # (-1, 1)
            vn_score = int(vn_raw * _VN_SCALE)
            blended = int((1.0 - blend) * raw + blend * vn_score)
            adjusted.append((move, blended))
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

        # Suppress book bonus when the book move's raw negamax score is negative
        # (AI already losing in that line — don't force a bad opening move).
        _book_dest_losing = False
        if book_dest and scored:
            book_raws = [raw for mv, raw in scored if mv.get("to", "") == book_dest]
            if book_raws and max(book_raws) < 0:
                _book_dest_losing = True

        adjusted = []
        for move, raw in scored:
            dest = move.get("to", "")
            delta = 0
            if book_dest and dest == book_dest:
                # Suppress bonus when landing on a dead square during placement,
                # or when the book move is already evaluated as losing.
                is_placement = not move.get("from")
                if not (is_placement and _book_dest_dead) and not _book_dest_losing:
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
        moves = _order_moves(board, moves, killers, self._history, _is_beginner=self._is_beginner)

        # Phase 4: promote top trajectory-line moves to the front of the root list.
        # Sort is stable: top-trajectory moves come first, existing order preserved within each tier.
        if self._trajectory_line:
            _top = {n for n, _ in self._trajectory_line[:3]}
            moves.sort(key=lambda mv: 0 if self._move_notation(mv) in _top else 1)

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
        self._opp_plies_budget = _MAX_OPP_PLIES_V2 if self.use_v2_heuristics else _MAX_OPP_PLIES

        _var_pct = (self._weights.move_variance_pct if self._weights else 0)
        scored_any = False
        for move in moves:
            nb = board.apply_move(move)
            _root_mn = self._move_notation(move)
            self._move_path_buf.append(_root_mn)
            try:
                score_raw = -self._negamax(nb, depth - 1, -beta, -alpha_raw, None, depth // 2, self._opp_plies_budget, 1)
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
            if abs(score_raw) < INF // 2 and not self.use_v2_heuristics:
                score = score_raw + tactical_move_bonus(
                    board, nb, self.color, self._active_weights(), self._opp_last_weak
                )
            else:
                score = score_raw
            if top_n > 1 or _var_pct > 0:
                all_scored.append((move, score))
            if score > best_score:
                best_score = score
                best_move = move
            if score_raw > alpha_raw:
                alpha_raw = score_raw
            if alpha_raw >= beta:
                break

        if _var_pct > 0 and all_scored and abs(best_score) < INF // 2:
            _spread = best_score - min(s for _, s in all_scored)
            _threshold = best_score - (_var_pct / 100.0) * max(1, _spread)
            _candidates = [mv for mv, sc in all_scored if sc >= _threshold]
            best_move = random.choice(_candidates)
        elif top_n > 1 and all_scored:
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

        # Fast terminal: O(1) piece-count check for captures.
        # Blockade is handled below by `if not moves` after move generation.
        _col = board.turn
        _opp_t = "B" if _col == "W" else "W"
        if board.pieces_placed[_col] == 9 and board.pieces_on_board[_col] < 3:
            return -(INF - ply)
        if board.pieces_placed[_opp_t] == 9 and board.pieces_on_board[_opp_t] < 3:
            return (INF - ply)

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

        # SE-4: endgame tablebase probe at all depths — ply-based scoring so faster
        # wins score higher than slower ones (INF - ply decreases as ply increases).
        # Moved before SE-8 extension so extensions don't bypass the DB hit.
        # Probe all available tables — EndgameSolvedDB.query() returns None when the
        # (nW, nB) combination has no loaded table, so no piece-count cap is needed here.
        if (self._endgame_solved_db is not None
                and self._endgame_solved_db.is_available()
                and getattr(self, "_db_active_this_move", True)
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
            _q_moves = get_all_legal_moves(board)
            if any(m.get("capture") for m in _q_moves):
                _qdepth = self._Q_DEPTH
                heur = self._qsearch(board, _qdepth, alpha, beta, endgame_state, _q_moves)
            else:
                if self.use_v2_heuristics:
                    heur = evaluate_v2(board, board.turn, weights=self._weights, _ply=ply)
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

        # ── Null-move pruning (V2 mode only) ─────────────────────────────────
        # Skip during: placement phase, endgame (total pieces <= 7), zugzwang risk
        # (own mobility < 3).  R = 2 (standard for NMM's branching factor).
        # _all_placed guard is critical: during placement swap_turn() is invalid.
        _NULL_R = 2
        if (self.use_v2_heuristics
                and depth >= 3
                and abs(alpha) < INF // 2
                and abs(beta) < INF // 2):
            _all_placed = (board.pieces_placed.get("W", 0) >= 9
                           and board.pieces_placed.get("B", 0) >= 9)
            if _all_placed:
                _total_pieces = sum(board.pieces_on_board.values())
                _in_endgame = _total_pieces <= 7
                _own_mob = 0
                _bpos = board.positions
                _bturn = board.turn
                for _sq in POSITIONS:
                    if _bpos[_sq] == _bturn:
                        for _nb in ADJACENCY[_sq]:
                            if not _bpos[_nb]:
                                _own_mob += 1
                                if _own_mob >= 3:
                                    break
                    if _own_mob >= 3:
                        break
                if _own_mob >= 3 and not _in_endgame:
                    null_board = board.swap_turn()
                    null_score = -self._negamax(
                        null_board, depth - 1 - _NULL_R,
                        -beta, -beta + 1,
                        endgame_state, 0, 0, ply + 1,
                    )
                    if null_score >= beta:
                        return beta

        # Sort at upper levels only — biggest benefit to alpha-beta, negligible overhead
        if depth >= 2:
            killers = self._killers[depth] if depth < 32 else None
            moves = _order_moves(board, moves, killers, self._history, _is_beginner=self._is_beginner)

        # IID (V2 mode only): when the TT has no hit at deep nodes, run a shallow
        # search to produce a TT entry whose best-move improves ordering quality.
        # The recursive call stores a TT entry; we immediately look it up and treat
        # the stored best-move as a TT hint so the existing promotion block below
        # can move it to the front of the list.
        if (self.use_v2_heuristics
                and depth >= 5
                and tt_move_to is None):
            try:
                self._negamax(
                    board, depth - 3, alpha, beta,
                    endgame_state, 0, 0, ply,
                )
                _iid_entry = self._tt.lookup(board.hash_key)
                if _iid_entry:
                    _, _, _, _iid_from, _iid_to = _iid_entry
                    if _iid_to:
                        tt_move_from, tt_move_to = _iid_from, _iid_to
            except (_SearchAbort, Exception):
                pass

        # SE-11b/11c: classify node and prepare trajectory/VN extension state.
        is_opp_node = (board.turn != self.color)
        _do_path = opp_plies_left > 0
        _next_opp_plies = max(0, opp_plies_left - 1) if is_opp_node else opp_plies_left

        # SE-11b: query trajectory frequency dict at first opponent ply only (307µs/call — too
        # expensive at ply 2 where ~27k nodes × 307µs ≈ 8 s overhead).
        _opp_freq = None
        if is_opp_node and opp_plies_left == self._opp_plies_budget and self._trajectory_db is not None:
            _opp_freq = self._trajectory_db.query_all_frequencies(
                board, min_samples=3
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
        # Gated to first opponent ply only (opp_plies_left == self._opp_plies_budget) to contain overhead.
        if (is_opp_node and opp_plies_left == self._opp_plies_budget
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

        if self.use_v2_heuristics:
            stand_pat = evaluate_v2(board, board.turn, weights=self._weights)
        else:
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
        self._opp_plies_budget = _MAX_OPP_PLIES_V2 if self.use_v2_heuristics else _MAX_OPP_PLIES
        results = []
        for i, move in enumerate(moves):
            nb = board.apply_move(move)
            _root_mn = self._move_notation(move)
            self._move_path_buf.append(_root_mn)
            try:
                score = -self._negamax(nb, depth - 1, -INF, INF, endgame_state, depth // 2, self._opp_plies_budget, 1)
                if abs(score) < INF // 2 and not self.use_v2_heuristics:
                    score += tactical_move_bonus(board, nb, self.color, self._active_weights(), self._opp_last_weak)
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
        Select a suboptimal-but-not-catastrophic move from the middle band of
        scored moves: skip the top ~30% (best moves) and the bottom ~30%
        (obvious disasters), then pick randomly from what remains.
        Uses a shallow fixed depth with a hard time cap.
        """
        self._deadline = time.time() + self._BLUNDER_TIME
        scored = self._score_all(board, moves, self._BLUNDER_DEPTH)
        self._deadline = math.inf
        scored.sort(key=lambda x: x[1], reverse=True)  # descending: best first
        n = len(scored)
        lo = max(0, round(n * 0.30))   # skip top 30%
        hi = max(lo + 1, round(n * 0.70))  # keep up to 70th percentile
        pool = scored[lo:hi] or scored  # fallback: all moves if pool is empty
        return random.choice(pool)[0]

    def diagnostic_scores(
        self,
        board: "BoardState",
        depth: int = 3,
        game_notations: list | None = None,
    ) -> list[dict]:
        """Score all legal moves at *depth* for the diagnostic overlay.

        Returns [{from, to, capture, score}, ...] sorted best-first.
        Safe to call between human turns — resets and restores search state.
        """
        from game.rules import get_all_legal_moves
        moves = get_all_legal_moves(board)
        if not moves:
            return []
        # Mirror choose_move preamble so _negamax has all state it needs.
        self._force_stop = False
        self._deadline   = time.time() + 20.0
        self._tt.clear()
        self._killers        = [[None, None] for _ in range(32)]
        self._history        = {}
        self._trajectory_db  = None          # skip trajectory during diagnostics
        self._trajectory_line = []
        self._game_notations = list(game_notations) if game_notations else []
        self._opp_last_weak  = False
        # Score from the perspective of whoever is to move — may differ from self.color
        # (e.g. scoring the human player's options). Temporarily override self.color so
        # _score_all's tactical_move_bonus and is_opp_node classification are correct.
        orig_color = self.color
        self.color = board.turn
        try:
            results = self._score_all(board, moves, depth)
        finally:
            self.color = orig_color
            self._deadline = math.inf
        results.sort(key=lambda x: x[1], reverse=True)
        return [
            {"from": m.get("from"), "to": m["to"], "capture": m.get("capture"), "score": int(s)}
            for m, s in results
        ]

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
        clear_eval_cache()  # SE-12: fresh cache per depth iteration
        _p_t0 = time.perf_counter()
        print(f"P:START budget={time_limit:.1f}s max_depth={max_depth}", flush=True)
        if moves is None:
            moves = get_all_legal_moves(board)
        best_move     = moves[0]
        use_adjustments = (
            (
                recognition is not None
                and recognition.status not in ("novel", "inactive")
            ) or (bool(trajectory_hints) and self._weights.opening_adherence > 0)
        )

        prev_score: int | None = None
        _vn_blend_active = self._value_net is not None and self._weights.value_net_blend > 0
        use_adjustments = use_adjustments or _vn_blend_active

        last_completed_depth = 1
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
                    if _vn_blend_active:
                        scored = self._apply_vn_blend(scored, board)
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
                last_completed_depth = depth
            except _SearchAbort:
                break                     # deadline hit mid-depth; keep previous best
        self._deadline = math.inf
        self.last_depth_reached = last_completed_depth
        _p_dt = time.perf_counter() - _p_t0
        print(f"P:END depth={last_completed_depth} t={_p_dt:.2f}s", flush=True)
        return best_move

    def _choose_rust_scored(
        self,
        board: BoardState,
        max_depth: int,
        recognition=None,
        trajectory_hints: "dict | None" = None,
        moves: "list | None" = None,
        time_limit_ms: "int | None" = None,
        top_n: int = 1,
    ) -> "dict | None":
        """Rust negamax returning per-move scores; Python applies hint semantics on top.

        Calls py_search_root_scored (full-window per-move scores), filters the
        result to the provided moves list, applies opening-book and trajectory
        bonuses, and picks the best remaining move.
        Returns None on Rust failure so the caller falls back to Python search.
        """
        _t0 = time.perf_counter()
        try:
            from . import native_core as _nc
            if not _nc.RUST_AVAILABLE:
                print("R:UNAVAILABLE (nmm_core not importable)", flush=True)
                return None
            import nmm_core as _rc
            white, black, wp, bp, stm = _nc.board_to_bits(board)
            if time_limit_ms is None:
                if self.time_budget_override is not None:
                    time_limit_ms = min(300_000, int(self.time_budget_override * 1000))
                else:
                    time_limit_ms = min(300_000, max_depth * max_depth * 300)

            # T-C4: lazy-init persistent Rust TT handle (reused across turns; cleared on new game).
            if self._rust_tt_handle is None:
                try:
                    self._rust_tt_handle = _rc.RustTtHandle()
                except Exception:
                    pass

            # T-C2: lazy-init mmap'd fullgame DB handle.
            if self._rust_fullgame_db_handle is None and self._fullgame_db is not None:
                try:
                    self._rust_fullgame_db_handle = _rc.FullgameDbHandle.open(
                        str(self._fullgame_db.path)
                    )
                except Exception:
                    pass

            # T-C3: lazy-init endgame solved DB handle (mmap'd .wdl files).
            if self._rust_endgame_solved_handle is None and self._endgame_solved_db is not None:
                try:
                    _esdb_dir = str(self._endgame_solved_db.db_dir)
                    self._rust_endgame_solved_handle = _rc.EndgameSolvedDbHandle.open(_esdb_dir)
                except Exception:
                    pass

            # T-C1: collect high-frequency opponent moves for SE-11b depth extension.
            _opp_ext: list = []
            if self._trajectory_db is not None:
                try:
                    _root_moves = moves if moves is not None else get_all_legal_moves(board)
                    for _mv in _root_moves:
                        _nb = board.apply_move(_mv)
                        _freqs = self._trajectory_db.query_all_frequencies(_nb)
                        for _notation, _freq in _freqs.items():
                            if _freq >= 0.5:
                                _triple = _notation_to_triple(_notation)
                                if _triple is not None:
                                    _opp_ext.append(_triple)
                except Exception:
                    pass

            # M3: preferred_root from trajectory line (top-3 moves promoted in Rust ordering).
            _preferred: list = []
            for _nota, _conf in self._trajectory_line[:3]:
                _t = _notation_to_triple(_nota)
                if _t is not None:
                    _preferred.append(_t)

            _threads = self.search_threads if self.search_threads > 1 else None
            _nodes, depth, raw_moves = _rc.py_search_root_scored(
                white, black, wp, bp, stm, max_depth, time_limit_ms,
                preferred_root=_preferred if _preferred else None,
                tt_handle=self._rust_tt_handle,
                db_handle=self._rust_fullgame_db_handle,
                endgame_db_handle=self._rust_endgame_solved_handle,
                opp_ext_moves=_opp_ext if _opp_ext else None,
                threads=_threads,
                mill_scale=self._weights.mill_count_scale,
                mob_scale=self._weights.mobility_scale,
                block_scale=self._weights.blocked_scale,
            )
            _dt = time.perf_counter() - _t0
            if not raw_moves:
                print(f"{self._search_label}:NO-MOVE depth={depth} nodes={_nodes} t={_dt:.2f}s (falling back to Python)", flush=True)
                return None

            # T-D2: filter raw index tuples before allocating move dicts.
            if moves is not None:
                allowed_idx = {
                    (
                        _POS_TO_IDX.get(m["from"]) if m.get("from") else None,
                        _POS_TO_IDX[m["to"]],
                        _POS_TO_IDX.get(m["capture"]) if m.get("capture") else None,
                    )
                    for m in moves
                }
                raw_moves = [(frm, to, cap, s) for frm, to, cap, s in raw_moves
                             if (frm, to, cap) in allowed_idx]
                if not raw_moves:
                    print(f"{self._search_label}:FILTERED-OUT depth={depth} nodes={_nodes} t={_dt:.2f}s (falling back to Python)", flush=True)
                    return None

            # Convert surviving Rust (from_idx, to_idx, cap_idx, score) tuples to (move_dict, score).
            scored: list[tuple[dict, int]] = [
                (
                    {
                        "from": None if frm is None else POSITIONS[frm],
                        "to":   POSITIONS[to],
                        "capture": None if cap is None else POSITIONS[cap],
                    },
                    score,
                )
                for frm, to, cap, score in raw_moves
            ]

            # Apply Python-side bonuses.
            n_bonuses = 0
            if recognition is not None and recognition.status not in ("novel", "inactive"):
                scored = self._apply_opening_adjustments(scored, recognition, board)
                n_bonuses += 1
            if trajectory_hints:
                scored = self._apply_trajectory_hints(scored, trajectory_hints)
                n_bonuses += 1
            if self._value_net is not None and self._weights.value_net_blend > 0:
                scored = self._apply_vn_blend(scored, board)
                n_bonuses += 1

            if top_n > 1:
                best_move = random.choice(sorted(scored, key=lambda x: x[1], reverse=True)[:top_n])[0]
            else:
                best_move = max(scored, key=lambda x: x[1])[0]
            self.last_depth_reached = depth
            self._nodes = _nodes  # expose Rust node count to test observers
            _cap_str = f" cap={best_move['capture']}" if best_move.get("capture") else ""
            print(f"{self._search_label}:OK depth={depth} nodes={_nodes} t={_dt:.2f}s "
                  f"to={best_move['to']}{_cap_str} adjusted={n_bonuses}",
                  flush=True)
            return best_move
        except Exception:
            _dt = time.perf_counter() - _t0
            _logger.exception("%s:FAIL after %.2fs — falling back to Python", self._search_label, _dt)
            print(f"{self._search_label}:FAIL after {_dt:.2f}s (see traceback above) — falling back to Python", flush=True)
            return None

    def position_eval(self, board: BoardState) -> float:
        """Return tanh-normalised score in (-1, +1): positive = White winning.

        Delegates to evaluate(strength_mode=True) which uses phase-calibrated
        scales (800/1500/3000) and returns ±1.0 for terminal positions.
        """
        from .heuristics import evaluate as _eval
        return _eval(board, "W", strength_mode=True)
