"""
ai/heuristics.py — Phase-weighted board evaluation for Nine Men's Morris.

evaluate(board, color) returns an integer score from color's perspective.
Positive = good for color, negative = bad.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from game.board import ADJACENCY, MILLS, POSITIONS, BoardState
from game.rules import get_game_phase, is_terminal


@dataclass
class HeuristicWeights:
    """Configurable tactical and positional weights sent from the UI."""
    # ── Tactical urgency (delta-based, applied per move) ─────────────────
    close_mill: int            = 500   # bonus per mill closed this move
    cycling_mill: int          = 300   # bonus for gaining a cycling mill setup (capped at 1 per move)
    block_opponent_mill: int   = 400   # bonus per opponent closeable mill neutralised
    stop_opponent_mills: int   = 450   # bonus per opponent 2-config dismantled
    feeder_diamond: int        = 200   # bonus for gaining a diamond/fork structure (capped at 1 per move)
    mill_wrapping: int         = 150   # bonus per own piece surrounding an opponent closed mill
    cardinal_block: int        = 200   # bonus for taking/clearing cross-node squares
    scatter_placement: int    = 75    # bonus for non-adjacent placement in first 6 moves
    setup_mill: int           = 100   # bonus per new two-config gained this move (placement phase)
    mill_opening: int         = 200   # bonus for opening a cycling-ready mill (enables next capture)
    mill_trap_build: int      = 180   # bonus for adding a 3rd+ open mill when already dominant (endgame)
    capture_disrupt_feeder: int = 300  # bonus when captured piece was a feeder for an opponent cycling mill
    capture_disrupt_diamond: int = 250 # bonus when captured piece was part of an opponent fork (diamond)
    # ── Positional base scale (applied inside evaluate) ──────────────────
    long_term_position: int   = 100   # % multiplier on entire positional base score
    mill_count_scale: int     = 100   # % multiplier on mill-count weights
    mobility_scale: int       = 100   # % multiplier on mobility weights
    blocked_scale: int        = 100   # % multiplier on blocked-pieces weights
    # ── Placement busy-opponent scan-ahead ──────────────────────────────────
    placement_busy_scan: int  = 120   # base weight per chain level (level-1 is free; 2,3,4 score ×1,2,3)
    # ── Convergence cluster block ────────────────────────────────────────────
    convergence_block: int    = 250   # bonus per opponent convergence cluster disrupted this placement
    # ── 6v4 sacrifice-to-fly ────────────────────────────────────────────────
    sacrifice_viable: int      = 200  # bonus in 6v4 when a strong 3-piece fly structure exists
    # ── Double-mill convergence prevention ──────────────────────────────────
    convergence_penalty: int   = 180  # penalty per opp fork precursor pair in move phase
    convergence_disrupt: int   = 220  # bonus per opp fork precursor pair broken this move
    # ── Ring crowding penalty ────────────────────────────────────────────────
    ring_crowding_penalty: int = 150  # penalty for placing 6th+ own piece on a single ring
    # ── Herding / mobility squeeze ───────────────────────────────────────────
    herding_squeeze: int       = 60   # bonus per opponent piece with exactly 1 legal move
    mobility_reduction: int    = 15   # bonus per opponent legal move removed this turn
    # ── Placement chain deferral (B-2) ──────────────────────────────────────
    defer_for_chain: int      = 300   # extra bonus for skipping a mill to execute a level-4 chain
    # ── Fork anticipation (B-4) ─────────────────────────────────────────────
    fork_anticipation: int    = 90    # bonus for blocking a square that would give opp a 2-move fork
    # ── Locked mill / redirected pin (B-7) ───────────────────────────────────
    locked_mill_penalty: int  = 80    # penalty per own locked mill (no exit squares) in evaluate()
    locked_mill_escape: int   = 160   # bonus for moving out of a locked mill toward a new 2-config
    redirected_pin: int       = 140   # bonus when a move double-pins an opponent blocker
    # ── Forked-mill cycling priority (B-8) ───────────────────────────────────
    block_cycling_priority: int = 120 # bonus for blocking the higher-cycling-freedom fork arm
    # ── Trajectory exploit (B-6, consumed by Coordinator) ────────────────────
    loss_exploit: int         = 150   # how strongly to exploit opponent losing-line trajectories
    # ── Cross-feeding 2-config pairs (B-16) ──────────────────────────────────
    own_convergence: int      = 250   # bonus per own pair sharing closing sq or pivot piece
    cross_feed_mobility: int  = 180   # bonus per own pair where a piece is adjacent to other's closing sq
    # ── Behaviour (consumed by GameAI, not heuristics) ───────────────────
    make_mistakes: int        = 0     # blunder probability 0-100 %
    opening_adherence: int    = 50    # how strongly to follow the opening book (0-100)


DEFAULT_WEIGHTS = HeuristicWeights()

INF: int = 10_000_000

# Phase weights: (closed_mills, blocked_opp, piece_diff, two_cfg, dbl_mill, win_cfg)
#
# KEY INVARIANT: mill_w > two_cfg + THREAT_WEIGHT
# Closing a mill consumes a two-config, so the net gain from closing must be
# positive: mill_w - (two_cfg + THREAT) > 0.  Here: 30 > (5+15)=20 ✓
# two_cfg is kept small so the primary two-config signal comes from the
# reachability-aware THREAT term (closeable mills only).
_WEIGHTS = {
    "place": (30,  12, 12,  5,   0,    0),
    "move":  (30,  48, 12,  5,  50,    0),
    "fly":   (32, 350,  2,  0,  90, 1190),
}

# Mobility and threat term weights per phase.
# _THREAT_WEIGHTS weights CLOSEABLE mills (reachable in one move only),
# giving an additional urgency signal on top of the structural two_cfg baseline.
_MOB_WEIGHTS    = {"place": 3,  "move": 8,  "fly": 20}
_THREAT_WEIGHTS = {"place": 15, "move": 18, "fly": 80}

# tanh normalization scales per phase (used by position_eval display, not search)
TANH_SCALE = {"place": 120, "move": 180, "fly": 280}

# Cardinal nodes: 4 connections each — highest mobility AND participate in 2 mills.
# These are the middle-ring midpoints: b4, d2, d6, f4.
_CARDINAL_NODES = frozenset({"b4", "d2", "d6", "f4"})

# Cross nodes: 3 connections each — outer and inner ring midpoints.
_CROSS_NODES_3 = frozenset({"d7", "g4", "d1", "a4", "d5", "e4", "d3", "c4"})

# Union used for cardinal_block bonus (rewards placing on any high-mobility node).
_CROSS_NODES = _CARDINAL_NODES | _CROSS_NODES_3

# Ring membership: outer / middle / inner concentric squares.
# Used to detect ring-crowding (4+ own pieces on one ring hurts mobility).
_RING_OUTER  = frozenset({"a7", "d7", "g7", "g4", "g1", "d1", "a1", "a4"})
_RING_MIDDLE = frozenset({"b6", "d6", "f6", "f4", "f2", "d2", "b2", "b4"})
_RING_INNER  = frozenset({"c5", "d5", "e5", "e4", "e3", "d3", "c3", "c4"})
_RINGS = (_RING_OUTER, _RING_MIDDLE, _RING_INNER)

# Outer-ring side mills: each contains two corner nodes (a7/g7/g1/a1) that
# have only 2 connections.  Closing one of these during early placement locks
# two pieces into low-mobility corner squares, hurting movement-phase options.
_OUTER_MILLS = frozenset(
    frozenset(m) for m in [
        ("a7", "d7", "g7"),
        ("g7", "g4", "g1"),
        ("g1", "d1", "a1"),
        ("a1", "a4", "a7"),
    ]
)

# Inner-ring mills: entirely on the innermost square.  Closing one of these
# confines your pieces to the inner ring and reduces long-term mobility, so
# late-placement mill urgency is NOT boosted for these.
_INNER_MILLS = frozenset(
    frozenset(m) for m in [
        ("c5", "d5", "e5"), ("e5", "e4", "e3"),
        ("e3", "d3", "c3"), ("c3", "c4", "c5"),
    ]
)

# Mill-cycle readiness: a closed mill with a slide-out square enables repeated
# captures (open/close each cycle).  Highest value in fly; still relevant in move.
_CYCLE_WEIGHTS = {"place": 8, "move": 22, "fly": 80}

# Fork-threat: a piece in 2+ open mills simultaneously.  Opponent cannot defend
# both in one move, so one mill closes next turn regardless.
_FORK_WEIGHTS  = {"place": 6, "move": 14, "fly": 55}

# Herding / encirclement: own pieces adjacent to each opponent piece.
# Rewards progressively surrounding opponent pieces to shrink their escape space.
# Irrelevant in fly phase (pieces can jump anywhere).
_HERD_WEIGHTS  = {"place": 6, "move": 18, "fly": 0}

# Near-blocked pressure: opponent pieces with exactly 1 legal move remaining.
# These are one step from total blockade — the goal of the herding tactic.
# 0 in placement (mobility is constantly fluctuating during piece deployment)
# and fly (adjacency constraints don't apply; pieces jump freely).
_NEAR_BLOCKED_WEIGHTS = {"place": 0, "move": 30, "fly": 0}

# Mill wrapping: own pieces occupying exit squares of opponent closed mills.
# A surrounded mill cannot easily cycle — the pivot piece has nowhere to slide.
# Not meaningful in placement (few closed mills exist yet).
_WRAP_WEIGHTS  = {"place": 0, "move": 40, "fly": 60}

# Fly-phase asymmetry: reward entering fly (3 pieces) when the opponent hasn't yet,
# and penalise giving the opponent fly while we remain in move phase.
# At 4v4 the search will prefer sacrificing a piece (3v4, us in fly) over
# capturing an opponent piece (4v3, them in fly).
_FLY_ASYM_WEIGHTS = {"place": 0, "move": 80, "fly": 0}

# Open-mill domination: uncoverable 2-configs in the dominant asymmetric endgame.
# Active when own pieces ≥ 6 and opp pieces ≤ 5 (7v4, 7v3, 6v4, 6v3 scenarios).
_DOMINATION_WEIGHTS = {"place": 0, "move": 150, "fly": 80}


def evaluate(
    board: BoardState,
    color: str,
    endgame_state=None,
    force_aggressive: bool = False,
    weights: HeuristicWeights | None = None,
) -> int:
    """Evaluate board from `color`'s perspective. Higher is better for color."""
    terminal, winner = is_terminal(board)
    if terminal:
        return INF if winner == color else -INF

    opp   = "B" if color == "W" else "W"
    phase = get_game_phase(board, color)
    w     = _WEIGHTS[phase]

    # Apply per-weight UI scale factors
    mill_w  = int(w[0] * weights.mill_count_scale / 100) if weights else w[0]
    block_w = int(w[1] * weights.blocked_scale    / 100) if weights else w[1]
    mob_w   = int(_MOB_WEIGHTS[phase] * weights.mobility_scale / 100) if weights else _MOB_WEIGHTS[phase]

    our_mills  = _closed_mills(board, color)
    opp_mills  = _closed_mills(board, opp)
    blocked    = _blocked_count(board, opp)
    piece_diff = board.pieces_on_board[color] - board.pieces_on_board[opp]
    our_two    = _two_configs(board, color)
    opp_two    = _two_configs(board, opp)
    our_dbl    = _double_mills(board, color)
    opp_dbl    = _double_mills(board, opp)
    win_cfg    = _win_config(board, opp)
    our_mob    = _mobility(board, color)
    opp_mob    = _mobility(board, opp)
    our_thr    = _mill_threats(board, color)
    opp_thr    = _mill_threats(board, opp)
    our_pos    = _position_value(board, color)
    opp_pos    = _position_value(board, opp)
    our_cycle  = _mill_cycle_ready(board, color)
    opp_cycle  = _mill_cycle_ready(board, opp)
    our_fork   = _fork_threats(board, color)
    opp_fork   = _fork_threats(board, opp)
    our_herd   = _encirclement(board, color)
    opp_herd   = _encirclement(board, opp)
    our_squeeze = _squeeze_count(board, opp)   # opponent near-blocked (good for us)
    opp_squeeze = _squeeze_count(board, color) # our own near-blocked (bad for us)
    our_wrap   = _mill_wrapping_pressure(board, color)
    opp_wrap   = _mill_wrapping_pressure(board, opp)
    fly_asym   = 0 if force_aggressive else _fly_asymmetry(board, color)
    our_dom    = _open_mill_domination(board, color)
    opp_dom    = _open_mill_domination(board, opp)

    base = (
        mill_w  * (our_mills - opp_mills)
        + block_w *  blocked
        + w[2]  *  piece_diff
        + w[3]  * (our_two  - opp_two)
        + w[4]  * (our_dbl  - opp_dbl)
        + w[5]  *  win_cfg
        + mob_w                  * (our_mob - opp_mob)
        + _THREAT_WEIGHTS[phase] * (our_thr - opp_thr)
        + 4 * (our_pos - opp_pos)
        + _CYCLE_WEIGHTS[phase]  * (our_cycle - opp_cycle)
        + _FORK_WEIGHTS[phase]   * (our_fork  - opp_fork)
        + _HERD_WEIGHTS[phase]   * (our_herd  - opp_herd)
        + _NEAR_BLOCKED_WEIGHTS[phase] * (our_squeeze - opp_squeeze)
        + _WRAP_WEIGHTS[phase]   * (our_wrap  - opp_wrap)
        + _FLY_ASYM_WEIGHTS[phase]   * fly_asym
        + _DOMINATION_WEIGHTS[phase] * (our_dom - opp_dom)
    )

    # Asymmetric endgame: boost two-config weight when piece-count dominant (7v4, 6v4).
    # The global two_cfg weight is kept low for balance across all phases; this correction
    # raises it only when spreading open mills is the primary winning mechanism.
    own_pieces = board.pieces_on_board[color]
    opp_pieces = board.pieces_on_board[opp]
    if own_pieces >= 6 and opp_pieces <= 4 and phase == "move":
        base += 25 * (our_two - opp_two)

    # 4v3: when facing a fly-mobile opponent, a fork (dual closeable mills) is the
    # primary winning mechanism — one threat is always blockable, two are not.
    # Boost fork weight and reward independent mill pairs heavily (insurance
    # so any single opponent capture still leaves us with a 2-config).
    opp_in_fly = board.pieces_placed.get(opp, 0) >= 9 and opp_pieces == 3
    if own_pieces == 4 and opp_in_fly and phase == "move":
        base += 40 * our_fork
        base += 300 * _independent_mill_pairs(board, color)  # was 90
        # Opponent fly threats are as dangerous as fly-phase threats even though
        # we are in move phase — apply the fly-urgency gap as an extra penalty.
        base -= (_THREAT_WEIGHTS["fly"] - _THREAT_WEIGHTS["move"]) * opp_thr

    # 3v4: fly attacker rewards separated opponent groups — disconnected pieces
    # can't defend each other, allowing the fly attacker to threaten one group at a time.
    # Suppressed when force_aggressive=True so the AI isn't penalised for giving fly.
    if phase == "fly" and opp_pieces == 4 and not force_aggressive:
        base += 180 * _piece_separation(board, color)

    # Fly fork: when both players are in fly phase, each 2-config is an immediate
    # threat (closeable in 1 move).  We can block at most 1 per turn, so any
    # surplus threats the opponent holds are essentially guaranteed captures.
    # Penalise (or reward) each uncoverable surplus threat heavily.
    if phase == "fly" and get_game_phase(board, opp) == "fly":
        opp_surplus = max(0, opp_thr - 1)  # we block 1 max; remainder are automatic
        own_surplus = max(0, our_thr - 1)
        base += 900 * (own_surplus - opp_surplus)  # was 600

    # 6v4 sacrifice-to-fly: when own has 6 pieces and opp has 4 and a strong
    # 3-piece fly nucleus already exists (closed mill or 2-config), reward the
    # position.  This nudges the search toward trades that reach a winning 3v3
    # rather than passively defending all 6 pieces until the search horizon.
    # Guard: only fires when 6v3 domination (≥3 own open mills) is not yet
    # available — once that path is open, the zugzwang plan is better.
    w_sac = weights.sacrifice_viable if weights else DEFAULT_WEIGHTS.sacrifice_viable
    if own_pieces == 6 and opp_pieces == 4 and phase == "move" and our_two < 3 and not force_aggressive:
        sq = _fly_sacrifice_quality(board, color)
        if sq > 0:
            base += w_sac * sq

    # Move-phase double-mill convergence penalty: opponent two-configs that share
    # a closing square (diamond) or a pivot piece are one move from an unblockable
    # fork.  Penalise these precursors to force the AI to disrupt them proactively.
    w_conv = weights.convergence_penalty if weights else DEFAULT_WEIGHTS.convergence_penalty
    if phase == "move":
        base -= w_conv * _double_mill_convergence(board, opp)

    # Cross-feeding 2-config pair bonus (B-16): reward own positions where two
    # independent 2-config groups mutually sustain each other — whichever group
    # the opponent attacks, the survivor can complete the other group's mill.
    #
    # Two sub-cases:
    #   own_convergence    — pairs sharing a closing square or pivot piece
    #                        (same computation as opp convergence, applied offensively)
    #   cross_feed_mobility — pairs where the closing squares differ but a piece
    #                         from one group is adjacent to the other's closing sq
    #
    # Applied in move and fly phases; the 4v3 independent_mill_pairs bonus already
    # covers placement-phase insurance, so we skip double-counting there.
    if phase in ("move", "fly"):
        w_own_conv = weights.own_convergence if weights else DEFAULT_WEIGHTS.own_convergence
        w_cfm      = weights.cross_feed_mobility if weights else DEFAULT_WEIGHTS.cross_feed_mobility
        base += w_own_conv * (_double_mill_convergence(board, color) - _double_mill_convergence(board, opp))
        base += w_cfm * (_cross_feed_mobility_pairs(board, color) - _cross_feed_mobility_pairs(board, opp))

    # Move-phase: reward non-contributing pieces assembling toward a 2-config.
    # Gradient: step-1 (×65), step-2 (×22), step-3 (×10), step-4 (×4).
    # Also reward free pieces approaching empty squares of 1-config mills (×12 weighted
    # score: step-1 from empty sq = 2 pts, step-2 = 1 pt in the helper).
    if phase == "move":
        base += 65 * (_free_piece_assembly(board, color) - _free_piece_assembly(board, opp))
        base += 22 * (_assembly_reach_count(board, color) - _assembly_reach_count(board, opp))
        base += 10 * (_assembly_step3_count(board, color) - _assembly_step3_count(board, opp))
        base +=  4 * (_assembly_step4_count(board, color) - _assembly_step4_count(board, opp))
        base += 12 * (_one_config_approach(board, color) - _one_config_approach(board, opp))

    # Move-phase locked-mill penalty: penalise each own closed mill that has no
    # exit squares (every neighbour is opponent-occupied).  These mills contribute
    # zero cycling value and represent stranded material.
    w_lmp = weights.locked_mill_penalty if weights else DEFAULT_WEIGHTS.locked_mill_penalty
    if phase == "move" and w_lmp > 0:
        for mill in MILLS:
            if all(board.positions[p] == color for p in mill):
                if _is_mill_locked(board, color, mill):
                    base -= w_lmp

    # Fly-phase structural bonuses: reward interpose pieces (own between two opp in a
    # mill line — blocks opp from closing while enabling own jump-to-close tactics) and
    # perpendicular-block pieces (own 2-config piece that simultaneously blocks an opp
    # 2-config, yielding double structural value from a single piece).
    if phase == "fly":
        base += 100 * (_interpose_count(board, color) - _interpose_count(board, opp))
        base += 80 * (_perp_block_count(board, color) - _perp_block_count(board, opp))

    # Apply overall positional scale (long_term_position=100 means no change)
    if weights and weights.long_term_position != 100:
        base = int(base * weights.long_term_position / 100)

    return base + _late_game_danger(board, color) + endgame_score(board, color, endgame_state)


# ── Feature helpers ───────────────────────────────────────────────────────────

def _closed_mills(board: BoardState, color: str) -> int:
    return sum(
        1 for mill in MILLS
        if all(board.positions[p] == color for p in mill)
    )


def _blocked_count(board: BoardState, color: str) -> int:
    """Count pieces of `color` with no legal adjacent empty square."""
    if get_game_phase(board, color) == "fly":
        return 0
    count = 0
    for pos in POSITIONS:
        if board.positions[pos] == color:
            if all(board.positions[n] != "" for n in ADJACENCY[pos]):
                count += 1
    return count


def _two_configs(board: BoardState, color: str) -> int:
    """Mills where color has exactly 2 pieces and 1 empty slot."""
    count = 0
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 2 and vals.count("") == 1:
            count += 1
    return count


def _double_mills(board: BoardState, color: str) -> int:
    """Pieces of `color` simultaneously part of 2+ closed mills."""
    count = 0
    for pos in POSITIONS:
        if board.positions[pos] == color:
            n = sum(
                1 for mill in MILLS
                if pos in mill and all(board.positions[p] == color for p in mill)
            )
            if n >= 2:
                count += 1
    return count


def _win_config(board: BoardState, opp: str) -> int:
    """1 if opponent is in fly phase — near-winning state."""
    return int(board.pieces_placed[opp] == 9 and board.pieces_on_board[opp] <= 3)


def _mobility(board: BoardState, color: str) -> int:
    """Count available destination squares for color (adjacency-based, phase-aware)."""
    phase = get_game_phase(board, color)
    if phase == "fly":
        empty = sum(1 for p in POSITIONS if board.positions[p] == "")
        return empty  # each piece can go anywhere empty; return empty count as proxy
    count = 0
    for pos in POSITIONS:
        if board.positions[pos] == color:
            count += sum(1 for n in ADJACENCY[pos] if board.positions[n] == "")
    return count


def _mill_threats(board: BoardState, color: str) -> int:
    """Count mills closeable in exactly one move (phase-aware reachability).

    Stricter than _two_configs: in move phase only counts mills where a friendly
    piece is actually adjacent to the empty closing square.  In place phase any
    two-config is closeable (can always place there).  In fly any empty square
    is reachable.  This makes the threat weight correctly reflect immediate danger
    rather than mere structural presence.
    """
    phase = get_game_phase(board, color)
    can_place = board.pieces_placed.get(color, 0) < 9
    count = 0
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 2 and vals.count("") == 1:
            empty = next(p for p in mill if board.positions[p] == "")
            if phase == "place":
                reachable = can_place
            elif phase == "fly":
                reachable = True
            else:
                reachable = any(board.positions[nb] == color for nb in ADJACENCY[empty])
            if reachable:
                count += 1
    return count


def _position_value(board: BoardState, color: str) -> int:
    """Positional score: cardinal (4-conn) = 5, cross (3-conn) = 3, corner (2-conn) = 2."""
    total = 0
    for pos in POSITIONS:
        if board.positions[pos] == color:
            if pos in _CARDINAL_NODES:
                total += 5
            elif pos in _CROSS_NODES_3:
                total += 3
            else:
                total += 2
    return total


def _mill_cycle_ready(board: BoardState, color: str) -> int:
    """
    Closed mills where at least one piece has a free adjacent square.
    Such a mill can be opened and re-closed every two moves to force a
    capture each cycle — the dominant winning pattern in the endgame.
    """
    count = 0
    for mill in MILLS:
        if all(board.positions[p] == color for p in mill):
            for pos in mill:
                if any(board.positions[nb] == "" for nb in ADJACENCY[pos]):
                    count += 1
                    break  # count each mill once even if multiple pieces can slide
    return count


def _fork_threats(board: BoardState, color: str) -> int:
    """
    Pieces simultaneously participating in 2+ open mills (two-configurations).
    A fork piece creates dual threats the opponent cannot both defend in one move.
    """
    open_mills = [
        mill for mill in MILLS
        if ([board.positions[p] for p in mill].count(color) == 2
            and [board.positions[p] for p in mill].count("") == 1)
    ]
    count = 0
    for pos in POSITIONS:
        if board.positions[pos] == color:
            if sum(1 for m in open_mills if pos in m) >= 2:
                count += 1
    return count


def _fly_asymmetry(board: BoardState, color: str) -> int:
    """
    +1 if color has entered fly phase (3 pieces, all 9 placed) and opponent has not.
    -1 if opponent has fly and color does not, BUT only when own pieces ≤ 5.

    The -1 penalty is capped at ≤ 5 own pieces because fly mobility is only a
    meaningful threat when piece counts are close.  At 6v3 or 7v3 the opponent's
    3 fly pieces are still losing badly, so penalising white for their fly is wrong.
    Rewards sacrificing down to 3 pieces to gain fly mobility before the opponent
    does, and penalises giving the opponent fly (e.g. capturing their 4th piece
    in a 4v4 position to leave them with 3 = fly advantage).
    """
    opp = "B" if color == "W" else "W"
    color_fly = (board.pieces_placed.get(color, 0) >= 9 and board.pieces_on_board[color] == 3)
    opp_fly   = (board.pieces_placed.get(opp,   0) >= 9 and board.pieces_on_board[opp]   == 3)
    if color_fly and not opp_fly:
        return 1
    if opp_fly and not color_fly and board.pieces_on_board[color] <= 5:
        return -1
    return 0


def _encirclement(board: BoardState, color: str) -> int:
    """
    Herding pressure: for each opponent piece, count how many of its adjacent
    squares are occupied by own pieces.  High score means opponent pieces are
    surrounded and have fewer escape routes.  Irrelevant in fly phase (pieces
    can jump to any empty square so adjacency confinement does not apply).
    """
    if get_game_phase(board, color) == "fly":
        return 0
    opp = "B" if color == "W" else "W"
    count = 0
    for pos in POSITIONS:
        if board.positions[pos] == opp:
            count += sum(1 for nb in ADJACENCY[pos] if board.positions[nb] == color)
    return count


def _squeeze_count(board: BoardState, color: str) -> int:
    """Count pieces of `color` with exactly 1 legal adjacent move remaining.

    A piece with 0 moves is already captured by _blocked_count.  This signal
    targets the intermediate state: pieces that still have one escape route but
    are one move away from total blockade.  High counts indicate the herding
    tactic is close to completion.  Irrelevant in fly phase (pieces jump freely).
    """
    if get_game_phase(board, color) == "fly":
        return 0
    count = 0
    for pos in POSITIONS:
        if board.positions[pos] == color:
            moves = sum(1 for n in ADJACENCY[pos] if board.positions[n] == "")
            if moves == 1:
                count += 1
    return count


def _open_mill_domination(board: BoardState, color: str) -> int:
    """Open-mill surplus in asymmetric endgame (own ≥ 6 pieces, opp ≤ 5).

    Counts own 2-configs in excess of what the opponent can physically block.
    With 3 open mills and 2 opponent pieces, one mill closes regardless of opp play.
    With 3 open mills and 3 opponent pieces, ALL opp pieces are pinned to blocking —
    the attacker's spare piece forces zugzwang (opp must move, uncovering a mill).
    Formula uses (opp_pieces - 1) so that equal-count pinning (3 mills, 3 opp) scores 1.
    """
    opp = "B" if color == "W" else "W"
    own_pieces = board.pieces_on_board[color]
    opp_pieces = board.pieces_on_board[opp]
    if own_pieces < 6 or opp_pieces > 5:
        return 0
    return max(0, _two_configs(board, color) - (opp_pieces - 1))


def _independent_mill_pairs(board: BoardState, color: str) -> int:
    """Count pairs of own 2-configs that share no own pieces.

    In 4v3 (own in move, opp in fly): two independent pairs ensure that even after
    the opponent captures one of our pieces, the remaining 3 still contain a 2-config.
    This is the 'insurance' structure the book prescribes for the 4-piece player.
    """
    two_cfg = [
        m for m in MILLS
        if ([board.positions[p] for p in m].count(color) == 2
            and [board.positions[p] for p in m].count("") == 1)
    ]
    count = 0
    for i in range(len(two_cfg)):
        for j in range(i + 1, len(two_cfg)):
            own_i = frozenset(p for p in two_cfg[i] if board.positions[p] == color)
            own_j = frozenset(p for p in two_cfg[j] if board.positions[p] == color)
            if not (own_i & own_j):
                count += 1
    return count


def _piece_separation(board: BoardState, color: str) -> int:
    """Return 1 if the opponent's pieces are graph-disconnected, else 0.

    In 3v4 (own fly, opp move): separated opponent groups can't defend each other.
    The fly-mobile attacker can threaten one group while the other is too far to help.
    Measured by BFS reachability among opponent pieces via board adjacency edges.
    Only meaningful when own is fly (3 pieces) and opp has exactly 4.
    """
    opp = "B" if color == "W" else "W"
    if board.pieces_on_board[opp] != 4:
        return 0
    opp_pos = [p for p in POSITIONS if board.positions[p] == opp]
    if not opp_pos:
        return 0
    opp_set = set(opp_pos)
    visited: set[str] = {opp_pos[0]}
    queue = [opp_pos[0]]
    while queue:
        curr = queue.pop()
        for nb in ADJACENCY[curr]:
            if nb in opp_set and nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return 1 if len(visited) < len(opp_pos) else 0


def _contested_mills(board: BoardState, color: str) -> int:
    """Count mills where own has 2 pieces and the opponent occupies the closing square.

    These are 'blocked' own mill attempts: color threatens a mill but the opponent
    is sitting in the only empty slot.  At the zugzwang position (6+v3 endgame),
    all 3 opponent pieces are in contested lines — any move uncovers a mill.
    Unlike _two_configs (requires an empty closing square), this detects the final
    zugzwang position where the closing squares are occupied by the opponent.
    """
    opp = "B" if color == "W" else "W"
    count = 0
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 2 and vals.count(opp) == 1:
            count += 1
    return count


def _free_piece_assembly(board: BoardState, color: str) -> int:
    """Count own 'free' pieces that neighbour a piece already in a 2-config.

    A free piece is one not currently in any closed mill or 2-config.  If it
    sits adjacent to a piece that IS in a 2-config, moving it one step could
    contribute to mill formation (it is 'assembling').  Counts such pieces
    so the evaluator can reward gradually gathering toward productive lines.
    Only meaningful in move phase (fly phase pieces jump anywhere, adjacency
    doesn't constrain assembly).
    """
    in_mill: set[str] = set()
    in_two: set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if all(v == color for v in vals):
            for p in mill:
                in_mill.add(p)
        elif vals.count(color) == 2 and vals.count("") == 1:
            for p in mill:
                if board.positions[p] == color:
                    in_two.add(p)
    count = 0
    for pos in POSITIONS:
        if board.positions[pos] == color and pos not in in_mill and pos not in in_two:
            if any(nb in in_two for nb in ADJACENCY[pos]):
                count += 1
    return count


def _assembly_reach_count(board: BoardState, color: str) -> int:
    """Count free pieces within 2 adjacency steps of a 2-config piece (step-2 only).

    Complements _free_piece_assembly (step-1) to create a pull gradient: pieces
    that are two hops away from the nearest 2-config are rewarded at a lower
    weight, encouraging them to drift toward productive formations.
    """
    in_mill: set[str] = set()
    in_two: set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if all(v == color for v in vals):
            for p in mill:
                in_mill.add(p)
        elif vals.count(color) == 2 and vals.count("") == 1:
            for p in mill:
                if board.positions[p] == color:
                    in_two.add(p)
    if not in_two:
        return 0
    # Squares directly adjacent to any in_two piece
    step1_squares: set[str] = set()
    for p in in_two:
        step1_squares.update(ADJACENCY[p])
    # Count free own pieces adjacent to step1 squares but not already step-1
    count = 0
    for pos in POSITIONS:
        if board.positions[pos] != color:
            continue
        if pos in in_mill or pos in in_two:
            continue
        if any(nb in in_two for nb in ADJACENCY[pos]):
            continue  # already counted as step-1 by _free_piece_assembly
        if any(nb in step1_squares for nb in ADJACENCY[pos]):
            count += 1
    return count


def _assembly_step3_count(board: BoardState, color: str) -> int:
    """Count free own pieces exactly 3 adjacency hops from any 2-config piece (step-3)."""
    in_mill: set[str] = set()
    in_two: set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if all(v == color for v in vals):
            for p in mill:
                in_mill.add(p)
        elif vals.count(color) == 2 and vals.count("") == 1:
            for p in mill:
                if board.positions[p] == color:
                    in_two.add(p)
    if not in_two:
        return 0
    step1: set[str] = set()
    for p in in_two:
        step1.update(ADJACENCY[p])
    step2: set[str] = set()
    for p in step1:
        step2.update(ADJACENCY[p])
    step3: set[str] = set()
    for p in step2:
        step3.update(ADJACENCY[p])
    count = 0
    for pos in POSITIONS:
        if board.positions[pos] != color:
            continue
        if pos in in_mill or pos in in_two:
            continue
        if any(nb in in_two   for nb in ADJACENCY[pos]):
            continue  # step-1
        if any(nb in step1    for nb in ADJACENCY[pos]):
            continue  # step-2
        if any(nb in step2    for nb in ADJACENCY[pos]):
            count += 1
    return count


def _assembly_step4_count(board: BoardState, color: str) -> int:
    """Count free own pieces exactly 4 adjacency hops from any 2-config piece (step-4)."""
    in_mill: set[str] = set()
    in_two: set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if all(v == color for v in vals):
            for p in mill:
                in_mill.add(p)
        elif vals.count(color) == 2 and vals.count("") == 1:
            for p in mill:
                if board.positions[p] == color:
                    in_two.add(p)
    if not in_two:
        return 0
    step1: set[str] = set()
    for p in in_two:
        step1.update(ADJACENCY[p])
    step2: set[str] = set()
    for p in step1:
        step2.update(ADJACENCY[p])
    step3: set[str] = set()
    for p in step2:
        step3.update(ADJACENCY[p])
    step4: set[str] = set()
    for p in step3:
        step4.update(ADJACENCY[p])
    count = 0
    for pos in POSITIONS:
        if board.positions[pos] != color:
            continue
        if pos in in_mill or pos in in_two:
            continue
        if any(nb in in_two  for nb in ADJACENCY[pos]):
            continue
        if any(nb in step1   for nb in ADJACENCY[pos]):
            continue
        if any(nb in step2   for nb in ADJACENCY[pos]):
            continue
        if any(nb in step3   for nb in ADJACENCY[pos]):
            count += 1
    return count


def _one_config_approach(board: BoardState, color: str) -> int:
    """Weighted count of free pieces approaching empty squares in 1-config mills.

    A 1-config mill has exactly one own piece and two empty squares.  Free own
    pieces adjacent to those empty squares score 2 (step-1); pieces two hops
    away score 1 (step-2).  Pieces already assigned to a closed mill, 2-config,
    or the 1-config itself are excluded — they are already positioned.

    This creates assembly pull toward nascent mills that the 2-config gradient
    misses entirely (which requires an existing 2-config to start from).
    Move phase only — fly phase ignores adjacency constraints.
    """
    in_mill: set[str] = set()
    in_two:  set[str] = set()
    in_one:  set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if all(v == color for v in vals):
            for p in mill: in_mill.add(p)
        elif vals.count(color) == 2 and vals.count("") == 1:
            for p in mill:
                if board.positions[p] == color: in_two.add(p)
        elif vals.count(color) == 1 and vals.count("") == 2:
            for p in mill:
                if board.positions[p] == color: in_one.add(p)

    # Collect all empty squares that belong to a 1-config mill
    target_empties: set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 1 and vals.count("") == 2:
            for p in mill:
                if board.positions[p] == "":
                    target_empties.add(p)

    if not target_empties:
        return 0

    step1_halo: set[str] = set()
    for sq in target_empties:
        step1_halo.update(ADJACENCY[sq])

    excluded = in_mill | in_two | in_one
    score = 0
    counted: set[str] = set()
    for pos in POSITIONS:
        if board.positions[pos] != color or pos in excluded or pos in counted:
            continue
        if any(nb in target_empties for nb in ADJACENCY[pos]):
            score += 2   # step-1: adjacent to an empty 1-config square
            counted.add(pos)
        elif any(nb in step1_halo for nb in ADJACENCY[pos]):
            score += 1   # step-2: two hops away
            counted.add(pos)
    return score


def _opponent_ring_concentration(board: BoardState, opp: str) -> list[int]:
    """Return [outer_count, middle_count, inner_count] of opponent pieces per ring."""
    return [sum(1 for p in ring if board.positions[p] == opp) for ring in _RINGS]


def _fork_in_n(board: BoardState, opp: str, n: int) -> set[str]:
    """Return the set of empty squares that, if occupied by `opp` within n moves,
    would create a fork (two simultaneous 2-configs).  n=2 is the main use case."""
    own = "B" if opp == "W" else "W"
    result: set[str] = set()
    if n < 1:
        return result
    for sq in POSITIONS:
        if board.positions[sq] != "":
            continue
        if board.positions[sq] == own:
            continue
        # Simulate opp placing at sq
        new_pos = dict(board.positions)
        new_pos[sq] = opp
        # Count resulting 2-configs for opp
        two_cfg = 0
        for mill in MILLS:
            vals = [new_pos[p] for p in mill]
            if vals.count(opp) == 2 and vals.count("") == 1:
                two_cfg += 1
        if two_cfg >= 2:
            result.add(sq)
            continue
        if n >= 2:
            # One more opp placement
            for sq2 in POSITIONS:
                if new_pos[sq2] != "" or sq2 == sq:
                    continue
                pos2 = dict(new_pos)
                pos2[sq2] = opp
                two_cfg2 = sum(
                    1 for mill in MILLS
                    if [pos2[p] for p in mill].count(opp) == 2
                    and [pos2[p] for p in mill].count("") == 1
                )
                if two_cfg2 >= 2:
                    result.add(sq)
                    break
    return result


def _is_mill_locked(board: BoardState, color: str, mill: tuple) -> bool:
    """True when every exit from every piece in mill is opponent-occupied.

    A locked mill has zero cycling value — the pivot cannot slide anywhere.
    Only meaningful in move phase (fly phase ignores adjacency).
    """
    opp = "B" if color == "W" else "W"
    mill_set = set(mill)
    for sq in mill:
        for nb in ADJACENCY[sq]:
            if nb not in mill_set and board.positions[nb] != opp:
                return False
    return True


def _is_anchored_blocker(board: BoardState, sq: str, color: str) -> bool:
    """True when the piece at sq is frozen — removing it benefits color.

    Case A: sq is the sole missing piece in a color 2-config (move it → color closes a mill).
    Case B: sq is adjacent to a color closed mill piece (move it → color gains a cycling exit).
    """
    for mill in MILLS:
        if sq in mill:
            if all(board.positions[p] == color for p in mill if p != sq):
                return True  # Case A
        else:
            if all(board.positions[p] == color for p in mill):
                if any(sq in ADJACENCY[p] for p in mill):
                    return True  # Case B
    return False


def _creates_redirected_pin(
    board: BoardState, color: str, from_sq: str, to_sq: str
) -> bool:
    """True when the move from_sq→to_sq causes an opponent piece to simultaneously
    block two own 2-configs (a double-pin).

    Detection: after the hypothetical move, find opponent pieces that are the sole
    blocker of two distinct own 2-configs.
    """
    opp = "B" if color == "W" else "W"
    new_pos = dict(board.positions)
    new_pos[from_sq] = ""
    new_pos[to_sq] = color

    for blocker_sq in POSITIONS:
        if new_pos[blocker_sq] != opp:
            continue
        pinned_count = 0
        for mill in MILLS:
            if blocker_sq not in mill:
                continue
            # own has 2 pieces, blocker is the 3rd
            vals = [new_pos[p] for p in mill]
            if vals.count(color) == 2 and new_pos[blocker_sq] == opp:
                # Count own pieces in this mill
                own_in_mill = [p for p in mill if new_pos[p] == color]
                empty_in_mill = [p for p in mill if new_pos[p] == ""]
                if len(own_in_mill) == 2 and len(empty_in_mill) == 0:
                    # blocker IS in a fully-blocked mill (all 3 squares filled)
                    pinned_count += 1
        if pinned_count >= 2:
            return True
    return False


def _mill_cycling_freedom(board: BoardState, color: str, mill: tuple) -> int:
    """Count empty non-mill exit squares reachable from any piece in mill.

    Used to judge how freely a closed mill can cycle (open/close for captures).
    Higher = more dangerous mill to surrender to the opponent.
    """
    mill_set = set(mill)
    exits: set[str] = set()
    for sq in mill:
        for nb in ADJACENCY[sq]:
            if nb not in mill_set and board.positions[nb] == "":
                exits.add(nb)
    return len(exits)


def _opponent_fork_arms(board: BoardState, color: str) -> list[tuple]:
    """Return [(mill_tuple, closing_square), ...] for each fork arm the opponent threatens.

    A fork arm is a mill where the opponent (opp) has 2 pieces and 1 empty square.
    When 2+ arms exist simultaneously the opponent has a fork.
    """
    opp = "B" if color == "W" else "W"
    arms = []
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(opp) == 2 and vals.count("") == 1:
            closing = next(p for p in mill if board.positions[p] == "")
            arms.append((mill, closing))
    return arms


def _own_piece_adj_to_closing(board: BoardState, color: str, closing_sq: str) -> bool:
    """True if own piece occupies any square adjacent to closing_sq.

    Used for the cardinal exception in B-8: if own piece already presses the
    closing square of the cardinal mill, that mill's cycling is already constrained,
    so blocking it is less urgent.
    """
    return any(board.positions[nb] == color for nb in ADJACENCY[closing_sq])


def _fly_sacrifice_quality(board: BoardState, color: str) -> int:
    """In a 6v4 position, estimate how strong a 3-piece flying endgame would be.

    Returns 2 if own already has a closed mill (excellent fly nucleus), 1 if own
    has at least one 2-config (good fly nucleus), 0 if the pieces are scattered.
    Only meaningful when called with own_pieces==6 and opp_pieces==4.
    """
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 3:
            return 2
        if vals.count(color) == 2 and vals.count("") == 1:
            return 1  # keep scanning for a closed mill
    return 0


def _interpose_count(board: BoardState, color: str) -> int:
    """Count own pieces sandwiched between two opponent pieces in a mill (own at middle index)."""
    opp = "B" if color == "W" else "W"
    count = 0
    for a, b, c in MILLS:
        if (board.positions[b] == color
                and board.positions[a] == opp
                and board.positions[c] == opp):
            count += 1
    return count


def _perp_block_count(board: BoardState, color: str) -> int:
    """Count own pieces that are simultaneously in an own 2-config and blocking an opponent 2-config.

    Example: white e5 in e3-e4-e5 (own 2-config) also blocks c5-d5-e5 (opp 2-config) at e5.
    """
    opp = "B" if color == "W" else "W"
    own_two_pieces: set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 2 and vals.count("") == 1:
            for p in mill:
                if board.positions[p] == color:
                    own_two_pieces.add(p)
    if not own_two_pieces:
        return 0
    counted: set[str] = set()
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(opp) == 2 and vals.count(color) == 1:
            for p in mill:
                if board.positions[p] == color and p in own_two_pieces and p not in counted:
                    counted.add(p)
    return len(counted)


def _double_mill_convergence(board: BoardState, opp: str) -> int:
    """Count opponent fork precursor pairs: two 2-configs that share either a
    common empty closing square (diamond) or a common opp pivot piece.

    A shared closing square means placing there closes BOTH mills simultaneously.
    A shared pivot means one piece participates in two different 2-configs and
    moving it can threaten either line.  Both patterns are one move away from
    an unblockable fork and should be disrupted before they mature.
    """
    closing: dict[str, list[int]] = {}  # sq -> [mill_index, ...]
    pivots:  dict[str, list[int]] = {}  # opp_piece -> [mill_index, ...]

    for i, mill in enumerate(MILLS):
        vals = [board.positions[p] for p in mill]
        if vals.count(opp) == 2 and vals.count("") == 1:
            emp = next(p for p in mill if board.positions[p] == "")
            closing.setdefault(emp, []).append(i)
            for p in mill:
                if board.positions[p] == opp:
                    pivots.setdefault(p, []).append(i)

    count = 0
    # Shared closing square (classic diamond)
    shared_pairs: set[frozenset[int]] = set()
    for mills_for_sq in closing.values():
        for a in range(len(mills_for_sq)):
            for b in range(a + 1, len(mills_for_sq)):
                pair = frozenset((mills_for_sq[a], mills_for_sq[b]))
                if pair not in shared_pairs:
                    shared_pairs.add(pair)
                    count += 1

    # Shared pivot piece (not already counted as shared closing square)
    for mills_for_piece in pivots.values():
        for a in range(len(mills_for_piece)):
            for b in range(a + 1, len(mills_for_piece)):
                pair = frozenset((mills_for_piece[a], mills_for_piece[b]))
                if pair not in shared_pairs:
                    shared_pairs.add(pair)
                    count += 1

    return count


def _cross_feed_mobility_pairs(board: BoardState, color: str) -> int:
    """Count own 2-config pairs linked by cross-adjacency (B-16 general case).

    Two independent 2-config groups form a cross-feeding pair when, after any
    opponent capture from either group, the surviving piece has the mobility
    to reach the OTHER group's closing square in one move (move phase) or any
    move (fly phase).  This gives the position resilience: whichever group the
    opponent attacks, the survivor migrates and closes the untouched mill.

    Only counts pairs whose closing squares differ AND own pieces don't overlap
    (shared-closing-square and shared-pivot cases are already handled by
    _double_mill_convergence applied to `color`).
    """
    configs: list[tuple[tuple[str, ...], str, frozenset[str]]] = []
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 2 and vals.count("") == 1:
            closing = next(p for p in mill if board.positions[p] == "")
            pieces  = frozenset(p for p in mill if board.positions[p] == color)
            configs.append((mill, closing, pieces))

    if len(configs) < 2:
        return 0

    phase = get_game_phase(board, color)
    count = 0

    for i in range(len(configs)):
        _, close_i, pieces_i = configs[i]
        for j in range(i + 1, len(configs)):
            _, close_j, pieces_j = configs[j]

            # Skip pairs already covered by _double_mill_convergence:
            # shared closing square or shared own piece (pivot).
            if close_i == close_j or (pieces_i & pieces_j):
                continue

            if phase == "fly":
                # Fly: every square reachable → all independent non-shared pairs qualify.
                count += 1
            else:
                # Move: check if any piece in group_i is adjacent to close_j,
                # or any piece in group_j is adjacent to close_i.
                if (any(close_j in ADJACENCY[p] for p in pieces_i)
                        or any(close_i in ADJACENCY[p] for p in pieces_j)):
                    count += 1

    return count


def _late_game_danger(board: BoardState, color: str) -> int:
    """Structural danger/dominance bonus in asymmetric endgame positions.

    Three patterns:
    1. Defensive penalty: ≤5 own pieces vs opponent with ≥2 closed mills.
       Fly mobility inflates the losing side's score; this corrects for that.
    2. Cycling mill dominance: opponent has ≥2 cycling-ready mills (can force a
       capture every 2 turns).  This is nearly always decisive.
    3. Attack reward: own pieces ≥ 6, opp ≤ 4, with 2+ open mills — the dominant
       side should aggressively accumulate open mills toward the three-mill trap.
    """
    opp = "B" if color == "W" else "W"
    our_pieces = board.pieces_on_board[color]
    opp_pieces = board.pieces_on_board[opp]
    opp_mills  = _closed_mills(board, opp)
    penalty = 0

    if our_pieces <= 5 and opp_mills >= 2:
        severity = opp_mills * 200 + max(0, opp_pieces - our_pieces) * 60
        penalty -= severity

    opp_cycle = _mill_cycle_ready(board, opp)
    if opp_cycle >= 2:
        penalty -= opp_cycle * 130

    # Attack reward: reward positions where the dominant side has spread open mills
    # beyond the opponent's blocking capacity (the three-mill-trap pattern).
    own_open = _two_configs(board, color)
    if our_pieces >= 6 and opp_pieces <= 4 and own_open >= 2:
        uncoverable = max(0, own_open - opp_pieces)
        penalty += own_open * 80 + uncoverable * 180

    # Zugzwang detection: own ≥ 6 pieces, opp has exactly 3 with all pieces
    # occupying own contested mills (closing squares blocked).  _two_configs
    # returns 0 here because closing squares are full, so this is the only
    # signal that captures the final zugzwang configuration.
    if our_pieces >= 6 and opp_pieces == 3:
        contested = _contested_mills(board, color)
        if contested >= 2:
            penalty += contested * 120 + max(0, contested - 2) * 230

    return penalty


# ── Public tactical detectors (Stage 5.12) ───────────────────────────────────

def detect_double_mills(board: BoardState, color: str) -> list[str]:
    """Return positions that are pivot pieces belonging to 2+ closed mills."""
    return [
        pos for pos in POSITIONS
        if board.positions[pos] == color
        and sum(
            1 for mill in MILLS
            if pos in mill and all(board.positions[p] == color for p in mill)
        ) >= 2
    ]


def detect_feeder_mills(board: BoardState, color: str) -> list[tuple[str, ...]]:
    """Return closed mills that have at least one same-color adjacent feeder piece.

    A feeder piece is a same-color piece adjacent to the mill but not part of it.
    Its presence means the mill can be cycled: slide a mill piece out, slide the
    feeder in to re-form the mill, capturing again next turn.
    """
    result = []
    for mill in MILLS:
        if all(board.positions[p] == color for p in mill):
            mill_set = set(mill)
            if any(
                board.positions[nb] == color
                for pos in mill
                for nb in ADJACENCY[pos]
                if nb not in mill_set
            ):
                result.append(tuple(mill))
    return result


def detect_diamonds(board: BoardState, color: str) -> list[str]:
    """Return empty squares that are the closing square for 2+ own two-configs.

    Placing on any returned square would close two mills simultaneously (a fork),
    which the opponent cannot fully defend in a single response.
    """
    closing: dict[str, int] = {}
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 2 and vals.count("") == 1:
            empty = next(p for p in mill if board.positions[p] == "")
            closing[empty] = closing.get(empty, 0) + 1
    return [pos for pos, count in closing.items() if count >= 2]


def opponent_mills_in_n_moves(board: BoardState, color: str, n: int = 2) -> int:
    """Count mills `color` can form within `n` moves (1 or 2).

    n=1: mills closeable this turn (= _closeable_mills).
    n=2: adds mills reachable in two moves (one own piece + two reachable empties).
    """
    if n < 1:
        return 0
    count = _closeable_mills(board, color)
    if n >= 2:
        phase = get_game_phase(board, color)
        for mill in MILLS:
            vals = [board.positions[p] for p in mill]
            if vals.count(color) == 1 and vals.count("") == 2:
                if phase in ("place", "fly"):
                    count += 1
                else:
                    empties = [p for p in mill if board.positions[p] == ""]
                    if any(
                        any(board.positions[nb] == color for nb in ADJACENCY[e])
                        for e in empties
                    ):
                        count += 1
    return count


# ── Tactical urgency (internal helpers) ──────────────────────────────────────

def _closeable_mills(board: BoardState, color: str) -> int:
    """Count 2-config mills that color can close in exactly one move."""
    phase = get_game_phase(board, color)
    can_place = board.pieces_placed.get(color, 0) < 9
    count = 0
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 2 and vals.count("") == 1:
            empty = next(p for p in mill if board.positions[p] == "")
            if phase == "place":
                reachable = can_place
            elif phase == "fly":
                reachable = True
            else:
                reachable = any(board.positions[nb] == color for nb in ADJACENCY[empty])
            if reachable:
                count += 1
    return count


def _cycling_mill_setup(board: BoardState, color: str) -> int:
    """Count cycling opportunities — two types:

    Type A: Two open 2-configs whose empty closing squares are adjacent.
            A single pivot piece can shuttle E1→E2→E1 to force a capture
            every two turns.

    Type B: A cycle-ready closed mill (at least one piece has an adjacent
            free exit square P) paired with an open 2-config whose closing
            square equals P.  This captures the "one mill closed, one ready
            to close" state — e.g. White has c5-d5-e5 closed with d5 free
            to exit to d6, and d6 is the closing square of b6-d6-f6.

    The tactical bonus is only given when this count INCREASES (delta > 0),
    so already-established cycling positions do not score again.  In fly
    phase every empty square is reachable so all 2-config pairs qualify.
    """
    phase = get_game_phase(board, color)

    # Collect empty closing squares of all own 2-configs
    open_closings: list[str] = []
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 2 and vals.count("") == 1:
            open_closings.append(next(p for p in mill if board.positions[p] == ""))

    count = 0
    n = len(open_closings)

    # Type A: adjacent closing-square pairs
    for i in range(n):
        for j in range(i + 1, n):
            if phase == "fly" or open_closings[j] in ADJACENCY[open_closings[i]]:
                count += 1

    # Type B: cycle-ready closed mill whose free exit is a 2-config closing square
    if phase != "fly":
        open_closing_set = set(open_closings)
        for mill in MILLS:
            if not all(board.positions[p] == color for p in mill):
                continue
            mill_set = set(mill)
            for p in mill:
                for nb in ADJACENCY[p]:
                    if nb not in mill_set and nb in open_closing_set:
                        count += 1
                        break  # count each closed mill at most once
                else:
                    continue
                break

    return count


def _feeder_diamond(board: BoardState, color: str) -> int:
    """Count empty squares shared by 2+ own 2-configs (diamond / fork structures).

    A diamond is 4 own pieces all adjacent to one key empty square, forming two
    simultaneous mill threats.  Example: a4-c4-b2-b6 all border b4; either
    a4-b4-c4 or b2-b4-b6 closes when b4 is filled.  If one anchor is captured,
    a remaining piece slides to b4 to close the other mill.
    """
    closing: dict[str, int] = {}
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 2 and vals.count("") == 1:
            empty = next(p for p in mill if board.positions[p] == "")
            closing[empty] = closing.get(empty, 0) + 1
    return sum(1 for c in closing.values() if c >= 2)


def _mill_wrapping_pressure(board: BoardState, color: str) -> int:
    """Own pieces occupying exit squares of every opponent closed mill.

    An exit square is any square adjacent to a mill piece but not in the mill
    itself.  High coverage means the opponent's mill is surrounded and cannot
    be easily exploited by cycling (the pivot has nowhere to slide to).
    Counted per mill so a piece bordering two mills contributes twice.

    Returns 0 when the opponent is in fly phase — fly pieces can jump to any
    empty square, so blocking adjacency exits does not prevent mill cycling.
    """
    opp = "B" if color == "W" else "W"
    if get_game_phase(board, opp) == "fly":
        return 0
    total = 0
    for mill in MILLS:
        if all(board.positions[p] == opp for p in mill):
            mill_set = set(mill)
            covered: set[str] = set()
            for pos in mill:
                for nb in ADJACENCY[pos]:
                    if nb not in mill_set and board.positions[nb] == color:
                        covered.add(nb)
            total += len(covered)
    return total


def _cross_node_count(board: BoardState, color: str) -> int:
    """Count pieces of `color` sitting on cross/cardinal nodes."""
    return sum(1 for p in _CROSS_NODES if board.positions[p] == color)


def _convergence_cluster_count(board: BoardState, opp: str) -> int:
    """Count mills where the opponent has 3 pieces able to converge within 2 adjacency
    moves along unblocked paths (no own pieces occupy target squares or intermediates).

    Applies to any mill on the board — outer, middle, inner ring, or cross-ring.
    Uses bipartite matching to verify 3 distinct opponent pieces can each claim a
    distinct mill square within 2 moves.
    """
    own = "B" if opp == "W" else "W"
    opp_pieces = [p for p in POSITIONS if board.positions[p] == opp]
    if len(opp_pieces) < 3:
        return 0

    def _can_reach(piece: str, target: str) -> bool:
        if piece == target:
            return True  # already there (0 moves)
        if board.positions[target] == own:
            return False  # own piece occupies target
        if board.positions[target] == opp:
            return False  # another opp piece already occupies target
        # target is empty — check 1-step and 2-step
        if target in ADJACENCY[piece]:
            return True
        for mid in ADJACENCY[piece]:
            if board.positions[mid] == "" and target in ADJACENCY[mid]:
                return True
        return False

    count = 0
    for mill in MILLS:
        mill_sqs = list(mill)
        # Skip if any mill square is owned — own piece blocks convergence
        if any(board.positions[sq] == own for sq in mill_sqs):
            continue
        # Build per-square candidate lists
        candidates: list[list[str]] = [
            [p for p in opp_pieces if _can_reach(p, sq)]
            for sq in mill_sqs
        ]
        # Bipartite matching: 3 distinct pieces → 3 distinct squares
        def _match(idx: int, used: frozenset) -> bool:
            if idx == len(mill_sqs):
                return True
            for piece in candidates[idx]:
                if piece not in used:
                    if _match(idx + 1, used | frozenset([piece])):
                        return True
            return False

        if _match(0, frozenset()):
            count += 1

    return count


def _placement_chain_scan(board: BoardState, color: str) -> int:
    """Busy-opponent placement initiative scan (placement phase only).

    Scans up to 4 half-moves ahead to find forcing sequences where every AI
    placement compels an opponent response, ideally ending with a mill closure.
    Also rewards "two-for-one" placements that block an opponent threat while
    simultaneously creating a new AI 2-config — maintaining initiative even
    while defending.

    Uses a simplified board model (no mill-capture handling) so it stays fast.
    Called from tactical_move_bonus on every candidate placement.

    Returns 0–4:
    4 — chain found where the last piece closes a mill
    3 — fork reachable within chain (2 simultaneous threats opp can't both block)
    2 — sustained initiative: forcing pressure persists after one opp response
    1 — single immediate forcing threat (opp must respond)
    0 — no forcing initiative from this position
    """
    opp = "B" if color == "W" else "W"
    our_rem = 9 - board.pieces_placed.get(color, 0)
    opp_rem = 9 - board.pieces_placed.get(opp, 0)
    if our_rem <= 0:
        return 0

    def _threats(b: BoardState, c: str) -> list[str]:
        seen: list[str] = []
        for mill in MILLS:
            vals = [b.positions[p] for p in mill]
            if vals.count(c) == 2 and vals.count("") == 1:
                e = next(p for p in mill if b.positions[p] == "")
                if e not in seen:
                    seen.append(e)
        return seen

    def _place(b: BoardState, c: str, pos: str) -> BoardState:
        o = "B" if c == "W" else "W"
        new_pos = dict(b.positions)
        new_pos[pos] = c
        return BoardState(
            positions=new_pos,
            turn=o,
            pieces_on_board={**b.pieces_on_board, c: b.pieces_on_board[c] + 1},
            pieces_placed={**b.pieces_placed, c: b.pieces_placed.get(c, 0) + 1},
            pieces_captured=dict(b.pieces_captured),
        )

    def _closes_mill(b: BoardState, c: str, pos: str) -> bool:
        for mill in MILLS:
            if pos in mill and b.positions[pos] == "" and all(
                b.positions[p] == c for p in mill if p != pos
            ):
                return True
        return False

    def _productive(b: BoardState, c: str) -> list[str]:
        """Placements that close a mill, create a 2-config, or block+create (two-for-one)."""
        o = "B" if c == "W" else "W"
        result: list[str] = []
        for pos in POSITIONS:
            if b.positions[pos] != "":
                continue
            useful = False
            blocks_opp_mill = False
            for mill in MILLS:
                if pos not in mill:
                    continue
                vals = [b.positions[p] for p in mill]
                own = vals.count(c)
                opp_cnt = vals.count(o)
                emp = vals.count("")
                if own == 2 and opp_cnt == 0:             # closes or 2-config
                    useful = True
                    break
                if own == 1 and opp_cnt == 0 and emp == 2: # creates 2-config
                    useful = True
                    break
                if opp_cnt == 2 and emp == 1:              # blocks opp
                    blocks_opp_mill = True
            # Two-for-one: blocking an opponent threat while creating own 2-config
            if not useful and blocks_opp_mill:
                for mill in MILLS:
                    if pos not in mill:
                        continue
                    vals = [b.positions[p] for p in mill]
                    if vals.count(c) == 1 and vals.count(o) == 0 and vals.count("") == 2:
                        useful = True
                        break
            if useful:
                result.append(pos)
        return result

    # ── Chain scan ────────────────────────────────────────────────────────────

    my_threats = _threats(board, color)

    # Seed path: current placement only creates a 1-config (no immediate 2-config).
    # Try one "seed" placement that would create a 2-config, then continue the chain.
    # Capped at level 3 because the seed step is not forced.
    if not my_threats:
        if our_rem < 2:
            return 0
        seeds = _productive(board, color)
        if not seeds:
            return 0
        seed_best = 0
        for seed in seeds[:4]:
            if board.positions.get(seed) != "":
                continue
            b_seed = _place(board, color, seed)
            t_seed = _threats(b_seed, color)
            if not t_seed:
                continue
            seed_best = max(seed_best, 1)
            if opp_rem <= 0:
                seed_best = max(seed_best, 3 if len(t_seed) >= 2 else 2)
                continue
            for threat_sq in t_seed[:3]:
                if b_seed.positions.get(threat_sq) != "":
                    continue
                b_blocked = _place(b_seed, opp, threat_sq)
                for close_sq in POSITIONS:
                    if (b_blocked.positions.get(close_sq) == ""
                            and _closes_mill(b_blocked, color, close_sq)):
                        seed_best = max(seed_best, 3)
                        break
                t2 = _threats(b_blocked, color)
                if len(t2) >= 2:
                    seed_best = max(seed_best, 3)
                elif t2:
                    seed_best = max(seed_best, 2)
            if seed_best >= 3:
                break
        return seed_best

    best = 1

    # Opponent cannot respond (finished placing) — threats are permanent
    if opp_rem <= 0:
        return 3 if len(my_threats) >= 2 else 2

    for threat in my_threats[:3]:
        if board.positions.get(threat, "") != "":
            continue
        b1 = _place(board, opp, threat)

        # After opp blocks, do we still have threats?
        t1 = _threats(b1, color)

        if len(t1) >= 2:
            best = max(best, 3)
        elif t1:
            best = max(best, 2)

        if our_rem < 2:
            continue

        # AI places a second piece — look for fork or mill closure
        for ai_sq in _productive(b1, color)[:6]:
            if b1.positions.get(ai_sq, "") != "":
                continue

            # Does this placement close a mill immediately?
            if _closes_mill(b1, color, ai_sq):
                best = max(best, 4)
                break

            b2 = _place(b1, color, ai_sq)
            t2 = _threats(b2, color)

            if len(t2) >= 2:
                best = max(best, 3)
            elif t2 and opp_rem >= 2 and our_rem >= 2:
                # Continue one more round: opp blocks again
                for block2 in t2[:2]:
                    if b2.positions.get(block2, "") != "":
                        continue
                    b3 = _place(b2, opp, block2)
                    for ai_sq3 in _productive(b3, color)[:4]:
                        if b3.positions.get(ai_sq3, "") != "":
                            continue
                        if _closes_mill(b3, color, ai_sq3):
                            best = max(best, 4)
                            break
                        b4 = _place(b3, color, ai_sq3)
                        if len(_threats(b4, color)) >= 2:
                            best = max(best, 3)
                    if best >= 4:
                        break

            if best >= 4:
                break

        if best >= 4:
            break

    return best


def tactical_move_bonus(
    before: BoardState,
    after: BoardState,
    color: str,
    weights: HeuristicWeights | None = None,
    opp_last_weak: bool = False,
) -> int:
    """Delta-based tactical bonus added directly to the root-move score.

    Applied in _score_all / _root_search AFTER the negamax score is computed,
    so it does not invert through negamax negation.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    opp = "B" if color == "W" else "W"

    # Mills closed this move
    mills_delta = max(0, _closed_mills(after, color) - _closed_mills(before, color))

    # Cycling mill setup gained (own) or disrupted for opponent this move.
    # Capped at 1: a move either creates/destroys a cycling opportunity or it doesn't.
    # Without the cap, moves that create many pairs at once would score 3-4× too high
    # and dominate the close_mill bonus, causing the AI to build structure over winning.
    cycling_gain   = min(1, max(0, _cycling_mill_setup(after, color) - _cycling_mill_setup(before, color)))
    opp_cycle_lost = min(1, max(0, _cycling_mill_setup(before, opp)  - _cycling_mill_setup(after,  opp)))

    # Opponent immediate closeable threats neutralised
    blocked = max(0, _closeable_mills(before, opp) - _closeable_mills(after, opp))

    # Opponent 2-configs dismantled (broader than closeable — any 2-piece setup)
    own_two_before = _two_configs(before, color)
    own_two_after  = _two_configs(after,  color)
    two_cfg_broken = max(0, _two_configs(before, opp) - _two_configs(after, opp))

    # Diamond / fork structures gained this move (capped at 1 for same reason as cycling).
    diamond_gain = min(1, max(0, _feeder_diamond(after, color) - _feeder_diamond(before, color)))

    # Mill wrapping pressure gained (own pieces covering opponent mill exit squares)
    wrap_gain = max(0, _mill_wrapping_pressure(after, color) - _mill_wrapping_pressure(before, color))

    # Cardinal / cross-node control gained or opponent evicted
    our_cross_gained = max(0, _cross_node_count(after, color) - _cross_node_count(before, color))
    opp_cross_lost   = max(0, _cross_node_count(before, opp)  - _cross_node_count(after,  opp))

    # Early-game scatter: bonus for placing non-adjacent in first 6 placements
    scatter = 0
    if (get_game_phase(before, color) == "place"
            and before.pieces_placed.get(color, 0) < 6):
        for pos in POSITIONS:
            if after.positions[pos] == color and before.positions[pos] != color:
                if not any(before.positions[nb] == color for nb in ADJACENCY[pos]):
                    scatter = weights.scatter_placement
                break

    # New two-configs gained this move (setup for future mill opportunities).
    # In placement phase: reward building toward mills during piece deployment.
    # In movement phase: also reward — gaining a new 2-config in move phase is
    # equally important since mills can no longer be created by placement alone.
    setup_mill_bonus = 0
    before_phase = get_game_phase(before, color)
    if before_phase == "place" and before.pieces_placed.get(color, 0) < 9:
        two_cfg_gained = max(0, own_two_after - own_two_before)
        setup_mill_bonus = weights.setup_mill * two_cfg_gained
    elif before_phase == "move":
        two_cfg_gained = max(0, own_two_after - own_two_before)
        # Slightly higher weight in move phase — gaining a 2-config is now a
        # concrete mill threat, not just a structural seed.
        setup_mill_bonus = int(weights.setup_mill * 1.3) * two_cfg_gained

    # Mill-opening bonus: reward sliding out of a closed mill when the position
    # still has a cycling-ready mill (i.e. the move enables a future recapture).
    # This encourages deliberate opening/closing cycles rather than passivity.
    mill_opened = max(0, _closed_mills(before, color) - _closed_mills(after, color))
    mill_open_bonus = 0
    if mill_opened > 0 and _mill_cycle_ready(after, color) > 0:
        mill_open_bonus = weights.mill_opening * mill_opened

    capture_this_move = after.pieces_on_board[opp] < before.pieces_on_board[opp]

    # Capture quality bonuses: reward capturing structurally important opponent pieces.
    capture_feeder_bonus = 0
    capture_diamond_bonus = 0
    if capture_this_move:
        captured_pos = next(
            (pos for pos in POSITIONS
             if before.positions[pos] == opp and after.positions[pos] != opp),
            None,
        )
        if captured_pos is not None:
            # Feeder disruption: captured piece was adjacent to (but not in) a closed opponent mill.
            # Removing it breaks the cycling potential of that mill.
            for mill in MILLS:
                mill_set = set(mill)
                if all(before.positions[p] == opp for p in mill):
                    adj_non_mill = {
                        nb for mp in mill for nb in ADJACENCY[mp] if nb not in mill_set
                    }
                    if captured_pos in adj_non_mill and before.positions[captured_pos] == opp:
                        capture_feeder_bonus = weights.capture_disrupt_feeder
                        break

            # Diamond disruption: captured piece was part of an opponent 2-config
            # that contributed to a fork (two 2-configs sharing a closing square).
            if not capture_feeder_bonus:
                opp_forks = detect_diamonds(before, opp)
                for fork_sq in opp_forks:
                    for mill in MILLS:
                        if fork_sq in mill:
                            mill_pieces = [p for p in mill if p != fork_sq]
                            if (all(before.positions[p] == opp for p in mill_pieces)
                                    and captured_pos in mill_pieces):
                                capture_diamond_bonus = weights.capture_disrupt_diamond
                                break
                    if capture_diamond_bonus:
                        break

    # Safe-capture bonus: reward captures that remove ALL opponent closeable mills when opp
    # had at least one before — the capture actively neutralised all remaining mill threats.
    safe_capture_bonus = 0
    if capture_this_move and _closeable_mills(before, opp) > 0 and _closeable_mills(after, opp) == 0:
        safe_capture_bonus = 180

    # Outer-ring mill penalty during early placement (pieces 1–6).
    # Each outer-ring side mill (a7-d7-g7, g7-g4-g1, g1-d1-a1, a1-a4-a7)
    # contains two corner squares with only 2 connections each.  Completing
    # one in the first six placements locks two pieces into the lowest-mobility
    # positions on the board, hurting the movement phase.
    # Skipped when the mill comes with an immediate capture — material gain
    # justifies the mobility cost in that case.
    # Does NOT fire at pieces 7-9 (late_mill_bonus below handles those).
    outer_mill_penalty = 0
    if (mills_delta > 0 and not capture_this_move
            and get_game_phase(before, color) == "place"
            and before.pieces_placed.get(color, 0) < 6):
        for mill in MILLS:
            if (all(after.positions[p] == color for p in mill)
                    and not all(before.positions[p] == color for p in mill)):
                if frozenset(mill) in _OUTER_MILLS:
                    outer_mill_penalty += 1
        outer_mill_penalty *= int(weights.close_mill * 0.65)

    # Late-placement mill urgency (pieces 7-9, i.e. pieces_placed >= 6).
    # Closing a mill on the OUTER or MIDDLE ring in this window is still
    # valuable — any mill at this stage is better than none.
    # Inner-ring mills are excluded: they confine pieces to the smallest square
    # and reduce long-term mobility more than they gain from the mill itself.
    late_mill_bonus = 0
    if (mills_delta > 0 and get_game_phase(before, color) == "place"
            and before.pieces_placed.get(color, 0) >= 6):
        for mill in MILLS:
            if (all(after.positions[p] == color for p in mill)
                    and not all(before.positions[p] == color for p in mill)):
                if frozenset(mill) not in _INNER_MILLS:
                    late_mill_bonus += 1
        late_mill_bonus *= int(weights.close_mill * 0.6)  # 60% extra urgency

    # Three-mill-trap builder: bonus for gaining the 3rd+ open mill when already dominant.
    # At this point opp can't cover all mills and must leave one open; actively accelerates
    # the zugzwang setup the book describes for 7v4 and 6v4 endgames.
    trap_build_bonus = 0
    if (own_two_before >= 2
            and own_two_after > own_two_before
            and after.pieces_on_board[color] >= 6
            and after.pieces_on_board[opp] <= 5):
        trap_build_bonus = weights.mill_trap_build

    # Fly-fork creation bonus: in fly phase, going from <2 own 2-configs to ≥2 in
    # one move creates a fork that the opponent can block at most one of.
    # Opening a closed mill to set up this fork is the primary 3v3 winning strategy.
    fly_fork_bonus = 0
    after_phase = get_game_phase(after, color)
    if after_phase == "fly" and own_two_after >= 2 and own_two_before < 2:
        fly_fork_bonus = 750

    # Fly-free-close bonus: in fly phase, reward closing a mill using a piece that
    # was NOT already in that mill before the move — the "free piece" jumped in from
    # elsewhere, keeping the other two mill pieces intact as an ongoing threat.
    fly_free_close_bonus = 0
    if after_phase == "fly" and mills_delta > 0:
        moved_from = next(
            (p for p in POSITIONS if before.positions[p] == color and after.positions[p] != color),
            None,
        )
        if moved_from is not None:
            for mill in MILLS:
                if (all(after.positions[p] == color for p in mill)
                        and not all(before.positions[p] == color for p in mill)
                        and moved_from not in mill):
                    fly_free_close_bonus = 200
                    break

    # Ring-crowding penalty: fires when own placement is the 6th+ piece on one ring
    # (outer/middle/inner).  Concentrating ≥4 pieces on a single ring reduces
    # movement-phase mobility severely — those pieces share few exit squares and can
    # be blocked by opponent cardinal-point control.  Max one penalty per move.
    ring_crowd_penalty = 0
    if get_game_phase(before, color) == "place":
        new_sq = next(
            (p for p in POSITIONS
             if after.positions[p] == color and before.positions[p] != color),
            None,
        )
        if new_sq is not None:
            for ring in _RINGS:
                if new_sq in ring:
                    own_on_ring = sum(1 for p in ring if before.positions[p] == color)
                    if own_on_ring >= 5:  # placing the 6th+ piece on this ring
                        ring_crowd_penalty = weights.ring_crowding_penalty
                    break

    # Placement busy-opponent chain scan.
    # Rewards forcing sequences where every placement compels an opp response
    # while building toward a mill. Skipped if the current move already closes a
    # mill (in that case the close_mill bonus is dominant and chain planning is moot).
    # Level 1 (single threat) gets modest credit; levels 2–4 scale up with chain
    # length and quality. Amplified when opponent's last move was structurally weak.
    busy_chain_bonus = 0
    _before_phase = get_game_phase(before, color)
    _had_mill_available = _closeable_mills(before, color) > 0
    if _before_phase == "place" and mills_delta == 0:
        if not _had_mill_available:
            # Normal path: no mill available, run full scan.
            chain = _placement_chain_scan(after, color)
        else:
            # Mill was available but this move skips it.
            # Only defer to a chain if we are deep in placement (pieces 7-9)
            # and the chain qualifies as level-4. Earlier in the game an
            # immediate mill + capture is categorically stronger than a
            # two-move deferred mill because capture happens sooner.
            _late_placement = before.pieces_placed.get(color, 0) >= 6
            if not _late_placement:
                chain = 0
            else:
                our_rem_check = 9 - after.pieces_placed.get(color, 0)
                chain = _placement_chain_scan(after, color) if our_rem_check >= 2 else 0
                if chain < 4:
                    chain = 0  # only level-4 overrides taking the available mill
        if chain >= 1:
            if chain == 1:
                _base_chain = int(weights.placement_busy_scan * 0.4)
            else:
                _base_chain = weights.placement_busy_scan * (chain - 1)
            if opp_last_weak:
                _base_chain = int(_base_chain * 1.5)
            # Extra bonus when deliberately skipping an available mill for a level-4 chain.
            # Only applies in the late-placement window (pieces 7-9): the plan examples
            # show 9-piece chains, not early-game forks where the immediate mill + capture
            # is strictly more forcing than a 2-move-deferred mill.
            if chain == 4 and _had_mill_available and before.pieces_placed.get(color, 0) >= 6:
                _base_chain += weights.defer_for_chain
            busy_chain_bonus = _base_chain

    # Convergence cluster disruption: bonus for placement that breaks an opponent
    # convergence cluster — 3 opponent pieces that can each reach a distinct square
    # in the same mill within 2 adjacency moves along unblocked paths.
    # Only fires in placement phase (movement phase already has evaluate() for this).
    conv_bonus = 0
    if get_game_phase(before, color) == "place":
        conv_before = _convergence_cluster_count(before, opp)
        if conv_before > 0:
            conv_after = _convergence_cluster_count(after, opp)
            disrupted = max(0, conv_before - conv_after)
            conv_bonus = disrupted * weights.convergence_block

    # Double-mill convergence disruption: bonus for move-phase moves that reduce the
    # number of opponent fork precursor pairs (shared closing square or shared pivot).
    # Only fires in move phase — placement phase uses the separate convergence_block path.
    dmc_bonus = 0
    if get_game_phase(before, color) == "move":
        dmc_before = _double_mill_convergence(before, opp)
        if dmc_before > 0:
            dmc_after = _double_mill_convergence(after, opp)
            disrupted = max(0, dmc_before - dmc_after)
            dmc_bonus = disrupted * weights.convergence_disrupt

    # Mobility-reduction bonus: each opponent legal move removed by this move earns
    # a direct reward in move phase.  Herding the opponent into a corner (zero moves)
    # is a win condition, so moves that tighten the noose deserve explicit credit
    # even when they don't close a mill or capture a piece.
    mob_reduction_bonus = 0
    if get_game_phase(before, color) == "move":
        opp_mob_delta = max(0, _mobility(before, opp) - _mobility(after, opp))
        mob_reduction_bonus = weights.mobility_reduction * opp_mob_delta

    # B-3 — Ring crowding: cardinal preference.
    # When the opponent has 3+ pieces on one ring, bonus for placing on a cardinal
    # square adjacent to that ring (the cross-node connectors that control exit lines).
    # Outer ring concentrated → prefer middle-ring cardinals (d6/f4/d2/b4).
    # Middle ring concentrated → prefer outer cardinals (d7/g4/d1/a4) or inner (d5/e4/d3/c4).
    ring_cardinal_bonus = 0
    if get_game_phase(before, color) == "place":
        _placed_sq = next(
            (p for p in POSITIONS if after.positions[p] == color and before.positions[p] != color),
            None,
        )
        if _placed_sq is not None and _placed_sq in _CROSS_NODES:
            opp_conc = _opponent_ring_concentration(before, opp)
            # [outer_count, middle_count, inner_count]
            _ring_adjacency = [
                _RING_MIDDLE,  # connectors for outer-ring concentration
                _RING_OUTER | _RING_INNER,  # connectors for middle-ring concentration
                _RING_MIDDLE,  # connectors for inner-ring concentration
            ]
            for ring_idx, ring_count in enumerate(opp_conc):
                if ring_count >= 3 and _placed_sq in _ring_adjacency[ring_idx]:
                    ring_cardinal_bonus = max(ring_cardinal_bonus,
                                              int(weights.cardinal_block * 0.5))
                    break

    # B-4 — Fork anticipation: bonus for blocking a square that (within 2 moves)
    # would give the opponent two simultaneous 2-configs.  Applies in placement
    # and move phase; not fly phase.
    fork_anticip_bonus = 0
    if get_game_phase(before, color) not in ("fly",):
        _fork_sqs = _fork_in_n(before, opp, 2)
        if _fork_sqs:
            _moved_to = next(
                (p for p in POSITIONS if after.positions[p] == color and before.positions[p] != color),
                None,
            )
            if _moved_to is None:
                # movement: find destination
                _moved_to = next(
                    (p for p in POSITIONS if after.positions[p] == color and before.positions[p] == ""),
                    None,
                )
            if _moved_to in _fork_sqs:
                fork_anticip_bonus = weights.fork_anticipation

    # B-7 — Locked mill escape and redirected-pin creation.
    # Neither fires in placement or fly phase.
    locked_escape_bonus = 0
    redirected_pin_bonus = 0
    if get_game_phase(before, color) == "move":
        # Determine the piece that moved (from_sq → to_sq)
        _from_sq = next(
            (p for p in POSITIONS if before.positions[p] == color and after.positions[p] != color),
            None,
        )
        _to_sq_mv = next(
            (p for p in POSITIONS if after.positions[p] == color and before.positions[p] == ""),
            None,
        )
        if _from_sq and _to_sq_mv:
            opp = "B" if color == "W" else "W"
            # Locked mill escape: piece was adjacent to a frozen opponent blocker.
            # Two blocker types, same bonus but different destination requirements:
            #
            # Case A — 2-config blocker: opponent sits on our mill's closing square.
            #   Moving it hands us a mill, so it can't leave. Destination must build
            #   a new 2-config (we're sacrificing a mill threat for freedom elsewhere).
            #
            # Case B — mill-exit blocker: opponent sits adjacent to a piece in our
            #   closed mill, guarding that exit. Any piece adjacent to the blocker —
            #   including the mill piece itself and pieces nearby like c3 — gains
            #   freedom by stepping away, since the blocker can't give chase.
            #   No destination check: moving away from a frozen exit-guard is
            #   inherently good regardless of where we land.
            for nb in ADJACENCY[_from_sq]:
                if before.positions[nb] != opp:
                    continue
                is_case_a = any(
                    nb in mill and all(before.positions[p] == color for p in mill if p != nb)
                    for mill in MILLS
                )
                is_case_b = (not is_case_a) and any(
                    nb not in mill
                    and _from_sq not in mill  # exclude the mill piece itself (that's cycling, not escaping)
                    and all(before.positions[p] == color for p in mill)
                    and any(nb in ADJACENCY[p] for p in mill)
                    for mill in MILLS
                )
                if is_case_a:
                    for m2 in MILLS:
                        if _to_sq_mv in m2:
                            vals = [after.positions[p] for p in m2]
                            if vals.count(color) == 2 and vals.count("") == 1:
                                locked_escape_bonus = weights.locked_mill_escape
                                break
                elif is_case_b:
                    locked_escape_bonus = weights.locked_mill_escape
                if locked_escape_bonus:
                    break
            # Redirected pin: move causes opponent piece to double-block two own 2-configs
            if _creates_redirected_pin(before, color, _from_sq, _to_sq_mv):
                redirected_pin_bonus = weights.redirected_pin

    # Cycling close bonus: closing a mill that immediately sets up the next cycle.
    # Fires in move phase when the newly closed mill has a free exit that is the
    # closing square of another own 2-config — the pivot can slide straight back
    # to force another capture next turn.  Uses cycling_mill weight so the slider
    # controls both building the dual-mill structure and executing the cycle.
    cycling_close_bonus = 0
    if mills_delta > 0 and get_game_phase(before, color) == "move":
        for mill in MILLS:
            if all(after.positions[p] == color for p in mill) and not all(before.positions[p] == color for p in mill):
                mill_set = set(mill)
                for p in mill:
                    for nb in ADJACENCY[p]:
                        if nb in mill_set or after.positions[nb] != "":
                            continue
                        # nb is a free exit of the newly closed mill
                        for m2 in MILLS:
                            if nb in m2:
                                vals = [after.positions[q] for q in m2]
                                if vals.count(color) == 2 and vals.count("") == 1:
                                    cycling_close_bonus = weights.cycling_mill
                                    break
                        if cycling_close_bonus:
                            break
                    if cycling_close_bonus:
                        break
                break

    # B-8 — Forked mill blocking: when opponent has 2+ fork arms, reward blocking the
    # arm with higher cycling freedom (not necessarily the cardinal arm).
    # Cardinal exception: if own piece already presses the closing square of a cardinal
    # arm, that arm's cycling is already constrained — treat it as less urgent.
    # Gate: placement and move phase only; not fly phase.
    cycling_block_bonus = 0
    if get_game_phase(before, color) not in ("fly",):
        _arms = _opponent_fork_arms(before, color)
        if len(_arms) >= 2:
            _placed_or_moved = next(
                (p for p in POSITIONS if after.positions[p] == color and before.positions[p] != color),
                None,
            )
            if _placed_or_moved is None:
                _placed_or_moved = next(
                    (p for p in POSITIONS if after.positions[p] == color and before.positions[p] == ""),
                    None,
                )
            if _placed_or_moved is not None:
                # Score each arm by cycling freedom after hypothetical closure
                def _arm_freedom(arm_mill, closing):
                    sim = dict(before.positions)
                    sim[closing] = opp
                    class _Sim:
                        positions = sim
                    return _mill_cycling_freedom(_Sim(), opp, arm_mill)

                # Adjust for cardinal exception: if own piece presses the closing sq
                # of a cardinal arm, reduce its effective freedom
                def _effective_freedom(arm_mill, closing):
                    base_f = _arm_freedom(arm_mill, closing)
                    if closing in _CROSS_NODES and _own_piece_adj_to_closing(before, color, closing):
                        base_f = max(0, base_f - 2)
                    return base_f

                arm_freedoms = [
                    (closing, _effective_freedom(m, closing))
                    for m, closing in _arms
                ]
                arm_freedoms.sort(key=lambda x: x[1], reverse=True)
                # The highest-freedom closing square is the one we WANT to block
                best_close_sq, best_f = arm_freedoms[0]
                worst_f = arm_freedoms[-1][1]
                if _placed_or_moved == best_close_sq and best_f > worst_f:
                    freedom_diff = best_f - worst_f
                    cycling_block_bonus = int(weights.block_cycling_priority * (1 + freedom_diff * 0.1))

    return (
        weights.close_mill            * mills_delta
        + weights.cycling_mill        * (cycling_gain + opp_cycle_lost)
        + cycling_close_bonus
        + weights.block_opponent_mill * blocked
        + weights.stop_opponent_mills * two_cfg_broken
        + weights.feeder_diamond      * diamond_gain
        + weights.mill_wrapping       * wrap_gain
        + weights.cardinal_block      * (our_cross_gained + opp_cross_lost)
        + scatter
        + setup_mill_bonus
        + mill_open_bonus
        + late_mill_bonus
        + trap_build_bonus
        + fly_fork_bonus
        + mob_reduction_bonus
        + capture_feeder_bonus
        + capture_diamond_bonus
        + busy_chain_bonus
        + conv_bonus
        + dmc_bonus
        + safe_capture_bonus
        + fly_free_close_bonus
        + ring_cardinal_bonus
        + fork_anticip_bonus
        + locked_escape_bonus
        + redirected_pin_bonus
        + cycling_block_bonus
        - outer_mill_penalty
        - ring_crowd_penalty
    )


# ── Endgame supplement ────────────────────────────────────────────────────────

def endgame_score(board: BoardState, color: str, endgame_state=None) -> int:
    if endgame_state is None or not endgame_state.active:
        return 0

    opp      = "B" if color == "W" else "W"
    mob_self = endgame_state.mobility_white if color == "W" else endgame_state.mobility_black
    mob_opp  = endgame_state.mobility_black if color == "W" else endgame_state.mobility_white

    score = (mob_self - mob_opp) * 20

    if mob_opp <= 2 and mob_self >= 4:
        score += 200

    if (
        endgame_state.pattern == "mill_cycle"
        and endgame_state.pattern_notes.startswith(color)
    ):
        score += 150

    return score
