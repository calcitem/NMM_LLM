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
    # ── Positional base scale (applied inside evaluate) ──────────────────
    long_term_position: int   = 100   # % multiplier on entire positional base score
    mill_count_scale: int     = 100   # % multiplier on mill-count weights
    mobility_scale: int       = 100   # % multiplier on mobility weights
    blocked_scale: int        = 100   # % multiplier on blocked-pieces weights
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
_HERD_WEIGHTS  = {"place": 2, "move": 12, "fly": 0}

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
    if phase == "fly" and opp_pieces == 4:
        base += 180 * _piece_separation(board, color)

    # Fly fork: when both players are in fly phase, each 2-config is an immediate
    # threat (closeable in 1 move).  We can block at most 1 per turn, so any
    # surplus threats the opponent holds are essentially guaranteed captures.
    # Penalise (or reward) each uncoverable surplus threat heavily.
    if phase == "fly" and get_game_phase(board, opp) == "fly":
        opp_surplus = max(0, opp_thr - 1)  # we block 1 max; remainder are automatic
        own_surplus = max(0, our_thr - 1)
        base += 900 * (own_surplus - opp_surplus)  # was 600

    # Move-phase: reward non-contributing pieces that are adjacent to a 2-config
    # piece (assembling toward a productive formation — free piece assembly).
    if phase == "move":
        base += 40 * (_free_piece_assembly(board, color) - _free_piece_assembly(board, opp))

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
    """Count pairs of own 2-configs whose empty closing squares are adjacent.

    This measures the STRUCTURAL SETUP for cycling, not the act of moving back and
    forth.  A cycling setup exists when two open mills (each needing only one more
    piece) share adjacent empty closing squares E1 and E2: one piece can slide
    E1→E2→E1 to force a capture every two turns.  The tactical bonus is only given
    when this setup is GAINED (delta > 0 between before and after the move), so
    simply moving a piece back and forth between already-existing positions scores
    zero.  In fly phase every empty square is reachable so all pairs of 2-configs
    qualify as potential cycling setups.
    """
    phase = get_game_phase(board, color)
    empties = []
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 2 and vals.count("") == 1:
            empties.append(next(p for p in mill if board.positions[p] == ""))
    count = 0
    n = len(empties)
    for i in range(n):
        for j in range(i + 1, n):
            if phase == "fly" or empties[j] in ADJACENCY[empties[i]]:
                count += 1
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
    """
    opp = "B" if color == "W" else "W"
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


def tactical_move_bonus(
    before: BoardState,
    after: BoardState,
    color: str,
    weights: HeuristicWeights | None = None,
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

    # Late-placement mill urgency (pieces 7-9, i.e. pieces_placed >= 6).
    # Closing a mill on the OUTER or MIDDLE ring in this window is critical —
    # it's likely the last chance to form a mill with good piece placement.
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

    return (
        weights.close_mill            * mills_delta
        + weights.cycling_mill        * (cycling_gain + opp_cycle_lost)
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
