"""
ai/heuristics.py — Phase-weighted board evaluation for Nine Men's Morris.

evaluate(board, color) returns an integer score from color's perspective.
Positive = good for color, negative = bad.
"""

from __future__ import annotations
import math
from game.board import ADJACENCY, MILLS, POSITIONS, BoardState
from game.rules import get_game_phase, is_terminal

INF: int = 10_000_000

# Phase weights: (closed_mills, blocked_opp, piece_diff, two_cfg, dbl_mill, win_cfg)
# Classic Kukreja weights — double-mill pivot (1086) only in move phase, not place.
_WEIGHTS = {
    "place": (14,  10, 11, 8,    0,    0),
    "move":  (14,  43, 10, 7,   42,    0),
    "fly":   (16, 350,  1, 0,    0, 1190),
}

# Mobility and threat term weights per phase
_MOB_WEIGHTS    = {"place": 3,  "move": 8,  "fly": 20}
_THREAT_WEIGHTS = {"place": 8,  "move": 12, "fly": 18}

# tanh normalization scales per phase (used by position_eval display, not search)
TANH_SCALE = {"place": 120, "move": 180, "fly": 280}

# Cross/cardinal nodes have 3 neighbours → more mobile and flexible
_CROSS_NODES = frozenset({
    "d7", "d6", "d5",
    "g4", "f4", "e4",
    "d1", "d2", "d3",
    "a4", "b4", "c4",
})

# Mill-cycle readiness: a closed mill with a slide-out square enables repeated
# captures (open/close each cycle).  Highest value in fly; still relevant in move.
_CYCLE_WEIGHTS = {"place": 8, "move": 22, "fly": 45}

# Fork-threat: a piece in 2+ open mills simultaneously.  Opponent cannot defend
# both in one move, so one mill closes next turn regardless.
_FORK_WEIGHTS  = {"place": 6, "move": 14, "fly": 28}

# Herding / encirclement: own pieces adjacent to each opponent piece.
# Rewards progressively surrounding opponent pieces to shrink their escape space.
# Irrelevant in fly phase (pieces can jump anywhere).
_HERD_WEIGHTS  = {"place": 2, "move": 9,  "fly": 0}

# Fly-phase asymmetry: reward entering fly (3 pieces) when the opponent hasn't yet,
# and penalise giving the opponent fly while we remain in move phase.
# At 4v4 the search will prefer sacrificing a piece (3v4, us in fly) over
# capturing an opponent piece (4v3, them in fly).
_FLY_ASYM_WEIGHTS = {"place": 0, "move": 80, "fly": 0}


def evaluate(
    board: BoardState,
    color: str,
    endgame_state=None,
    force_aggressive: bool = False,
) -> int:
    """Evaluate board from `color`'s perspective. Higher is better for color."""
    terminal, winner = is_terminal(board)
    if terminal:
        return INF if winner == color else -INF

    opp   = "B" if color == "W" else "W"
    phase = get_game_phase(board, color)
    w     = _WEIGHTS[phase]

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

    base = (
        w[0] * (our_mills - opp_mills)
        + w[1] *  blocked
        + w[2] *  piece_diff
        + w[3] * (our_two  - opp_two)
        + w[4] * (our_dbl  - opp_dbl)
        + w[5] *  win_cfg
        + _MOB_WEIGHTS[phase]    * (our_mob - opp_mob)
        + _THREAT_WEIGHTS[phase] * (our_thr - opp_thr)
        + 2 * (our_pos - opp_pos)
        + _CYCLE_WEIGHTS[phase]  * (our_cycle - opp_cycle)
        + _FORK_WEIGHTS[phase]   * (our_fork  - opp_fork)
        + _HERD_WEIGHTS[phase]   * (our_herd  - opp_herd)
        + _FLY_ASYM_WEIGHTS[phase] * fly_asym
    )
    return base + endgame_score(board, color, endgame_state)


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
    """Count 2-piece open mills (can be closed in one move)."""
    count = 0
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 2 and vals.count("") == 1:
            count += 1
    return count


def _position_value(board: BoardState, color: str) -> int:
    """Sum of positional scores: cross nodes = 3, corner nodes = 2."""
    total = 0
    for pos in POSITIONS:
        if board.positions[pos] == color:
            total += 3 if pos in _CROSS_NODES else 2
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
    -1 if opponent has fly and color does not.
    0 if both or neither are in fly.

    Rewards sacrificing down to 3 pieces to gain fly mobility before the opponent
    does, and penalises giving the opponent fly (e.g. capturing their 4th piece
    in a 4v4 position to leave them with 3 = fly advantage).
    """
    opp = "B" if color == "W" else "W"
    color_fly = (board.pieces_placed.get(color, 0) >= 9 and board.pieces_on_board[color] == 3)
    opp_fly   = (board.pieces_placed.get(opp,   0) >= 9 and board.pieces_on_board[opp]   == 3)
    if color_fly and not opp_fly:
        return 1
    if opp_fly and not color_fly:
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
