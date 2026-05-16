"""
ai/heuristics.py — Kukreja-style phase-weighted board evaluation.

evaluate(board, color) returns an integer score from color's perspective.
Positive = good for color, negative = bad.
"""

from __future__ import annotations

from game.board import ADJACENCY, MILLS, POSITIONS, BoardState
from game.rules import get_game_phase, is_terminal

INF: int = 10_000_000

# Phase weights: (closed_mills, blocked_opp, piece_diff, two_config, double_mills, win_config)
_WEIGHTS = {
    "place": (14,  10, 11, 8, 1086,    0),
    "move":  (14,  43, 10, 7,   42,    0),
    "fly":   (16, 350,  1, 0,    0, 1190),
}


def evaluate(board: BoardState, color: str, endgame_state=None) -> int:
    """Evaluate board from `color`'s perspective. Higher is better for color."""
    terminal, winner = is_terminal(board)
    if terminal:
        return INF if winner == color else -INF

    opp = "B" if color == "W" else "W"
    phase = get_game_phase(board, color)
    w = _WEIGHTS[phase]

    our_mills  = _closed_mills(board, color)
    opp_mills  = _closed_mills(board, opp)
    blocked    = _blocked_count(board, opp)
    piece_diff = board.pieces_on_board[color] - board.pieces_on_board[opp]
    our_two    = _two_configs(board, color)
    opp_two    = _two_configs(board, opp)
    our_dbl    = _double_mills(board, color)
    opp_dbl    = _double_mills(board, opp)
    win_cfg    = _win_config(board, opp)

    base = (
        w[0] * (our_mills  - opp_mills)
        + w[1] * blocked
        + w[2] * piece_diff
        + w[3] * (our_two  - opp_two)
        + w[4] * (our_dbl  - opp_dbl)
        + w[5] * win_cfg
    )
    return base + endgame_score(board, color, endgame_state)


def _closed_mills(board: BoardState, color: str) -> int:
    return sum(
        1 for mill in MILLS
        if all(board.positions[p] == color for p in mill)
    )


def _blocked_count(board: BoardState, color: str) -> int:
    """Count pieces of `color` with no legal adjacent empty square. Fly-phase pieces are never blocked."""
    if get_game_phase(board, color) == "fly":
        return 0
    count = 0
    for pos in POSITIONS:
        if board.positions[pos] == color:
            if all(board.positions[n] != "" for n in ADJACENCY[pos]):
                count += 1
    return count


def _two_configs(board: BoardState, color: str) -> int:
    """Count mills with exactly 2 squares of `color` and 1 empty (open mill potential)."""
    count = 0
    for mill in MILLS:
        vals = [board.positions[p] for p in mill]
        if vals.count(color) == 2 and vals.count("") == 1:
            count += 1
    return count


def _double_mills(board: BoardState, color: str) -> int:
    """Count pieces of `color` simultaneously part of 2+ closed mills (double-mill pivot pieces)."""
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
    """Return 1 if opponent is in fly phase (≤3 pieces after placing all 9) — a near-winning state."""
    return int(board.pieces_placed[opp] == 9 and board.pieces_on_board[opp] <= 3)


def endgame_score(board: BoardState, color: str, endgame_state=None) -> int:
    """
    Supplementary endgame evaluation added on top of evaluate().

    Rewards:
    - Mobility advantage  — more moves available than the opponent
    - Zugzwang pressure   — opponent has very few moves (≤ 2)
    - Mill cycle          — a closed mill can be opened/closed at will
    """
    if endgame_state is None or not endgame_state.active:
        return 0

    opp = "B" if color == "W" else "W"
    mob_self = endgame_state.mobility_white if color == "W" else endgame_state.mobility_black
    mob_opp  = endgame_state.mobility_black if color == "W" else endgame_state.mobility_white

    score = 0

    # Mobility advantage (each extra legal move is worth ~20 points in endgame)
    score += (mob_self - mob_opp) * 20

    # Zugzwang pressure on the opponent
    if mob_opp <= 2 and mob_self >= 4:
        score += 200

    # Mill cycle controlled by us
    if (
        endgame_state.pattern == "mill_cycle"
        and endgame_state.pattern_notes.startswith(color)
    ):
        score += 150

    return score
